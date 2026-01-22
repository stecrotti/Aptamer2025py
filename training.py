import torch
from utils import one_hot
import selex_distribution
from callback import ConvergenceMetricsCallback
import sampling

def init_chains(
    n_rounds: int,
    n_chains: int,
    L: int,
    q: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:

    if dtype is None:
        dtype = torch.float32
    chains = [torch.randint(low=0, high=q, size=(n_chains, L), device=device)
            for _ in range(n_rounds)]
    
    chains_tensor = torch.stack([one_hot(c, num_classes=q).to(dtype=dtype, device=device) for c in chains])
    
    return chains_tensor   # n_rounds, n_chains, L, q

def update_chains_default():
    def update_chains(chains, t, model, n_sweeps):
        # model.sample_metropolis_uniform_sites(chains, t, n_sweeps)
        return sampling.sample_metropolis(model, chains, t, n_sweeps)
    return update_chains

def compute_moments_model_at_round(model, chains, t):
    # this L is not physically a likelihood, more like a computational trick
    return model.compute_energy_up_to_round(chains, t).mean()

def compute_moments_data_at_round(model, data_batch, t):
    # this L is not physically a likelihood, more like a computational trick
    return model.compute_energy_up_to_round(data_batch, t).mean()

def compute_grad_model(model, L_model, retain_graph):
    params = tuple(model.parameters())

    grad_model = torch.autograd.grad(
        outputs=L_model,
        inputs=params,
        retain_graph=retain_graph,
        create_graph=False
    )

    return grad_model

def compute_grad_data(model, L_data, retain_graph):
    params = tuple(model.parameters())

    grad_data = torch.autograd.grad(
        outputs=L_data,
        inputs=params,
        retain_graph=retain_graph,
        create_graph=False
    )

    return grad_data

def compute_total_gradient(model, grad_model, grad_data):
    grad_total = tuple(-(gm - ge) for gm, ge in zip(grad_model, grad_data))
    
    with torch.no_grad():
        for p, g in zip(model.parameters(), grad_total):
            p.grad = g

    return grad_total


def train(
    model: selex_distribution.MultiRoundDistribution,
    data_loaders,
    total_reads,
    chains: torch.Tensor,
    n_sweeps: int,
    lr: float,    
    max_epochs: int,
    target_pearson = 0.999,
    thresh_slope = 1e-2,
    l2reg: float = 0.0,
    log_weights: torch.Tensor | None = None,
    callbacks = [ConvergenceMetricsCallback()],
    update_chains = update_chains_default()
):
    n_rounds = len(chains)
    ts = torch.arange(n_rounds)
    assert chains.shape[0] == n_rounds
    normalized_total_reads = total_reads / total_reads.sum(0, keepdim=True)

    n_chains = chains.shape[1] 
    device=chains.device
    dtype=chains.dtype
    log_n_chains = torch.log(torch.tensor(n_chains, device=device, dtype=dtype)).item()
    energies_AIS = [model.compute_energy_up_to_round(chains[t], t) for t in ts]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2reg)

    log_likelihood = - log_n_chains
    if log_weights is None:
        log_weights = torch.zeros(n_rounds, n_chains, device=device, dtype=dtype)

    epochs = 0   
    converged = (epochs > max_epochs)
    for callback in callbacks:
        callback.before_training(max_epochs=max_epochs)
        
    while not converged:
        optimizer.zero_grad()
        L_model = L_data = 0
        log_likelihood = 0.0
        for t in ts:
            # update chains and log weights for estimate of normalization
            with torch.no_grad():
                energy_model_t = update_chains(chains, t, model, n_sweeps)
                log_weights[t] += energies_AIS[t] - energy_model_t
                energies_AIS[t] = energy_model_t
            # compute gradient
            L_m = compute_moments_model_at_round(model, chains[t].clone(), t)
            L_model = L_model + normalized_total_reads[t] * L_m
            
            # extract batch of data from round t
            data_batch = data_loaders[t].get_batch()
            L_d = compute_moments_data_at_round(model, data_batch, t)
            L_data = L_data + normalized_total_reads[t] * L_d
            logZt = (torch.logsumexp(log_weights[t], dim=0)).item() - log_n_chains
            log_likelihood += (normalized_total_reads[t] * (L_d.detach().clone() - logZt)).item()
        
            # TODO: compute round-wise convergence metrics

        # Compute gradient
        grad_model = compute_grad_model(model, L_model, retain_graph=True)
        # grad_model = compute_grad_model_indep(model)
        # grad_data = grad_data_cached
        grad_data = compute_grad_data(model, L_data, retain_graph=False)
        grad_total = compute_total_gradient(model, grad_model, grad_data)
        # do gradient step on params
        optimizer.step()

        epochs += 1
        converged = (epochs > max_epochs)

        # callbacks
        for callback in callbacks:
            c = callback.after_step(model=model, chains=chains, total_reads=total_reads, 
                         log_likelihood = log_likelihood, epochs=epochs,
                         grad_model=grad_model, grad_data=grad_data, grad_total=grad_total,
                         target_pearson=target_pearson, thresh_slope=thresh_slope)
            converged = converged or c