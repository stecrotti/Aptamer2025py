import torch
from utils import one_hot
import selex_distribution
from callback import ConvergenceMetricsCallback
import sampling
import copy

def init_chains(
    n_rounds: int,
    n_chains: int,
    L: int,
    q: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:

    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cpu')

    chains = [torch.randint(low=0, high=q, size=(n_chains, L), device=device)
            for _ in range(n_rounds)]
    chains_tensor = torch.stack([one_hot(c, num_classes=q).to(dtype=dtype, device=device) for c in chains])
    
    return chains_tensor   # n_rounds, n_chains, L, q

def update_chains_default():
    def update_chains(chains, t, model, n_sweeps):
        return sampling.sample_metropolis(model, chains, t, n_sweeps)
    return update_chains

def compute_moments_model_at_round(model, chains, t):
    # this L is not physically a likelihood, more like a computational trick
    return - model.compute_energy_up_to_round(chains, t).mean()

def compute_moments_data_at_round(model, data_batch, t):
    # this L is not physically a likelihood, more like a computational trick
    return - model.compute_energy_up_to_round(data_batch, t).mean()

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
    # minus because we want gradient of *negative* loglikelihood
    grad_total = tuple(-(gd - gm) for gm, gd in zip(grad_model, grad_data))
    
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
    max_epochs: int,
    target_pearson = 0.99,
    thresh_slope = 1e-2,
    l2reg: float = 0.0,
    log_weights: torch.Tensor | None = None,
    optimizer = None,
    lr = 1e-2, 
    callbacks = [ConvergenceMetricsCallback()],
    update_chains = update_chains_default()
):
    device = chains.device
    dtype = chains.dtype
    n_rounds, n_chains, L, q = chains.size()
    ts = torch.arange(n_rounds, device=device)
    assert chains.shape[0] == n_rounds
    normalized_total_reads = total_reads / total_reads.sum(0, keepdim=True)

    log_n_chains = torch.log(torch.tensor(n_chains, device=device, dtype=dtype)).item()
    energies_AIS = [model.compute_energy_up_to_round(chains[t], t) for t in ts]
    model_prev = copy.deepcopy(model)
    Llogq = L * torch.log(torch.tensor(q, device=device, dtype=dtype)).item()

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2reg)

    log_likelihood = torch.nan
    if log_weights is None:
        log_weights = torch.full((n_rounds, n_chains), Llogq, device=device, dtype=dtype)

    epochs = 0   
    converged = (epochs > max_epochs)
    for callback in callbacks:
        callback.before_training(model=model, max_epochs=max_epochs)

    model.train()  
    while not converged:
        for batches in zip(*[iter(dl) for dl in data_loaders]):
            optimizer.zero_grad()
            L_model = L_data = 0
            log_likelihood = 0.0
            for t in ts:
                # update chains and log weights for estimate of normalization
                with torch.no_grad():
                    energies_AIS[t] = update_chains(chains, t, model, n_sweeps)
                # compute gradient
                L_m = compute_moments_model_at_round(model, chains[t].clone(), t)
                L_model = L_model + normalized_total_reads[t] * L_m
                
                # extract batch of data from round t
                data_batch = batches[t]
                L_d = compute_moments_data_at_round(model, data_batch, t)
                L_data = L_data + normalized_total_reads[t] * L_d
                logZt = Llogq + (torch.logsumexp(log_weights[t], dim=0)).item() - log_n_chains
                Lt = L_d.detach().clone() - logZt
                log_likelihood += (normalized_total_reads[t] * Lt).item()

            # Compute gradient
            grad_model = compute_grad_model(model, L_model, retain_graph=True)
            grad_data = compute_grad_data(model, L_data, retain_graph=False)
            grad_total = compute_total_gradient(model, grad_model, grad_data)
            # do gradient step on params
            optimizer.step()

            # update logweights for importance sampling estimate of Z
            for t in ts:
                energy_prev = model_prev.compute_energy_up_to_round(chains[t], t)
                log_weights[t] += energy_prev - energies_AIS[t]
            model_prev = copy.deepcopy(model)

            epochs += 1
            converged = (epochs > max_epochs)

            # callbacks
            for callback in callbacks:
                c = callback.after_step(model=model, chains=chains, total_reads=total_reads, 
                            data_loaders=data_loaders, model_prev=model_prev,
                            log_likelihood = log_likelihood, epochs=epochs,
                            grad_model=grad_model, grad_data=grad_data, grad_total=grad_total,
                         target_pearson=target_pearson, thresh_slope=thresh_slope)
            converged = converged or c
            
            if converged:
                model.zero_grad()
                return

    model.zero_grad()