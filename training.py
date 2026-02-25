import torch
from utils import one_hot
import selex_distribution
from callback import ConvergenceMetricsCallback
import sampling
import math
import utils
import pathlib
import glob
import os
import matplotlib.pyplot as plt

def init_chains(
    n_rounds: int,
    n_chains: int,
    L: int,
    q: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    fi = None
) -> torch.Tensor:

    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cpu')

    if fi is None:
        chains = [torch.randint(low=0, high=q, size=(n_chains, L), device=device)
                for _ in range(n_rounds)]
    else:
        assert len(fi) == n_rounds
        chains = [torch.multinomial(fi[t], num_samples=n_chains, replacement=True).T
              for t in range(n_rounds)]
    
    chains_tensor = torch.stack([one_hot(c, num_classes=q).to(dtype=dtype, device=device) for c in chains])
    
    return chains_tensor   # n_rounds, n_chains, L, q

def update_chains_default():
    def update_chains(chains, t, model, n_sweeps):
        return sampling.sample_metropolis(model, chains, t, n_sweeps)
    return update_chains

def compute_moments_at_round(model, x, t):
    # this L is not physically a likelihood, more like a computational trick
    return - model.compute_energy_up_to_round(x, t).mean()

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

def load_checkpoints(checkpoint_filename):
    cps = []
    dirpath = pathlib.Path(__file__).parent.resolve() / f'experiments/checkpoints/{checkpoint_filename}'
    dirpath = str(dirpath)
    files = glob.glob(dirpath + f'/{checkpoint_filename}*.pt')

    for file in sorted(files, key=os.path.getmtime):
        cp = torch.load(file, weights_only=False)
        cps.append(cp)

    print(f'Loaded {len(cps)} files.')
    
    return cps


def train(
    model: selex_distribution.MultiRoundDistribution,
    data_loaders,
    total_reads: torch.IntTensor,
    chains: torch.Tensor,
    n_sweeps: int,   
    max_epochs: int,
    target_pearson = 1.0,
    thresh_slope = 0.0,
    l2reg: float = 0.0,
    log_weights: torch.Tensor | None = None,
    optimizer = None,
    lr = 1e-2, 
    data_loaders_valid = None,
    total_reads_valid = None,
    log_multinomial_factors = None,
    log_multinomial_factors_valid = None,
    callbacks = [ConvergenceMetricsCallback()],
    update_chains = update_chains_default()
):
    device = chains.device
    dtype = chains.dtype
    n_rounds, n_chains, L, q = chains.size()
    ts = torch.arange(n_rounds, device=device)
    assert chains.shape[0] == n_rounds

    if log_multinomial_factors is None:
        log_multinomial_factors = torch.zeros(n_rounds)

    log_likelihood_normaliz = (total_reads + log_multinomial_factors).sum().item()

    # log_n_chains = torch.log(torch.tensor(n_chains, device=device, dtype=dtype)).item()
    energies_AIS = [torch.zeros_like(chains[t]) for t in ts]
    Llogq = L * torch.log(torch.tensor(q, device=device, dtype=dtype)).item()

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2reg)

    log_likelihood = torch.nan
    if log_weights is None:
        log_weights = torch.zeros(n_rounds, n_chains, dtype=dtype, device=device)

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
                L_m = compute_moments_at_round(model, chains[t].clone(), t)
                L_model += total_reads[t] * L_m / log_likelihood_normaliz
                
                # extract batch of data from round t
                data_batch = batches[t]
                L_d = compute_moments_at_round(model, data_batch, t)
                L_data += total_reads[t] * L_d / log_likelihood_normaliz
                # logZt = Llogq + (torch.logsumexp(log_weights[t], dim=0)).item() - log_n_chains
                # Lt = L_d.detach().clone() - logZt
                # log_likelihood += (log_multinomial_factors[t] + total_reads[t] * Lt).item()
            # log_likelihood /= log_likelihood_normaliz

            # Compute gradient
            grad_model = compute_grad_model(model, L_model, retain_graph=True)
            grad_data = compute_grad_data(model, L_data, retain_graph=False)
            grad_total = compute_total_gradient(model, grad_model, grad_data)
            # do gradient step on params
            optimizer.step()

            if data_loaders_valid is not None:
                if log_multinomial_factors_valid is None:
                    raise ValueError("Validation set was provided, but not the corresponding log-multinomial factors.")
                batches_valid = [next(iter(dl)) for dl in data_loaders_valid]
                log_likelihood_valid = estimate_log_likelihood(model, batches_valid, total_reads_valid, log_weights,
                                                               log_multinomial_factors_valid)
            else:
                log_likelihood_valid = None

            with torch.no_grad():
                # update logweights for importance sampling estimate of normalization constant
                for t in ts:
                    energy_new = model.compute_energy_up_to_round(chains[t], t)
                    log_weights[t] += energies_AIS[t] - energy_new
                log_likelihood = estimate_log_likelihood(model, batches, total_reads, log_weights, log_multinomial_factors)

                epochs += 1
                converged = (epochs > max_epochs)

                # callbacks
                for callback in callbacks:
                    c = callback.after_step(model=model, chains=chains, total_reads=total_reads, 
                                data_loaders=data_loaders, log_likelihood_valid=log_likelihood_valid, 
                                og_multinomial_factors=log_multinomial_factors,
                                data_loaders_valid=data_loaders_valid, total_reads_valid=total_reads_valid,
                                log_likelihood = log_likelihood, epochs=epochs,
                                grad_model=grad_model, grad_data=grad_data, grad_total=grad_total,
                                target_pearson=target_pearson, thresh_slope=thresh_slope,
                                optimizer=optimizer, log_weights=log_weights,
                                log_multinomial_factors_valid=log_multinomial_factors_valid)
                converged = converged or c
                
            if converged:
                model.zero_grad()
                return

    model.zero_grad()

@torch.no_grad
def estimate_logprobability_up_to_round(model, x: torch.tensor, t, log_weights: torch.tensor):
    batch_size, L, q = x.size()
    en = model.compute_energy_up_to_round(x, t).mean().item()
    Llogq = L * math.log(q)
    logZt = Llogq - math.log(len(log_weights)) + (torch.logsumexp(log_weights, dim=0)).item()
    logp = - en - logZt
    return logp

@torch.no_grad
def estimate_log_likelihood(model, batches, total_reads, log_weights, log_multinomial_factors):
    n_rounds = len(batches)
    assert len(log_weights) == n_rounds
    log_likelihood_normaliz = (total_reads + log_multinomial_factors).sum().item()
    log_likelihood = 0.0
    for t in range(n_rounds):
        Lt = estimate_logprobability_up_to_round(model, batches[t], t, log_weights[t])
        log_likelihood += (log_multinomial_factors[t] + total_reads[t] * Lt).item()

    return log_likelihood / log_likelihood_normaliz

@torch.no_grad
def compute_weights_AIS(model, batches, n_chains, n_sweeps, step):
    x = batches[0][0]
    L, q = x.size()
    dtype = x.dtype
    device = x.device
    chains = init_chains(len(batches), n_chains, L, q, dtype=dtype, device=device)
    beta_schedule = torch.arange(step, 1+step, step).to(dtype=dtype, device=device)
    _, log_weights = sampling.estimate_normalizations(model, chains, n_sweeps, beta_schedule)

    return log_weights


@torch.no_grad
def estimate_log_likelihood_AIS(model, batches, total_reads, log_multinomial_factors, 
                                n_chains, n_sweeps, step):
    log_weights = compute_weights_AIS(model, batches, n_chains, n_sweeps, step)
    return estimate_log_likelihood(model, batches, total_reads, log_weights, log_multinomial_factors)

def scatter_moments(model, data_loaders, chains, total_reads, log_likelihood_normaliz=1, **kwargs):
    batches = [next(iter(dl)) for dl in data_loaders]
    n_rounds = len(batches)
    
    L_data = 0
    model.zero_grad()
    for t in range(n_rounds):
        L_d = - model.compute_energy_up_to_round(batches[t].clone(), t).mean()
        L_data += total_reads[t] * L_d / log_likelihood_normaliz
    grad_data = compute_grad_data(model, L_data, retain_graph=False)
    
    L_model = 0
    model.zero_grad()
    for t in range(n_rounds):
        L_m = - model.compute_energy_up_to_round(chains[t].clone(), t).mean()
        L_model += total_reads[t] * L_m / log_likelihood_normaliz
    grad_model = compute_grad_data(model, L_model, retain_graph=False)
    
    n_param = len(list(model.named_parameters()))
    
    fig, axes = plt.subplots(1, n_param, **kwargs)
    for (i, (param_name, _)) in enumerate(model.named_parameters()):
        ax = axes[i]
        x = grad_model[i].cpu()
        y = grad_data[i].cpu()
        if param_name.endswith('.J'):
            x = utils.off_diagonal_terms(x)
            y = utils.off_diagonal_terms(y)
        ax.scatter(x, y)
        ax.plot(x.reshape(-1), x.reshape(-1), label='identity', color='gray', ls='--')
        ax.legend()
        ax.set_xlabel('Moments model')
        ax.set_ylabel('Moments data')
        ax.set_title(param_name)

    fig.suptitle('Moment matching (units on axes up to const factor)')
    fig.tight_layout()

    return grad_model, grad_data, fig, axes