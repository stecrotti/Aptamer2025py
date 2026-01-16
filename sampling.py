import torch
from utils import one_hot

def metropolis_step_uniform_sites(
    chains: torch.Tensor,
    compute_energy,
    beta: float = 1.0
    # random number generator!
):
    energy_curr = compute_energy(chains)
    device = chains.device
    dtype = chains.dtype
    n_chains, L, q = chains.shape

    idx = torch.randint(0, L, (1,), device=device)[0]
    res_new = one_hot(torch.randint(0, q, (n_chains,), device=device), num_classes=q).to(dtype)
    chains_new = chains.clone()
    chains_new[:, idx, :] = res_new

    energy_new = compute_energy(chains_new)
    delta_energy = energy_new - energy_curr
    accept_prob = torch.exp(-beta * delta_energy)
    rand_vals = torch.rand(n_chains, device=device, dtype=dtype)
    accepted = accept_prob > rand_vals

    # Only update accepted chains at the chosen site
    chains[accepted, idx, :] = res_new[accepted]

def sample_metropolis_uniform_sites(
    chains: torch.Tensor,
    compute_energy,
    n_sweeps: int,
    beta: float = 1.0,
    # random number generator!
):
    L = chains.size(1)
    n_steps = n_sweeps * L
    with torch.no_grad():
        for step in torch.arange(n_steps):
            metropolis_step_uniform_sites(chains, compute_energy, beta)