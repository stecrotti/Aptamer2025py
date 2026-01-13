import torch
from adabmDCA.functional import one_hot

@torch.compile
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
    idx = torch.randint(0, L, (1,), device=chains.device)[0]
    res_curr = chains[:, idx, :]
    # Propose a random new residue
    res_new = one_hot(torch.randint(0, q, (n_chains,), device=chains.device), num_classes=q).to(dtype)
    chains_new = chains.clone()
    idx_expanded = idx.view(1, 1, 1).expand(n_chains, 1, q)
    chains_new.scatter_(1, idx_expanded, res_new.unsqueeze(1))
    energy_new = compute_energy(chains_new) # shape (n_chains,)
    delta_energy = energy_new - energy_curr
    accept_prob = torch.exp(- beta * delta_energy).unsqueeze(-1)
    accepted = accept_prob > torch.rand((n_chains, 1), device=device, dtype=dtype)
    chains.masked_scatter_(accepted.unsqueeze(-1), chains_new)
    # TODO: use randexp

def sample_metropolis_uniform_sites(
    chains: torch.Tensor,
    compute_energy,
    n_sweeps: int,
    beta: float = 1.0,
    # random number generator!
):
    L = chains.size(1)
    n_steps = n_sweeps * L
    for step in torch.arange(n_steps):
        metropolis_step_uniform_sites(chains, compute_energy, beta)