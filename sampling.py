import torch

def _sample_metropolis(model, chains, t, n_steps, beta = 1.0):
    B, L, q = chains.shape
    device = chains.device
    e_current = model.compute_energy_up_to_round(chains, t)

    for _ in range(n_steps):
        proposal_flip_indices = torch.randint(0, L, (B,), device=device)
        proposal_new_tokens = torch.randint(0, q, (B,), device=device)
        
        # Flatten to (B, N*q), apply changes, then reshape
        chains_flat = chains.view(B, -1)
        
        # Zero out the entire position (q consecutive values)
        zero_indices = (proposal_flip_indices * q).unsqueeze(1) + torch.arange(q, device=device).unsqueeze(0)
        chains_flat_zeroed = chains_flat.scatter(1, zero_indices, 0)
        
        # Set the new token
        new_token_indices = (proposal_flip_indices * q + proposal_new_tokens).unsqueeze(1)
        chains_flat_proposal = chains_flat_zeroed.scatter(1, new_token_indices, 1)
        
        chains_proposal = chains_flat_proposal.view(B, L, q)
        e_proposal = model.compute_energy_up_to_round(chains_proposal, t)

        metropolis_acceptance_mask = torch.log(torch.rand(B, device=device)) < beta * (e_current - e_proposal)

        chains = torch.where(metropolis_acceptance_mask.view(B, 1, 1), chains_proposal, chains)
        e_current = torch.where(metropolis_acceptance_mask, e_proposal, e_current)

    return chains, e_current


def sample_metropolis(model, chains, t, n_sweeps, beta=1.0):
    L = chains.size(2)
    n_steps = n_sweeps * L
    with torch.no_grad():
        chains_t = chains.select(0, t)
        chains_t, energies = _sample_metropolis(model, chains_t, t, n_steps, beta)
        chains[t] = chains_t
    return energies