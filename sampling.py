import torch
from tqdm.autonotebook import tqdm
from adabmDCA.dca import get_seqid_stats

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


# Copied from https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/resampling.py
def compute_mixing_time(
    model,
    t,
    data: torch.Tensor,
    n_max_sweeps: int,
    beta: float,
):
    """Computes the mixing time using the t and t/2 method. The sampling will halt when the mixing time is reached or
    the limit of `n_max_sweeps` sweeps is reached.

    Args:
        sampler (Callable): Sampling function.
        data (torch.Tensor): Initial data.
        params (Dict[str, torch.Tensor]): Parameters for the sampling.
            - "bias": Tensor of shape (L, q) - local biases.
            - "coupling_matrix": Tensor of shape (L, q, L, q) - coupling matrix.
        n_max_sweeps (int): Maximum number of sweeps.
        beta (float): Inverse temperature for the sampling.

    Returns:
        Dict[str, List[Union[float, int]]]: Results of the mixing time analysis.
            - "seqid_t": List of average sequence identities at time t.
            - "std_seqid_t": List of standard deviations of sequence identities at time t.
            - "seqid_t_t_half": List of average sequence identities between t and t/2.
            - "std_seqid_t_t_half": List of standard deviations of sequence identities between t and t/2.
            - "t_half": List of t/2 values (integers).
    """

    torch.manual_seed(0)
    
    n_chains, L, q = data.size()
    # Initialize chains at random
    sample_t = data
    # Copy sample_t to a new variable sample_t_half
    sample_t_half = sample_t.clone()

    # Initialize variables
    results = {
        "seqid_t": [],
        "std_seqid_t": [],
        "seqid_t_t_half": [],
        "std_seqid_t_t_half": [],
        "t_half": [],
    }

    # Loop through sweeps
    pbar = tqdm(
        total=n_max_sweeps,
        colour="red",
        dynamic_ncols=True,
        leave=False,
        ascii="-#",
    )
    pbar.set_description("Iterating until the mixing time is reached")
        
    for i in range(1, n_max_sweeps + 1):
        pbar.update(1)
        # Set the seed to i
        torch.manual_seed(i)
        # Perform a sweep on sample_t
        sample_metropolis(model, chains=sample_t, t=t, n_sweeps=1, beta=beta)
        # sample_t = sampler(chains=sample_t, params=params, nsweeps=1, beta=beta)

        if i % 2 == 0:
            # Set the seed to i/2
            torch.manual_seed(i // 2)
            # Perform a sweep on sample_t_half
            sample_metropolis(model, chains=sample_t_half, t=t, n_sweeps=1, beta=beta)
            # sample_t_half = sampler(chains=sample_t_half, params=params, nsweeps=1, beta=beta)

            # Calculate the average distance between sample_t and itself shuffled
            perm = torch.randperm(len(sample_t), device=sample_t.device)
            seqid_t, std_seqid_t = get_seqid_stats(sample_t, sample_t[perm])
            seqid_t, std_seqid_t = seqid_t / L, std_seqid_t / L

            # Calculate the average distance between sample_t and sample_t_half
            seqid_t_t_half, std_seqid_t_t_half = get_seqid_stats(sample_t, sample_t_half)
            seqid_t_t_half, std_seqid_t_t_half = seqid_t_t_half / L, std_seqid_t_t_half / L

            # Store the results
            results["seqid_t"].append(seqid_t.item())
            results["std_seqid_t"].append(std_seqid_t.item())
            results["seqid_t_t_half"].append(seqid_t_t_half.item())
            results["std_seqid_t_t_half"].append(std_seqid_t_t_half.item())
            results["t_half"].append(i // 2)

            # Check if they have crossed
            if torch.abs(seqid_t - seqid_t_t_half) / torch.sqrt(std_seqid_t**2 + std_seqid_t_t_half**2) < 0.1:
                break

        if i == n_max_sweeps:
            print(f"Mixing time not reached within {n_max_sweeps // 2} sweeps.")
            
    pbar.close()

    return results