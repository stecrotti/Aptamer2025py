import torch
from tqdm.autonotebook import tqdm
from adabmDCA.dca import get_seqid_stats
import matplotlib.pyplot as plt
import numpy as np
import utils

@torch.no_grad
def _sample_metropolis(chains, compute_energy, n_steps, beta = 1.0):
    B, L, q = chains.shape
    device = chains.device
    e_current = compute_energy(chains)

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
        e_proposal = compute_energy(chains_proposal)

        metropolis_acceptance_mask = torch.log(torch.rand(B, device=device)) < beta * (e_current - e_proposal)

        chains = torch.where(metropolis_acceptance_mask.view(B, 1, 1), chains_proposal, chains)
        e_current = torch.where(metropolis_acceptance_mask, e_proposal, e_current)

    return chains, e_current


def sample_metropolis(model, chains, t, n_sweeps, beta=1.0):
    L = chains.size(2)
    n_steps = n_sweeps * L
    with torch.no_grad():
        chains_t = chains.select(0, t)
        compute_energy = lambda x : model.compute_energy_up_to_round(x, t)
        chains_t, energies = _sample_metropolis(chains_t, compute_energy, n_steps, beta)
        chains[t] = chains_t
    return energies


# Copied from https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/resampling.py
def compute_mixing_time(
    model,
    t,
    data: torch.Tensor,
    n_max_sweeps: int,
    beta: float = 1.0,
    pbar_desc: str = "Iterating until mixing time is reached"
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
    
    n_rounds, n_chains, L, q = data.size()
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
    pbar.set_description(pbar_desc)
        
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
            perm = torch.randperm(len(sample_t[t]), device=sample_t.device)
            seqid_t, std_seqid_t = get_seqid_stats(sample_t[t], sample_t[t][perm])
            seqid_t, std_seqid_t = seqid_t / L, std_seqid_t / L

            # Calculate the average distance between sample_t and sample_t_half
            seqid_t_t_half, std_seqid_t_t_half = get_seqid_stats(sample_t[t], sample_t_half[t])
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

def compute_and_plot_mixing_time(model, chains, n_max_sweeps = 10**3):
    n_rounds, n_chains, L, q = chains.size()
    res = []
    for t in range(n_rounds):
        res_t = compute_mixing_time(model, t, chains, n_max_sweeps, 
                                    pbar_desc = f"Computing mixing time for round {t}")
        res.append(res_t)

    fig, axes = plt.subplots(1, n_rounds, figsize=(12,4))
    for t in range(n_rounds):
        ax = axes[t]
        results = res[t]
        
        ax.plot(results["t_half"], results["seqid_t"], label=r"SeqID$(t)$", color="#002CFF")  # Blue
        ax.fill_between(results["t_half"],
                        np.array(results["seqid_t"]) - np.array(results["std_seqid_t"]),
                        np.array(results["seqid_t"]) + np.array(results["std_seqid_t"]),
                        color="#002CFF", alpha=0.2)
        ax.plot(results["t_half"], results["seqid_t_t_half"], label=r"SeqID$(t, t/2)$", color="#67BAA6")  # Orange
        ax.fill_between(results["t_half"],
                        np.array(results["seqid_t_t_half"]) - np.array(results["std_seqid_t_t_half"]),
                        np.array(results["seqid_t_t_half"]) + np.array(results["std_seqid_t_t_half"]),
                        color="#67BAA6", alpha=0.2)
        ax.set_xlabel(r"$t/2$ (Sweeps)")
        ax.set_ylabel("Sequence Identity")
        ax.legend(loc='upper right')
        ax.set_title(f"Round {t}")
        fig.suptitle("Mixing time")
        
        # Add annotation for mixing time
        ax.annotate(r"$t^{\mathrm{mix}}=$" + f"{results['t_half'][-1]}", xy=(0.96, 0.7), xycoords='axes fraction', fontsize=15,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    fig.tight_layout()

    mixing_times = [results['t_half'][-1] for results in res]

    return mixing_times, fig

@torch.no_grad
def sample_indep_sites(h: torch.tensor, n_samples: int, dtype=torch.float32, beta: float = 1.0,
                       device=torch.device('cpu'),
                       generator=torch.Generator()):
    L, q = h.size()
    logits = beta * h.unsqueeze(0).expand(n_samples, -1, -1)  # Shape: (nsamples, L, q)
    sampled_indices = torch.multinomial(torch.softmax(logits.reshape(-1, q), dim=-1), 
                                        num_samples=1, generator=generator).squeeze(-1)
    sampled_sequences = utils.one_hot(sampled_indices, num_classes=q).view(n_samples, L, q).to(dtype).to(device)

    return sampled_sequences

@torch.no_grad
def simulated_annealing(chains, compute_energy, n_steps, beta_schedule, callback=None):
    energies = compute_energy(chains)
    for beta in beta_schedule:
        chains, energies = _sample_metropolis(chains, compute_energy, n_steps, beta)
        if callback:
            callback(chains, energies, beta)

    return chains, energies