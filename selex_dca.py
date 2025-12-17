import torch
import numpy as np
import matplotlib.pyplot as plt
import adabmDCA

from typing import Dict, Callable, Tuple, List, Any
from tqdm.autonotebook import tqdm
from adabmDCA.statmech import _update_weights_AIS
from adabmDCA.stats import _get_slope
from adabmDCA.fasta import get_tokens, encode_sequence
from adabmDCA.functional import one_hot


def extract_Cij_from_freq(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    total_reads: torch.Tensor,
    mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts the lower triangular part of the covariance matrices of the data and chains starting from the frequencies.

    Args:
        fij (torch.Tensor): Two-point frequencies of the data. (n_rounds x L x q x q x q)
        pij (torch.Tensor): Two-point frequencies of the chains. (n_rounds x L x q x q x q)
        fi (torch.Tensor): Single-point frequencies of the data. (n_rounds x L x q)
        pi (torch.Tensor): Single-point frequencies of the chains. (n_rounds x L x q)
        mask (torch.Tensor | None, optional): Mask for comparing just a subset of the couplings. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted two-point frequencies of the data and chains.
    """
    L = fi.size(-2)
    
    t = torch.arange(len(total_reads))
    fi_ps = ((t * total_reads)[:,None,None] * fi).sum(dim=0)
    fi_ps = fi_ps / fi_ps.sum(dim=-1)[:,None]
    pi_ps = ((t * total_reads)[:,None,None] * pi).sum(dim=0)
    pi_ps = pi_ps / pi_ps.sum(dim=-1)[:,None]
    fij_ps = ((t * total_reads)[:,None,None,None,None] * fij).sum(dim=0)
    fij_ps = fij_ps / fij_ps.sum(dim=(-1,-3))[:,None,:,None]
    pij_ps = ((t * total_reads)[:,None,None,None,None] * pij).sum(dim=0)
    pij_ps = pij_ps / pij_ps.sum(dim=(-1,-3))[:,None,:,None]
    
    
    # Compute the covariance matrices
    cov_data = fij_ps - torch.einsum('ij,kl->ijkl', fi_ps, fi_ps)
    cov_chains = pij_ps - torch.einsum('ij,kl->ijkl', pi_ps, pi_ps)
    
    # Only use a subset of couplings if a mask is provided
    if mask is not None:
        cov_data = torch.where(mask, cov_data, torch.tensor(0.0, device=cov_data.device, dtype=cov_data.dtype))
        cov_chains = torch.where(mask, cov_chains, torch.tensor(0.0, device=cov_chains.device, dtype=cov_chains.dtype))
    
    # Extract only the entries of half the matrix and out of the diagonal blocks
    idx_row, idx_col = torch.tril_indices(L, L, offset=-1)
    fij_extract = cov_data[idx_row, :, idx_col, :].reshape(-1)
    pij_extract = cov_chains[idx_row, :, idx_col, :].reshape(-1)
    
    return fij_extract, pij_extract


def get_correlation_two_points(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    total_reads: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[float, float]:
    """Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains.

    Args:
        fij (torch.Tensor): Two-point frequencies of the data. (n_rounds x L x q x q x q)
        pij (torch.Tensor): Two-point frequencies of the chains. (n_rounds x L x q x q x q)
        fi (torch.Tensor): Single-point frequencies of the data. (n_rounds x L x q)
        pi (torch.Tensor): Single-point frequencies of the chains. (n_rounds x L x q)
        total_reads (torch.Tensor): Total number of reads at each round.
        mask (torch.Tensor | None, optional): Mask to select the couplings to use for the correlation coefficient. Defaults to None. 

    Returns:
        Tuple[float, float]: Pearson correlation coefficient of the two-sites statistics and slope of the interpolating line.
    """

    fij_extract, pij_extract = extract_Cij_from_freq(fij, pij, fi, pi, total_reads, mask)
    # torch.corrcoef does not support half precision
    pearson = torch.corrcoef(torch.stack([fij_extract.float(), pij_extract.float()]))[0, 1].item()
    slope = _get_slope(fij_extract.float(), pij_extract.float()).item()
    
    return pearson, slope

def init_parameters(fi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Initialize the parameters of the DCA model.

    Args:
        fi (torch.Tensor): Single-point frequencies of the data. (n_rounds x L x q)

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
    """
    n_rounds, L, q = fi.shape
    params = {}
    params["bias_Ns0"] = torch.log(fi[0])   # initialize with frequencies at first round
    params["couplings_Ns0"]  = torch.zeros((L, q, L, q), device=fi.device, dtype=fi.dtype)
    params["bias_ps"] = torch.zeros((L, q), device=fi.device, dtype=fi.dtype)
    params["couplings_ps"] = torch.zeros((L, q, L, q), device=fi.device, dtype=fi.dtype)
    
    return params

def get_params_at_round(
    params: Dict[str, torch.Tensor], 
    t: int,
    params_Ns0: Dict[str, torch.Tensor] | None = None):
    """Compute the parameters for the Potts model at round t.
    
    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model (Ns0 and ps), shared among rounds.
        t (int): the round index.
        params_Ns0 (Dict[str, torch.Tensor] | None = None): Optionally, use these parameters for round zero.
        
    Returns:
        Dict[str, torch.Tensor]: Biases and couplings to be MCMC sampled 
    """
    if params_Ns0 is None:
        bias_Ns0 = params["bias_Ns0"]
    else:
        bias_Ns0 = params_Ns0["bias_Ns0"]
    params_t = {}
    params_t["bias"] = bias_Ns0 + t * params["bias_ps"]
    params_t["coupling_matrix"] = t * params["couplings_ps"]

    if params_Ns0 is None:
        if "couplings_Ns0" in params:
            params_t["coupling_matrix"] += params["couplings_Ns0"]
    else:
        if "couplings_Ns0" in params_Ns0:
            params_t["coupling_matrix"] += params_Ns0["couplings_Ns0"]

    
    return params_t

def init_chains(
    num_rounds: int,
    num_chains: int,
    L: int,
    q: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    fi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Initialize the chains of the DCA model. If 'fi' is provided, the chains are sampled from the
    profile model, otherwise they are sampled uniformly at random.

    Args:
        num_rounds (int): Number of experiment rounds
        num_chains (int): Number of parallel chains.
        L (int): Length of the sequences.
        q (int): Number of values that each residue can assume.
        device (torch.device): Device where to store the chains.
        dtype (torch.dtype, optional): Data type of the chains. Defaults to torch.float32.
        fi (torch.Tensor | None, optional): Single-point frequencies. Defaults to None.

    Returns:
        torch.Tensor: Initialized parallel chains in one-hot encoding format.
    """
    if fi is None:
        chains = [torch.randint(low=0, high=q, size=(num_chains, L), device=device)
                  for _ in range(num_rounds)]
    else:
        chains = [torch.multinomial(fi_roundt, num_samples=num_chains, replacement=True).to(device=device).T
             for fi_roundt in fi]
    
    chains_tensor = torch.stack([one_hot(c, num_classes=q).to(dtype) for c in chains])
    
    return chains_tensor   # n_rounds, n_chains, L, q

@torch.jit.script
def compute_gradient(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
    total_reads: torch.Tensor,
    ts: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the log-likelihood of the model using PyTorch.

    Args:
        fi (torch.Tensor): Single-point frequencies of the data. (n_rounds x L x q)
        fij (torch.Tensor): Two-point frequencies of the data. (n_rounds x L x q x q x q)
        pi (torch.Tensor): Single-point frequencies of the chains. (n_rounds x L x q)
        pij (torch.Tensor): Two-point frequencies of the chains. (n_rounds x L x q x q x q)
        total_reads (torch.Tensor): Total number of reads at each round.
        ts (torch.Tensor): Just `range(len(total_reads))`. Needed because doing it inside will result in torchscript complaining

    Returns:
        Dict[str, torch.Tensor]: Gradient.
    """
    
    di = fi - pi
    dij = fij - pij    
    
    grad = {}
    W = total_reads.sum()
    grad["bias_Ns0"] = (total_reads[:,None,None] * di).sum(dim=0) / W
    grad["couplings_Ns0"] = (total_reads[:,None,None,None,None] * dij).sum(dim=0) / W
    grad["bias_ps"] = ((ts * total_reads)[:,None,None] * di).sum(dim=0) / W
    grad["couplings_ps"] = ((ts * total_reads)[:,None,None,None,None] * dij).sum(dim=0) / W
    
    return grad


def update_params(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
    total_reads: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask_ps: torch.Tensor,
    mask_Ns0: torch.Tensor,
    lr: float,
    l2reg: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Updates the parameters of the model.
    
    Args:
        fi (torch.Tensor): Single-point frequencies of the data. (n_rounds x L x q)
        fij (torch.Tensor): Two-point frequencies of the data. (n_rounds x L x q x q x q)
        pi (torch.Tensor): Single-point frequencies of the chains. (n_rounds x L x q)
        pij (torch.Tensor): Two-point frequencies of the chains. (n_rounds x L x q x q x q)
        total_reads (torch.Tensor): Total number of reads at each round.
        params (Dict[str, torch.Tensor]): Parameters of the model (Ns0 and ps), shared among rounds.
        mask_ps (torch.Tensor): Mask of the interaction graph for the couplings of ps.
        lr (float): Learning rate.
        l2reg (float): Constant for L2-regularization.
        
    Returns:
        Dict[str, torch.Tensor]: Updated parameters.
    """
    
    ts = torch.arange(len(total_reads))
    
    # Compute the gradient
    grad = compute_gradient(fi=fi, fij=fij, pi=pi, pij=pij, total_reads=total_reads, ts=ts)
    
    # Update parameters
    with torch.no_grad():
        for key in params:
            params[key] += lr * (grad[key] + l2reg * params[key])
        params["couplings_ps"] *= mask_ps # Remove autocorrelations
        params["couplings_Ns0"] *= mask_Ns0
    
    return params

def init_history():
    history = {
        "epochs": [],
        "pearson": [],
        "slope": [],
        "log-likelihood": [],
        "pearson_rounds": [],
    }
    return history

##### I'm dividing the log-likelihood (and its gradient) by the total number of reads at all rounds,
#####  to keep it "intensive" (= comparable between experiments with different numbers of reads)

def train(
    sampler: Callable,
    chains: torch.Tensor,
    fi: torch.Tensor,
    fij: torch.Tensor,
    total_reads: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    lr: float,    
    max_epochs: int,
    target_pearson: float,
    mask_ps: torch.Tensor | None = None,
    mask_Ns0: torch.Tensor | None = None,
    l2reg: float = 0.0,
    check_slope: bool = False,
    log_weights: torch.Tensor | None = None,
    history : Dict[str, List[float]] = init_history(),
    progress_bar: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    """Trains the model until the target Pearson correlation is reached or the maximum number of epochs is exceeded.

    Args:
        sampler (Callable): Sampling function.
        chains (torch.Tensor): Markov chains to sample from the model.
        fi (torch.Tensor): Single-point frequencies of the data. (n_rounds x L x q)
        fij (torch.Tensor): Two-point frequencies of the data. (n_rounds x L x q x q x q)
        params (Dict[str, torch.Tensor]): Parameters of the model (Ns0 and ps), shared among rounds.
        nsweeps (int): Number of Gibbs steps for each gradient estimation.
        lr (float): Learning rate.
        max_epochs (int): Maximum number of gradient updates to be done.
        target_pearson (float): Target Pearson coefficient.
        mask_ps (torch.Tensor, optional): Mask encoding the sparse graph for ps. Defaults to removing the (i,i) block
        mask_Ns0 (torch.Tensor, optional): Mask encoding the sparse graph for Ns0. Defaults to all zeros
        l2reg (float, optional): Constant for L2-regularization.
        check_slope (bool, optional): Whether to take into account the slope for the convergence criterion or not. Defaults to False.
        log_weights (torch.Tensor, optional): Log-weights used for the online computation of the log-likelihood. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar or not. Defaults to True.
        history (Dict[str, List[float]], optional): An already filled history, used to continue training from an intermediate state

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]: Updated chains and parameters, log-weights for the log-likelihood computation.
    """

    device = fi.device
    dtype = fi.dtype
    n_rounds, L, q = fi.shape
    n_chains = chains.size(1)

    if mask_ps is None:
        mask_ps = torch.ones(size=(L, q, L, q), dtype=torch.bool, device=device)
        mask_ps[torch.arange(L), :, torch.arange(L), :] = 0
    if mask_Ns0 is None:
        mask_Ns0 = torch.zeros(size=(L, q, L, q), dtype=torch.bool, device=device)

    log_likelihood = 0
    W = total_reads.sum()

    # log_weights used for the online computing of the log-likelihood
    if log_weights is None:
        log_weights = torch.zeros(n_rounds, n_chains, device=device, dtype=dtype)
    for t in torch.arange(n_rounds):
        logZt = (torch.logsumexp(log_weights[t], dim=0) - torch.log(torch.tensor(n_chains, device=device, dtype=dtype))).item()
        params_round_t = get_params_at_round(params, t)
        log_likelihood += total_reads[t] * adabmDCA.statmech.compute_log_likelihood(fi=fi[t], fij=fij[t], params=params_round_t, logZ=logZt)
    log_likelihood /= W
            
    # Compute the single-point and two-points frequencies of the simulated data
    pi = torch.stack([adabmDCA.stats.get_freq_single_point(data=c) for c in chains])
    pij = torch.stack([adabmDCA.stats.get_freq_two_points(data=c) for c in chains])

    
    def halt_condition(epochs, pearson, slope, check_slope):
        c1 = pearson < target_pearson
        c2 = epochs < max_epochs
        if check_slope:
            c3 = abs(slope - 1.) > 0.1
        else:
            c3 = False
        return not c2 * ((not c1) * c3 + c1)

    pearson_rounds, _ = zip(*[adabmDCA.stats.get_correlation_two_points(
                                fij=fij[t], pij=pij[t], fi=fi[t], pi=pi[t]
                                )
                        for t in range(len(total_reads))])
    pearson, slope = get_correlation_two_points(fij, pij, fi, pi, total_reads)

    epochs = 0
    if progress_bar: 
        pbar = tqdm(
            initial=0,
            total=max_epochs,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="{desc} {percentage:.2f}%[{bar}] Epoch: {n}/{total_fmt} [{elapsed}]"
        )
        pbar.set_description(f"Epochs: {epochs} - Pearson: {pearson:.2f}")
    
    while not halt_condition(epochs, pearson, slope, check_slope):
        
        # Store the previous parameters
        params_prev = {key: value.clone() for key, value in params.items()}
        
        # Update the parameters
        params = update_params(
            fi=fi,
            fij=fij,
            pi=pi,
            pij=pij,
            total_reads=total_reads,
            params=params,
            mask_ps=mask_ps,
            mask_Ns0=mask_Ns0,
            lr=lr,
            l2reg=l2reg
        )

        # Update Importance Sampling weights
        for t in range(n_rounds):
            params_prev_t = get_params_at_round(params_prev, t)
            params_curr_t = get_params_at_round(params, t)
            log_weights[t] = _update_weights_AIS(params_prev_t, params_curr_t, chains[t],  
                                                 log_weights[t])
        
        # Update the Markov chains
        for t in range(n_rounds):
            params_t = get_params_at_round(params, t)
            chains[t] = sampler(chains=chains[t], params=params_t, nsweeps=nsweeps)
        
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = torch.stack([adabmDCA.stats.get_freq_single_point(data=c) for c in chains])
        pij = torch.stack([adabmDCA.stats.get_freq_two_points(data=c) for c in chains])
        pearson_rounds, _ = zip(*[adabmDCA.stats.get_correlation_two_points(
                                    fij=fij[t], pij=pij[t], fi=fi[t], pi=pi[t]
                                    )
                            for t in range(len(total_reads))])
        pearson, slope = get_correlation_two_points(fij, pij, fi, pi, total_reads)


        # Estimate log-likelihood
        log_likelihood = 0
        for t in torch.arange(n_rounds):
            logZt = (torch.logsumexp(log_weights[t], dim=0) - torch.log(torch.tensor(n_chains, device=device, dtype=dtype))).item()
            params_round_t = get_params_at_round(params, t)
            log_likelihood += total_reads[t] * adabmDCA.statmech.compute_log_likelihood(fi=fi[t], fij=fij[t], params=params_round_t, logZ=logZt)
        log_likelihood /= W
        
        epochs += 1
        if progress_bar:
            pbar.n = epochs
            pbar.set_description(f"Epochs: {epochs} - Pearson: {pearson:.2f} - LL: {log_likelihood:.2f}")
        
        history["epochs"].append(epochs)
        history["pearson"].append(pearson)
        history["slope"].append(slope)
        history["log-likelihood"].append(log_likelihood)
        history["pearson_rounds"].append(pearson_rounds)
                
    if progress_bar:
        pbar.close()
        
    return chains, params, history

def compute_logNst(sequences, params, params_Ns0 = None):
    if params_Ns0 is None:
        params_Ns0 = params

    ts = range(len(sequences))
    sequences_unique, inverse_indices, counts = zip(*[
        torch.unique(seq_t, dim=0, return_inverse=True, return_counts=True)
        for seq_t in sequences])
    sequences_unique_oh = [one_hot(s) for s in sequences_unique]

    params_t = [get_params_at_round(params, t, params_Ns0) for t in ts]
    logNst = [-adabmDCA.statmech.compute_energy(sequences_unique_oh[t], params_t[t])
                   for t in ts]
    return logNst, sequences_unique, sequences_unique_oh, inverse_indices, counts

# Returns logNst vs logabundances counting each sequence once
def vectors_for_scatterplot_single_t_unique(logNst, 
                                            counts, 
                                            logNst_thresh=-torch.inf, 
                                            count_thresh=0):
    idx_unique_over_thresh = (logNst >= logNst_thresh) * (counts >= count_thresh)
    x = logNst[idx_unique_over_thresh]
    y = torch.log(counts[idx_unique_over_thresh])
    return x, y

# Returns logNst vs logabundances counting each sequence as many times as its empirical count
def vectors_for_scatterplot_single_t_nonunique(logNst, counts, logNst_thresh, inverse_indices):
    counts_nonunique = counts[inverse_indices]
    idx_unique_over_thresh = logNst[inverse_indices] >= logNst_thresh
    x = logNst[inverse_indices][idx_unique_over_thresh]
    y = torch.log(counts_nonunique[idx_unique_over_thresh])
    return x, y

def get_params_ps(params):
    return {"bias": params["bias_ps"], "coupling_matrix": params["couplings_ps"]}

def get_params_Ns0(params):
    return {"bias": params["bias_Ns0"], "coupling_matrix": params["couplings_Ns0"]}

def compute_logps(sequences, params):
    ts = range(len(sequences))
    sequences_unique, inverse_indices, counts = zip(*[
        torch.unique(sequences[t], dim=0, return_inverse=True, return_counts=True)
        for t in ts])
    sequences_unique_oh = [one_hot(s) for s in sequences_unique]
    logps = [-adabmDCA.statmech.compute_energy(sequences_unique_oh[t], params)
                   for t in ts]
    return logps

def guess_wildtype_from_sequence_counts(sequences_unique_round_zero, counts_round_zero):
    amax = counts_round_zero.argmax()
    wt = sequences_unique_round_zero[amax]
    return amax, wt

def guess_wildtype_from_site_counts(fi_round_zero):
    return fi_round_zero.argmax(dim=1)

def hamming(x, y):
    return (x != y).sum().item()


def get_contact_map(
    couplings : torch.Tensor,
    ) -> np.ndarray:
    """
    Computes the contact map from the model coupling matrix.

    Args:
        params (Dict[str, torch.Tensor]): Model parameters. Should contain:
            - "coupling_matrix": torch.Tensor of shape (L, q, L, q)
            - "bias": torch.Tensor of shape (L, q)

    Returns:
        np.ndarray: Contact map.
    """
    q = couplings.shape[1]
    device = couplings.device
       
    Jij = couplings

    # Compute the Frobenius norm
    cm = torch.sqrt(torch.square(Jij).sum([1, 3]))
    # Set to zero the diagonal
    cm = cm - torch.diag(cm.diag())
    # Compute the average-product corrected Frobenius norm
    Fapc = cm - torch.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    # set to zero the diagonal
    Fapc = Fapc - torch.diag(Fapc.diag())

    return Fapc.cpu().numpy()