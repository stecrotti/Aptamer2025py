import torch
import numpy as np
from Bio import SeqIO
import gzip
from pathlib import Path
from adabmDCA.fasta import get_tokens, encode_sequence
import sklearn
import glob
import matplotlib.pyplot as plt
import random
import datetime

TOKENS_PROTEIN = "*ACDEFGHIKLMNPQRSTVWY"
TOKENS_DNA = "ACGT"
TOKENS_RNA = 'ACGU'

# copied from adabmDCA
@torch.no_grad
def _one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
   
    if x.dim() not in (1, 2):
        raise ValueError("Input tensor x must be 1D or 2D")
    
    if num_classes < 0:
        num_classes = int(x.max() + 1)
    
    # Handle 1D case (single sequence)
    if x.dim() == 1:
        res = torch.zeros(x.shape[0], num_classes, device=x.device, dtype=dtype)
        index = (torch.arange(x.shape[0], device=x.device), x)
        values = torch.ones(x.shape[0], device=x.device, dtype=dtype)
        res.index_put_(index, values)
        return res
    
    # Handle 2D case (batch of sequences)
    res = torch.zeros(x.shape[0], x.shape[1], num_classes, device=x.device, dtype=dtype)
    tmp = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[1], device=x.device),
        indexing="ij",
    )
    index = (tmp[0], tmp[1], x)
    values = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=dtype)
    res.index_put_(index, values)
    
    return res


def one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
    Works for both 1D (single sequence) and 2D (batch of sequences) tensors.
    
    Args:
        x (torch.Tensor): Input tensor to be one-hot encoded. Shape (L,) or (batch_size, L).
        num_classes (int, optional): Number of classes. If -1, the number of classes is inferred from the input tensor. Defaults to -1.
        dtype (torch.dtype, optional): Data type of the output tensor. Defaults to torch.float32.
        
    Returns:
        torch.Tensor: One-hot encoded tensor. Shape (L, num_classes) for 1D input or (batch_size, L, num_classes) for 2D input.
    """
    return _one_hot(x, num_classes, dtype)

def log_factorial(x: int | torch.IntTensor):
    isscalar = isinstance(x, int)
    if isscalar:
        y = torch.IntTensor([x])
    else:
        y = x
    lf = torch.lgamma(y+1)
    if isscalar:
        return lf.item()
    else:
        return lf

def log_multinomial(n):
    N = n.sum().item()
    logNfact = log_factorial(N)
    lognfact = log_factorial(n).sum().item()
    logmult = logNfact - lognfact
    return logmult

def import_from_fasta(
    fasta_name: str | Path,
    tokens: str | None = None):

    # Open the file, handling both .fasta and .fas.gz formats
    if str(fasta_name).endswith(".gz"):
        with gzip.open(fasta_name, "rt") as fasta_file: 
            records = list(SeqIO.parse(fasta_file, "fasta-pearson"))
    else:
        with open(fasta_name, "r") as fasta_file:
            records = list(SeqIO.parse(fasta_file, "fasta-pearson"))

    # Import headers and sequences
    sequences = []
    names = []
    for record in records:
        names.append(str(record.id))
        sequences.append(str(record.seq))
    
    names = np.array(names)
    sequences = np.array(sequences)
        
    if (tokens is not None) and (len(sequences) > 0):
        sequences_encoded = encode_sequence(sequences, tokens)
    
    return sequences_encoded

def sequences_from_file(experiment_id: str, round_id: str, 
                        device=torch.device("cpu")): 
    dirpath = (Path(__file__) / "../../Aptamer2025/data" / experiment_id).resolve()
    filepath = dirpath / (experiment_id + round_id + "_merged.fastq_result.fas.gz")
    tokens = TOKENS_DNA
    sequences = import_from_fasta(filepath, tokens=tokens)
    seq = torch.tensor(sequences, device=device, dtype=torch.int32)
    
    return seq

def sequences_unique_and_counts(sequences): 
    n_sequences = len(sequences)
    sequences_unique, counts = torch.unique(sequences, dim=0, return_counts=True)
    assert counts.sum() == n_sequences
    log_multinomial_factors = log_multinomial(counts)
    
    return sequences_unique, counts, log_multinomial_factors

def sequences_counts_from_file(*args, **kwargs):
    sequences = sequences_from_file(*args, **kwargs)
    sequences_unique, counts, logmult = sequences_unique_and_counts(sequences)

    return sequences, sequences_unique, counts, logmult

def sequences_from_files(experiment_id: str, round_ids, verbose=True):
    sequences = []
    if verbose: print('Extracting sequences...')
    for round_id in round_ids:
        s = sequences_from_file(experiment_id, round_id)
        sequences.append(s)
        if verbose: print(f"Finished round {round_id}")

    return sequences

def sequences_counts_from_files(experiment_id: str, round_ids, verbose=True):
    sequences = []
    sequences_unique = []
    counts = []
    logmult = []
    for round_id in round_ids:
        s, su, c, l = sequences_counts_from_file(experiment_id, round_id)
        sequences.append(s)
        sequences_unique.append(su)
        counts.append(c)
        logmult.append(l)
        if verbose: print(f"Finished round {round_id}")
    
    return sequences, sequences_unique, counts, torch.tensor(logmult)

def sequences_from_files_detailed(experiment_id: str, round_ids, verbose=True, return_enrichments=False):
    n_rounds = len(round_ids)

    sequences = []
    sequences_unique = []
    counts = []
    log_multinomial_factors = []
    
    if verbose: print(f'Extracting sequences from {n_rounds} files...')
    for t in range(n_rounds):
        s, su, c, l = sequences_counts_from_file(experiment_id, round_ids[t])
        sequences.append(s)
        sequences_unique.append(su)
        counts.append(c)
        log_multinomial_factors.append(l)
        if verbose: print(f"Finished round {round_ids[t]}")
    log_multinomial_factors = torch.tensor(log_multinomial_factors)
        
    shifts = torch.cat((torch.tensor([0]), torch.cumsum(torch.tensor([len(s) for s in sequences_unique]), 0)), 0)[:-1]
    seq_unique_all = torch.cat(sequences_unique, dim=0)  
    if verbose:
        print('Merging sequences from all rounds in a single container...')
    sequences_unique_all, inverse_indices_all = torch.unique(seq_unique_all, dim=0, return_inverse=True)
    n_seq_tot = len(sequences_unique_all)
    if verbose:
        print('Assigning counts at each round to unique sequences...')
    counts_unique = []
    for t in range(n_rounds):
        print(f'\tStarting round {round_ids[t]}...')
        counts_t = torch.zeros(n_seq_tot, dtype=torch.int)
        for i in range(len(counts[t])):
            c = counts[t][i]
            seq_id = inverse_indices_all[shifts[t] + i]
            counts_t[seq_id] += c
        counts_unique.append(counts_t)
    if return_enrichments:
        if verbose:
            print('Calculating enrichments...')
        enrichments = [counts_unique[t+1] / counts_unique[t] for t in range(n_rounds-1)]
    if verbose:
        print('Finished')
    if return_enrichments:
        return sequences, sequences_unique, counts, log_multinomial_factors, sequences_unique_all, counts_unique, enrichments
    else:
        return sequences, sequences_unique, counts, log_multinomial_factors, sequences_unique_all, counts_unique

def sequences_from_file_thrombin(experiment_id: str, round_id: str, device):         
    dirpath = (Path(__file__) / "../../Aptamer2025/data" / experiment_id).resolve()   
    filepath = dirpath / (experiment_id + "_" + round_id + ".fasta")
    tokens = "ACGT"
    headers, sequences = import_from_fasta(filepath, tokens=tokens, filter_sequences=False, remove_duplicates=False)

    def parse_header(headers, s):
        header = headers[s]
        parsed = header[3:].split("-")
        assert(float(parsed[0]) == s)
        count = int(parsed[1])
        return count

    counts = [parse_header(headers, s) for s in range(len(headers))]
    seq = torch.repeat_interleave(
        torch.tensor(sequences, device=device, dtype=torch.int32), 
        torch.tensor(counts, device=device, dtype=torch.int32), 
        dim=0)

    return seq

def sequences_uniques_counts_from_file_ab6(round_id: str, 
                            dirpath = (Path(__file__) / "../../Aptamer2025/data/ab6/Txt files/").resolve()):
    files = glob.glob(f"*_{round_id}_aa.txt", root_dir=dirpath)
    nf = len(files)
    if nf != 1:
        raise ValueError(f"Expected round id {round_id} to match 1 file, got {nf} matches: {files}")
    filepath = files[0]
    sequences = []
    counts = []
    with open(dirpath / filepath) as file:
        for line in file:
            spl = line.split('\t')
            sequence = spl[0]
            count = spl[1]
            if len(sequence) == 6:
                sequences.append(sequence)
                counts.append(int(count))

    sequences = encode_sequence(sequences, TOKENS_PROTEIN)

    return sequences, counts

def sequences_from_file_ab6(round_id: str, dtype=torch.int32,
                            dirpath = (Path(__file__) / "../../Aptamer2025/data/ab6/Txt files/").resolve(),
                            return_log_multinomial_factors=False):
    sequences_unique, counts = sequences_uniques_counts_from_file_ab6(round_id, dirpath)
    sequences_unique = torch.tensor(sequences_unique, dtype=dtype)
    counts = torch.tensor(counts, dtype=dtype)
    sequences = torch.repeat_interleave(sequences_unique, counts, dim=0)

    if return_log_multinomial_factors:
        log_multinomial_factors = log_multinomial(counts)
        return sequences, log_multinomial_factors
    else:
        return sequences
    
def sequences_from_files_ab6(round_ids, dtype=torch.int32,
                            dirpath = (Path(__file__) / "../../Aptamer2025/data/ab6/Txt files/").resolve(),
                            return_log_multinomial_factors=False):
    
    s, l = zip(*[sequences_from_file_ab6(round_id, dtype=dtype, dirpath=dirpath,
                                         return_log_multinomial_factors=return_log_multinomial_factors)
                for round_id in round_ids])
    return s, torch.tensor(l)
    

@torch.jit.script
def get_count_single_point(
    data: torch.Tensor,
    pseudo_count: float = 0.0
    ) -> torch.Tensor: 
    
    total_count, L, q = data.shape
    Ri = data.sum(0)
    Ri_tilde = (1. - pseudo_count) * Ri + pseudo_count * (total_count / q) 
    return Ri_tilde

def get_freq_single_point(
    data: torch.Tensor,
    pseudo_count: float = 0.0
    ) -> torch.Tensor: 

    Ri = get_count_single_point(data, pseudo_count)
    return Ri / Ri.sum(dim=1, keepdim=True)

@torch.jit.script
def get_count_two_points(
    data: torch.Tensor,
    pseudo_count: float = 0.0
    ) -> torch.Tensor:
    
    device = data.device
    total_count, L, q = data.shape
    data_oh = data.reshape(total_count, q * L)
    
    Rij = data_oh.T @ data_oh
    Rij_tilde = (1. - pseudo_count) * Rij + pseudo_count * (total_count / q**2)
    # Diagonal terms must represent the single point frequencies
    Rij_diag = get_count_single_point(data, pseudo_count).ravel()
    # Set the diagonal terms of fij to the single point frequencies
    Rij_tilde = torch.diagonal_scatter(Rij_tilde, Rij_diag, dim1=0, dim2=1)

    mask = torch.ones(L, q, L, q, device=device)
    mask[torch.arange(L), :, torch.arange(L), :] = 0
    mask_2d = mask.reshape(L*q, L*q)
    mask_2d += torch.diag(torch.ones(L*q, device=device))
    Rij_tilde = Rij_tilde * mask_2d
    Rij_tilde = Rij_tilde.reshape(L, q, L, q)

    return Rij_tilde

def get_freq_two_points(
    data: torch.Tensor,
    pseudo_count: float = 0.0
    ) -> torch.Tensor: 

    Rij = get_count_two_points(data, pseudo_count)
    return Rij / Rij.sum(dim=(1,3), keepdim=True)
    
    
def frequences_from_sequences_oh(seq_oh, pseudo_count=0.0):
    Ri, Rij, Rt = counts_from_sequences_oh(seq_oh, pseudo_count=pseudo_count)
    fi = Ri / Rt
    fij = Rij / Rt
    
    return fi, fij, Rt
    
def frequences_from_sequences(seq, pseudo_count=0.001, dtype = torch.float32,
                                num_classes=-1):
    seq_oh = one_hot(seq, num_classes=num_classes).to(dtype)
    return frequences_from_sequences_oh(seq_oh, pseudo_count=pseudo_count)

def frequencies_from_file(experiment_id: str, round_id: str, device, dtype = torch.float32, pseudo_count = 0.001):
    seq = sequences_from_file(experiment_id, round_id, device, dtype)
    
    return frequences_from_sequences(seq, pseudo_count=pseudo_count)

def counts_from_sequences_oh(seq_oh, pseudo_count=0.0):
    Ri = get_count_single_point(data=seq_oh, pseudo_count=pseudo_count)
    Rij = get_count_two_points(data=seq_oh, pseudo_count=pseudo_count)
    Rt = seq_oh.size(0)
    
    return Ri, Rij, Rt
    
def counts_from_sequences(seq, pseudo_count=0.0, dtype = torch.float32, 
                            num_classes=-1):
    seq_oh = one_hot(seq, num_classes=num_classes).to(dtype)
    return counts_from_sequences_oh(seq_oh, pseudo_count=pseudo_count)

def counts_from_file(experiment_id: str, round_id: str, device, 
                     dtype = torch.float32, pseudo_count = 0.0):
    seq = sequences_from_file(experiment_id, round_id, device, dtype)
    
    return counts_from_sequences(seq, pseudo_count=pseudo_count)


def normalize_to_prob(x):
    assert(len(x.shape) == 2)
    norm = x.sum(dim=-1, keepdim=True)
    return x / norm

def normalize_to_logprob(x):
    assert(len(x.shape) == 2)
    norm = x.logsumexp(dim=-1, keepdim=True)
    return x - norm

def zerosum_gauge_bias(bias):
    return bias - bias.mean(dim=1, keepdim=True)

def zerosum_gauge_couplings(coupling_matrix):
    coupling_matrix -= coupling_matrix.mean(dim=1, keepdim=True) + \
                            coupling_matrix.mean(dim=3, keepdim=True) - \
                            coupling_matrix.mean(dim=(1, 3), keepdim=True)
    return coupling_matrix

def set_zerosum_gauge(J, h=None, mask=None):
    L, q = J.shape[:2]
    if mask is None:
        mask = torch.ones(L, q, L, q, device=J.device)
        mask[torch.arange(L), :, torch.arange(L), :] = 0
    Jmasked = J * mask

    if h is not None:
        dh = 0.5 * (Jmasked.mean(3).sum(2) + Jmasked.mean(1).sum(0) + Jmasked.mean(3).mean(1, keepdim=True).sum(2))
    
    dJ = Jmasked.mean(3, keepdim=True) + Jmasked.mean(1, keepdim=True) + Jmasked.mean((1,3), keepdim=True)
    J -= dJ

    if h is not None:
        h += dh
        h -= h.mean(dim=1, keepdim=True)
        return J, h

    return J

def random_data(n_sequences, L, q):
    x_ = torch.randint(q, (n_sequences, L))
    return one_hot(x_)

def compute_pca(*sequences_oh, n_components=2):
    pca = None
    pcs = []
    for seq in sequences_oh:
        xflat = seq.reshape(seq.shape[0], -1)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(xflat)
        xflat_normalized = scaler.transform(xflat)
        if pca is None:
            pca = sklearn.decomposition.PCA(n_components=n_components)
        pca.fit(xflat_normalized)
        pcs.append(pca.transform(xflat_normalized))
    if len(pcs)==1:
        return pcs[0]
    else:
        return tuple(pcs)
    
def rand_coupling_matrix(L, q, device=None, dtype=None, rescaling = None):
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    if rescaling is None:
        rescaling = L ** (-0.5)

    J_ = torch.randn(L*q, L*q, dtype=dtype, device=device)
    J_ = (J_ + J_.t()) / 2
    J = J_.reshape(L, q, L, q) * rescaling

    return J

def rand_sequences_oh(M, L, q, device=None, dtype=None):
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32

    x = torch.randint(q, (M, L))
    return one_hot(x).to(device=device, dtype=dtype)


def compute_pearson(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()

def compute_slope(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = len(x)
    num = n * (x @ y) - y.sum() * x.sum()
    den = n * (x @ x) - torch.square(x.sum())
    return torch.abs(num / den).item()

def off_diagonal_terms(J: torch.tensor):
    L, q, L1, q1 = J.size()
    assert q == q1
    assert L == L1
    idx_row, idx_col = torch.tril_indices(L, L, offset=-1)
    Jlower = J[idx_row, :, idx_col, :].reshape(-1)
    return Jlower

def field_from_wildtype(wt_oh: torch.tensor, mutation_rate, dtype=torch.float32):
    q = wt_oh.size(-1)
    p_wt = 1 - mutation_rate
    p_non_wt = mutation_rate / (q - 1)

    return torch.log(torch.where(wt_oh.to(torch.bool), p_wt, p_non_wt)).to(dtype)

@torch.no_grad
def epistasis(compute_energy, wt_oh):
    """
    Compute epistatic couplings as in equation (6) of the paper.
    
    ΔΔE(i→bi, j→bj) = ΔE(double) - ΔE(i→bi) - ΔE(j→bj)
    where ΔE is the energy change relative to wild-type
    """
    
    L, q = wt_oh.size()
    dtype = wt_oh.dtype
    device = wt_oh.device
    
    # Precompute wild-type energy
    E_wt = compute_energy(wt_oh)
    
    # Precompute all single mutation energies
    DE_single = torch.zeros(L, q, dtype=dtype, device=device)
    x = wt_oh.clone()
    
    for i in range(L):
        wt_i = wt_oh[i].clone()  # Save original
        for bi in range(q):
            x[i] = torch.zeros(q, dtype=dtype, device=device)
            x[i, bi] = 1
            DE_single[i, bi] = compute_energy(x) - E_wt
        x[i] = wt_i  # Restore
    
    # Compute double mutations and epistasis
    DDE = torch.zeros(L, q, L, q, dtype=dtype, device=device)
    
    for i in range(L):
        wt_i = wt_oh[i].clone()
        for bi in range(q):
            x[i] = torch.zeros(q, dtype=dtype, device=device)
            x[i, bi] = 1
            
            for j in range(L):
                if i == j:
                    continue  # Skip diagonal
                    
                wt_j = wt_oh[j].clone()
                for bj in range(q):
                    x[j] = torch.zeros(q, dtype=dtype, device=device)
                    x[j, bj] = 1
                    
                    # Double mutation energy
                    E_double = compute_energy(x) - E_wt
                    
                    # Epistasis
                    DDE[i, bi, j, bj] = E_double - DE_single[i, bi] - DE_single[j, bj]
                    
                x[j] = wt_j  # Restore j
            x[i] = wt_i  # Restore i
    
    return DDE

def group_rounds(sequences, 
                sequences_unique = None,
                counts = None,
                verbose = True,
                return_enrichments = False):
    n_rounds = len(sequences)
    if (sequences_unique is None) or (counts is None): 
        if verbose:
            print('Extracting unique sequences and counts at each round...')
        sequences_unique, counts = zip(*[
            torch.unique(seq_t, dim=0, return_counts=True)
            for seq_t in sequences])
    shifts = torch.cat((torch.tensor([0]), torch.cumsum(torch.tensor([len(s) for s in sequences_unique]), 0)), 0)[:-1]
    seq_unique_all = torch.cat(sequences_unique, dim=0)  
    if verbose:
        print('Merging sequences from all rounds in a single container...')
    sequences_unique_all, inverse_indices_all = torch.unique(seq_unique_all, dim=0, return_inverse=True)
    n_seq_tot = len(sequences_unique_all)
    if verbose:
        print('Assigning counts at each round to unique sequences...')
    counts_unique = []
    for t in range(n_rounds):
        print(f'\tStarting round {t}...')
        counts_t = torch.zeros(n_seq_tot, dtype=torch.int)
        for i in range(len(counts[t])):
            c = counts[t][i]
            seq_id = inverse_indices_all[shifts[t] + i]
            counts_t[seq_id] += c
        counts_unique.append(counts_t)
    if return_enrichments:
        if verbose:
            print('Calculating enrichments...')
        enrichments = [counts_unique[t+1] / counts_unique[t] for t in range(n_rounds-1)]
    if verbose:
        print('Finished')
    if return_enrichments:
        return sequences_unique_all, counts_unique, enrichments
    else:
        return sequences_unique_all, counts_unique

def _discard_cumsum_below(populations, thresh):
    populations = populations / populations.sum(0, keepdim=True)
    perm = torch.argsort(populations)
    id_edge = (populations[perm].cumsum(0) > thresh).to(float).argmax().item()
    idx_below_thresh = perm[:id_edge+1]
    idx_above_thresh = perm[id_edge+1:]
    assert len(idx_below_thresh) + len(idx_above_thresh) == len(populations)
    return idx_below_thresh, idx_above_thresh

def binned_logenrichments(model, sequences_unique_all_oh, enrichments, counts_unique, n_bins=25,
              selection_round = 1, plot=False, thresh=0.0, n_subsample=10**4):
    n_rounds = len(counts_unique)
    assert model.get_n_rounds() == n_rounds
    logps_all = - model.selection_energy_at_round(sequences_unique_all_oh, selection_round).detach()
    logps_binned, bins_ps = torch.histogram(logps_all, bins=n_bins)
    buckets_logps = torch.bucketize(logps_all, bins_ps)
    counts_binned_logps = [torch.tensor([counts_unique[t][buckets_logps == b].sum().item() for b in range(n_bins)]) for t in range(n_rounds)]
    enrichments_binned = [counts_binned_logps[t+1] / counts_binned_logps[t] for t in range(n_rounds - 1)]
    logenrich_binned_ps = [torch.log(enrichments_binned[t]) for t in range(n_rounds - 1)]
    idx_below_thresh, idx_above_thresh = _discard_cumsum_below(logps_binned, thresh)

    if plot:
        n_sel = model.get_n_selection_rounds()
        n_seq = len(logps_all)
        idx = random.sample(range(n_seq), min(n_subsample, n_seq))
        fig1, ax = plt.subplots(figsize=(4,3))
        for n in range(n_sel):
            ax.scatter(logps_all[idx], torch.log(enrichments[n][idx]), label=f'round {n} to {n+1}', s=2)
            ax.set_xlabel('log ps')
            ax.set_ylabel('log enrichment')
            ax.legend()
        ax.set_title('logps vs log enrichment')

        # hist, ax = plt.subplots(figsize=(4,3))
        # ax.bar(bins_ps[1:][idx_above_thresh], logps_binned[idx_above_thresh])
        # ax.bar(bins_ps[1:][idx_below_thresh], logps_binned[idx_below_thresh], 
        #            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0], alpha=0.1)
        # ax.set_xlabel('log ps')
        # ax.set_ylabel('count')
        # ax.set_title('Histogram of logps')

        fig2, ax = plt.subplots(figsize=(4,3))
        for n in range(n_sel):
            # ax.scatter(bins_ps[1:][idx_above_thresh], logenrich_binned_ps[n][idx_above_thresh], label=f'round {n} to {n+1}')
            # ax.scatter(bins_ps[1:][idx_below_thresh], logenrich_binned_ps[n][idx_below_thresh],
            #            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0], alpha=0.1)
            ax.plot(bins_ps[1:], logenrich_binned_ps[n], label=f'round {n} to {n+1}', marker='o')
            ax.set_xlabel('log ps')
            ax.set_ylabel('log enrichment, binned')
            ax.legend()
        ax.set_title('logps vs log enrichment - binned')

        return logps_all, bins_ps[1:], logps_binned, logenrich_binned_ps, idx_below_thresh, idx_above_thresh, fig1, fig2

    return logps_all, bins_ps[1:], logps_binned, logenrich_binned_ps, idx_below_thresh, idx_above_thresh

def binned_logcounts(model, sequences_unique_all_oh, counts_unique, n_bins = 25, plot=False, thresh=0.0,
                     n_subsample=10**4):
    n_rounds = len(counts_unique)
    assert model.get_n_rounds() == n_rounds
    logNst_unique = [- model.compute_energy_up_to_round(sequences_unique_all_oh, t).detach()
                for t in range(n_rounds)]
    logNst_binned, bins_Nst = zip(*[torch.histogram(l, bins=n_bins) for l in logNst_unique])
    buckets_logNst = [torch.bucketize(logNst_unique[t], bins_Nst[t]) for t in range(n_rounds)]
    logcounts_binned_logNst = [torch.tensor([counts_unique[t][buckets_logNst[t] == b].to(torch.float).mean().item() for b in range(n_bins)]).log() 
                            for t in range(n_rounds)]
    idx_below_thresh, idx_above_thresh = zip(*[_discard_cumsum_below(logNst_binned[t], thresh)
                                               for t in range(n_rounds)])
    if plot:
        n_seq = len(sequences_unique_all_oh)
        idx = random.sample(range(n_seq), min(n_subsample, n_seq))
        fig1, ax = plt.subplots(figsize=(4,3))
        for t in range(n_rounds):
            ax.scatter(logNst_unique[t][idx], torch.log(counts_unique[t][idx]), 
                       label=f'Round {t}', s=2)
            ax.set_xlabel('log Nst')
            ax.set_ylabel('log count')
            ax.legend()
        ax.set_title('logNst vs log count')

        # hist, ax = plt.subplots(figsize=(4,3))
        # for t in range(n_rounds):
        #     ax.bar(bins_Nst[t][1:][idx_above_thresh[t]], logNst_binned[t][idx_above_thresh[t]], 
        #                label=f'Round {t}')
        #     ax.bar(bins_Nst[t][1:][idx_below_thresh[t]], logNst_binned[t][idx_below_thresh[t]], 
        #                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][t], alpha=0.1)
        #     ax.set_xlabel('log Nst')
        #     ax.set_ylabel('count')
        #     ax.legend()
        # ax.set_title('Histogram of logNst')

        fig2, ax = plt.subplots(figsize=(4,3))
        for t in range(n_rounds):
            # ax.scatter(bins_Nst[t][1:][idx_above_thresh[t]], logcounts_binned_logNst[t][idx_above_thresh[t]], 
            #            label=f'Round {t}')
            # ax.scatter(bins_Nst[t][1:][idx_below_thresh[t]], logcounts_binned_logNst[t][idx_below_thresh[t]], 
            #            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][t], alpha=0.1)
            ax.plot(bins_Nst[t][1:], logcounts_binned_logNst[t], marker='o',
                       label=f'Round {t}')
            ax.set_xlabel('log Nst, binned')
            ax.set_ylabel('log count, binned')
            ax.legend()
        ax.set_title('logNst vs log count - binned')

        return logNst_unique, [bins_Nst[t][1:] for t in range(n_rounds)], logNst_binned, logcounts_binned_logNst, idx_below_thresh, idx_above_thresh, fig1, fig2

    return logNst_unique, [bins_Nst[t][1:] for t in range(n_rounds)], logNst_binned, logcounts_binned_logNst, idx_below_thresh, idx_above_thresh

def datetime_as_string():
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=1)))
    return now.strftime("%m-%d-%Y_%H-%M-%S")

def best_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if verbose:
        print(f'Selected device: {device}')
    return device

def subsample_sequences(sequences_unique_all, counts_unique, idx, 
                        device=torch.device('cpu')):
    n_rounds = len(counts_unique)
    sequences = []
    log_multinomial_factors = []
    total_reads = []
    q = sequences_unique_all.max().item() + 1
    
    for t in range(n_rounds):
        sequences.append(torch.repeat_interleave(sequences_unique_all[idx], counts_unique[t][idx], dim=0))
        log_multinomial_factors.append(log_multinomial(counts_unique[t][idx]))
        total_reads.append(counts_unique[t][idx].sum().item())
    
    log_multinomial_factors = torch.tensor(log_multinomial_factors).to(device)
    total_reads = torch.tensor(total_reads).to(device)
    sequences_oh = [one_hot(s, num_classes=q) for s in sequences]

    return sequences_oh, total_reads, log_multinomial_factors

def hamming(x: torch.tensor, y: torch.tensor):
    if torch.is_floating_point(x) and torch.is_floating_point(y):
        L = x.size(-2)
        return L - (x * y).sum((-2,-1))
    else:
        return (x != y).sum(-1)