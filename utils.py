from typing import Dict, Callable, Tuple, List, Any
import torch
import numpy as np
from Bio import SeqIO
import gzip
from pathlib import Path
from adabmDCA.fasta import get_tokens, encode_sequence
import sklearn
import glob

TOKENS_PROTEIN = "*ACDEFGHIKLMNPQRSTVWY"

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


def import_from_fasta(
    fasta_name: str | Path,
    tokens: str | None = None,
    filter_sequences: bool = False,
    remove_duplicates: bool = False,
    return_mask: bool = False):

    """Import sequences from a fasta or compressed fasta (.fas.gz) file. The following operations are performed:
    - If 'tokens' is provided, encodes the sequences in numeric format.
    - If 'filter_sequences' is True, removes the sequences whose tokens are not present in the alphabet.
    - If 'remove_duplicates' is True, removes the duplicated sequences.
    - If 'return_ids' is True, returns also the indices of the original sequences.

    Args:
        fasta_name (str | Path): Path to the fasta or compressed fasta (.fas.gz) file.
        tokens (str | None, optional): Alphabet to be used for the encoding. If provided, encodes the sequences in numeric format.
        filter_sequences (bool, optional): If True, removes the sequences whose tokens are not present in the alphabet. Defaults to False.
        remove_duplicates (bool, optional): If True, removes the duplicated sequences. Defaults to False.
        return_ids (bool, optional): If True, returns also the indices of the original sequences. Defaults to False.

    Raises:
        RuntimeError: The file is not in fasta format.

    Returns:
        Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (headers, sequences) or (headers, sequences, original_ids) if 'return_ids' is True.
    """
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
    
    # Filter sequences
    if filter_sequences:
        if tokens is None:
            raise ValueError("Argument 'tokens' must be provided if 'filter_sequences' is True.")
        tokens = get_tokens(tokens)
        tokens_list = [a for a in tokens]
        clean_names = []
        clean_sequences = []
        clean_mask = []
        for n, s in zip(names, sequences):
            if all(c in tokens_list for c in s):
                if n == "":
                    n = "unknown_sequence"
                clean_names.append(n)
                clean_sequences.append(s)
                clean_mask.append(True)
            else:
                print(f"Unknown token found: removing sequence {n}")
                clean_mask.append(False)
        names = np.array(clean_names)
        sequences = np.array(clean_sequences)
        mask = np.array(clean_mask)
        
    else:
        names = np.array(names)
        sequences = np.array(sequences)
        mask = np.full(len(sequences), True)
    
    # Remove duplicates
    if remove_duplicates:
        sequences, unique_ids = np.unique(sequences, return_index=True)
        # sort to preserve the original order
        order = np.argsort(unique_ids)
        sequences = sequences[order]
        names = names[unique_ids[order]]
        # set to false the mask elements corresponding to the duplicates
        original_indices_post_filtering = np.where(mask)[0]
        original_indices_of_unique_items = original_indices_post_filtering[unique_ids]
        mask_unique = np.full(len(mask), False)
        mask_unique[original_indices_of_unique_items] = True
        mask = mask & mask_unique
        
    if (tokens is not None) and (len(sequences) > 0):
        sequences = encode_sequence(sequences, tokens)
        
    out = (names, sequences)
    if return_mask:
        out = out + (mask,)
    
    return out

def sequences_from_file(experiment_id: str, round_id: str, 
                        device=torch.device("cpu")): 
    dirpath = (Path(__file__) / "../../Aptamer2025/data" / experiment_id).resolve()
    filepath = dirpath / (experiment_id + round_id + "_merged.fastq_result.fas.gz")
    tokens = "ACGT"
    headers, sequences = import_from_fasta(filepath, tokens=tokens, filter_sequences=False, remove_duplicates=False)
    seq = torch.tensor(sequences, device=device, dtype=torch.int32)
    
    return seq

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
                            dirpath = (Path(__file__) / "../../Aptamer2025/data/ab6/Txt files/").resolve(),
                            device: torch.device = torch.device('cpu')):
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

def sequences_from_file_ab6(round_id: str, 
                            dirpath = (Path(__file__) / "../../Aptamer2025/data/ab6/Txt files/").resolve(),
                            device: torch.device = torch.device('cpu')):
    sequences, counts = sequences_uniques_counts_from_file_ab6(round_id, dirpath, device)

    seq = torch.repeat_interleave(
        torch.tensor(sequences, device=device, dtype=torch.int32), 
        torch.tensor(counts, device=device, dtype=torch.int32), 
        dim=0)

    return seq
    

@torch.jit.script
def get_count_single_point(
    data: torch.Tensor,
    pseudo_count: float,
    ) -> torch.Tensor: 
    
    total_count, L, q = data.shape
    Ri = data.sum(0)
    Ri_tilde = (1. - pseudo_count) * Ri + pseudo_count * (total_count / q) 
    return Ri_tilde

def get_freq_single_point(
    data: torch.Tensor,
    pseudo_count: float
    ) -> torch.Tensor: 

    Ri = get_count_single_point(data, pseudo_count)
    return Ri / Ri.sum(dim=1, keepdim=True)

@torch.jit.script
def get_count_two_points(
    data: torch.Tensor,
    pseudo_count: float,
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
    pseudo_count: float,
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
    Jresh = J.reshape(q*L, q*L)
    idx_row_up, idx_col_up = torch.triu_indices(L*q, L*q, offset=L) 
    idx_row_low, idx_col_low = torch.tril_indices(L*q, L*q, offset=-L)
    Joff = torch.cat((Jresh[idx_row_up, idx_col_up], Jresh[idx_row_low, idx_col_low]))
    return Joff

def field_from_wildtype(wt_oh: torch.tensor, mutation_rate, dtype=torch.float32):
    q = wt_oh.size(-1)
    p_wt = 1 - mutation_rate
    p_non_wt = mutation_rate / (q - 1)

    return torch.log(torch.where(wt_oh.to(torch.bool), p_wt, p_non_wt)).to(dtype)

@torch.no_grad
def epistasis(compute_energy, wt_oh):

    def one_hot_mini(a, q, dtype, device):
        x = torch.zeros(q, dtype=dtype, device=device)
        x[a] = 1
        return x

    L, q = wt_oh.size()
    dtype=wt_oh.dtype
    device=wt_oh.device
    DDE = torch.zeros(L, q, L, q, dtype=dtype, device=device)
    E_wt = compute_energy(wt_oh)
    x = wt_oh.clone()
    for i in range(L):
        for bi in range(q):
            for j in range(L):
                for bj in range(q):
                    x[i,:] = one_hot_mini(bi, q, dtype=dtype, device=device)
                    E_i = compute_energy(x) - E_wt
                    x[j,:] = one_hot_mini(bj, q, dtype=dtype, device=device)
                    E_double = compute_energy(x) - E_wt
                    x[i,:] = wt_oh[i,:] 
                    E_j = compute_energy(x) - E_wt
                    x[j,:] = wt_oh[j,:]
                    DDE[i,bi,j,bj] = E_double - E_i - E_j
    
    return DDE

def unique_sequences_counts_enrichments(sequences, verbose=True):
    n_rounds = len(sequences)
    if verbose:
        print('Extracting unique sequences and counts at each round...')
    sequences_unique, inverse_indices, counts = zip(*[
        torch.unique(seq_t, dim=0, return_inverse=True, return_counts=True)
        for seq_t in sequences])
    shifts = torch.cat((torch.tensor([0]), torch.cumsum(torch.tensor([len(s) for s in sequences_unique]), 0)), 0)[:-1]
    seq_unique_all = torch.cat(sequences_unique, dim=0)  
    if verbose:
        print('Merging sequences from all rounds in a single container...')
    sequences_unique_all, inverse_indices_all, counts_all = torch.unique(seq_unique_all, dim=0, return_inverse=True, return_counts=True)
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
    if verbose:
        print('Calculating enrichments...')
    enrichments = [counts_unique[t+1] / counts_unique[t] for t in range(n_rounds-1)]
    if verbose:
        print('Finished')

    return sequences_unique_all, counts_unique, enrichments