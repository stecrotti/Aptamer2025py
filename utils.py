from typing import Dict, Callable, Tuple, List, Any
import torch
import numpy as np
from Bio import SeqIO
import gzip
from pathlib import Path
import adabmDCA
from adabmDCA.functional import one_hot
from adabmDCA.fasta import get_tokens, encode_sequence


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

def sequences_from_file(experiment_id: str, round_id: str, device): 
    dirpath = "../../Aptamer2025/data/" + experiment_id
    filepath = dirpath + "/" + experiment_id + round_id + "_merged.fastq_result.fas.gz"
    tokens = "ACGT"
    headers, sequences = import_from_fasta(filepath, tokens=tokens, filter_sequences=False, remove_duplicates=False)
    seq = torch.tensor(sequences, device=device, dtype=torch.int32)
    
    return seq

def sequences_from_file_thrombin(experiment_id: str, round_id: str, device):         
    dirpath = "../../Aptamer2025/data/" + experiment_id
    filepath = dirpath + "/" + experiment_id + "_" + round_id + ".fasta"
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

@torch.jit.script
def get_count_single_point(
    data: torch.Tensor,
    pseudo_count: float,
    ) -> torch.Tensor: 
    
    total_count, L, q = data.shape
    Ri = data.sum(0)
    Ri_tilde = (1. - pseudo_count) * Ri + pseudo_count * (total_count / q) 
    return Ri_tilde

@torch.jit.script
def get_count_two_points(
    data: torch.Tensor,
    pseudo_count: float,
    ) -> torch.Tensor:
    
    total_count, L, q = data.shape
    data_oh = data.reshape(total_count, q * L)
    
    Rij = data_oh.T @ data_oh
    Rij_tilde = (1. - pseudo_count) * Rij + pseudo_count * (total_count / q**2)
    # Diagonal terms must represent the single point frequencies
    Rij_diag = get_count_single_point(data, pseudo_count).ravel()
    # Set the diagonal terms of fij to the single point frequencies
    Rij_tilde = torch.diagonal_scatter(Rij_tilde, Rij_diag, dim1=0, dim2=1)

    mask = torch.ones(L, q, L, q)
    mask[torch.arange(L), :, torch.arange(L), :] = 0
    mask_2d = mask.reshape(L*q, L*q)
    mask_2d += torch.diag(torch.ones(L*q))
    Rij_tilde = Rij_tilde * mask_2d
    Rij_tilde = Rij_tilde.reshape(L, q, L, q)

    return Rij_tilde
    
    
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

def set_zerosum_gauge(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Sets the zero-sum gauge on the coupling matrix and biases.
    
    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model.
        
    Returns:
        Dict[str, torch.Tensor]: New dictionary with modified parameters.
            "bias": torch.Tensor of shape (L, q)
            "coupling_matrix": torch.Tensor of shape (L, q, L, q)
    """
    params = {key: value.clone() for key, value in params.items()}

    for key in params:
        if key.startswith("bias"):
            params[key] = zerosum_gauge_bias(params[key])
        elif key.startswith("coupling"): 
            params[key] = zerosum_gauge_couplings(params[key])
    
    return params