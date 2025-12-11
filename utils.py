from typing import Dict, Callable, Tuple, List, Any
import torch

def normalize_to_prob(x):
    assert(len(x.shape) == 2)
    norm = x.sum(dim=-1, keepdim=True)
    return x / norm

def normalize_to_logprob(x):
    assert(len(x.shape) == 2)
    norm = x.logsumexp(dim=-1, keepdim=True)
    return x - norm

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
            bias = params[key]
            bias -= bias.mean(dim=1, keepdim=True)
            params[key] = bias
        elif key.startswith("coupling"): 
            coupling_matrix = params[key]
            coupling_matrix -= coupling_matrix.mean(dim=1, keepdim=True) + \
                            coupling_matrix.mean(dim=3, keepdim=True) - \
                            coupling_matrix.mean(dim=(1, 3), keepdim=True)
            params[key] = coupling_matrix
    
    return params