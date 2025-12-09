import torch
from typing import Dict, Tuple, Any
from tqdm.autonotebook import tqdm

def init_parameters(fi: torch.Tensor) -> Dict[str, torch.Tensor]:
    _, L, q = fi.shape
    params = {}
    params["bias_Ns0"] = torch.log(fi[0])   # initialize with frequencies at first round
    # params["bias_ps"] = torch.zeros((L, q), device=fi.device, dtype=fi.dtype)
    params["bias_ps"] = torch.log(fi[-1]) - torch.log(fi[0])
    # normalize_to_logprob(params["bias_ps"])
    
    return params

def get_params_at_round(params: Dict[str, torch.Tensor], t: int):
    params_t = {}
    bias = params["bias_Ns0"] + t * params["bias_ps"]
    # params_t["bias"] = bias - bias.logsumexp(dim=1, keepdim=True)
    # params_t['bias'] = normalize_to_logprob(bias)
    # params_t["bias"] = bias - bias.mean(dim=1, keepdim=True)
    params_t["bias"] = bias
    
    return params_t

def get_params_ps(params):
    return {"bias": params["bias_ps"]}

def compute_energy(
    x: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:

    batch_size = x.shape[0]
    x_flat = x.view(batch_size, -1)
    bias_flat = params["bias"].view(-1)
    energy = - x_flat @ bias_flat
    
    return energy

@torch.jit.script
def compute_gradient_and_loglikelihood(
    fi: torch.Tensor,
    pi: torch.Tensor,
    total_reads: torch.Tensor,
    ts: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], float]:
    
    di = fi - pi    
    
    grad = {}
    W = total_reads.sum()
    ll = (total_reads[:,None,None] * fi * pi.log()).sum() / W
    grad["bias_Ns0"] = (total_reads[:,None,None] * di).sum(dim=0) / W
    grad["bias_ps"] = ((ts * total_reads)[:,None,None] * di).sum(dim=0) / W
    
    return grad, ll


def update_params(
    fi: torch.Tensor,
    pi: torch.Tensor,
    total_reads: torch.Tensor,
    params: Dict[str, torch.Tensor],
    lr: float,
    l2reg: float = 0.0
) -> Tuple[Dict[str, torch.Tensor], float]:
    
    ts = torch.arange(len(total_reads))
    
    # Compute the gradient
    grad, ll = compute_gradient_and_loglikelihood(
        fi=fi, pi=pi, total_reads=total_reads, ts=ts)
    
    # Update parameters
    with torch.no_grad():
        for key in params:
            params[key] += lr * (grad[key] + l2reg * params[key])
            # params[key] -= params[key].logsumexp(dim=1, keepdims=True)
    
    return params, ll

def normalize_to_prob(x):
    assert(len(x.shape) == 2)
    norm = x.sum(dim=-1, keepdim=True)
    return x / norm

def normalize_to_logprob(x):
    assert(len(x.shape) == 2)
    norm = x.logsumexp(dim=-1, keepdim=True)
    return x - norm


def train(
    fi: torch.Tensor,
    total_reads: torch.Tensor,
    params: Dict[str, torch.Tensor],
    lr: float = 1e-2,    
    max_epochs: int = 5*10**4,
    target_error: float = 1e-12,
    l2reg: float = 0.0,
    progress_bar: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:

    history = {
        "epochs": [],
        "log-likelihood": [],
        "err": []
    }

    def halt_condition(epochs, err):
        c1 = (epochs > max_epochs)
        c2 = (err < target_error)
        return c1 + c2
    
    epochs = 0
    err = torch.inf

    if progress_bar: 
        pbar = tqdm(
            initial=0,
            total=max_epochs,
            miniters=100,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="{desc} {percentage:.2f}%[{bar}] Epoch: {n}/{total_fmt} [{elapsed}]"
        )
        pbar.set_description(f"Epochs: {epochs} - LL: {log_likelihood:.2f}")
    

    while not halt_condition(epochs, err):
        # Store the previous parameters
        params_prev = {key: value.clone() for key, value in params.items()}
        ts = torch.arange(len(total_reads))
        pi = torch.stack([normalize_to_prob(
                    get_params_at_round(params, t)["bias"].exp()
                    )
                    for t in ts])
        
        # Update the parameters
        params, log_likelihood = update_params(
            fi=fi,
            pi=pi,
            total_reads=total_reads,
            params=params,
            lr=lr,
            l2reg=l2reg
        )

        epochs += 1
        err = 0
        for key in params:
            d = (params[key] - params_prev[key]).abs()
            err = max(err, d.max().item())

        if progress_bar:
            pbar.n = epochs
            pbar.set_description(f"Epochs: {epochs} - Error: {err:.2e} - LL: {log_likelihood:.2f}")
        
        history["epochs"].append(epochs)
        history["log-likelihood"].append(log_likelihood)
        history["err"].append(err)
                
    if progress_bar:
        pbar.close()
        
    return params, history