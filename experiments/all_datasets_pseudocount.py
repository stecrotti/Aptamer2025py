import torch
import numpy as np
import matplotlib.pyplot as plt
import adabmDCA

from adabmDCA.utils import get_device, get_dtype, get_mask_save
from adabmDCA.sampling import get_sampler

import sys
sys.path.append('..')
from selex_dca import *

import pickle

def run_with_pseudocount(experiment_idx, pseudocount):
    experiment_ids = ['Dop8V030', 'Dop8V930', 'Dop8V2430'] 
    max_epochs = [2000, 6000, 3000]
    
    experiment_id = experiment_ids[experiment_idx]
    round_ids = ["ARN", "R01", "R02N"]
    
    device = get_device("")
    dtype = get_dtype("float32")
    
    pc_str = format(pseudocount, '.2f')
    
    filename = experiment_id + "freq_pseudocount" + pc_str + ".pkl"
    filepath = "saved/freq_pseudocount/" + filename
    
    with open(filepath, 'rb') as f:
        fi, fij, total_reads = pickle.load(f)
    
    n_rounds, L, q = fi.size()
    sampler_alg = "gibbs"
    sampler = torch.jit.script(get_sampler(sampler_alg))
    
    nchains = 10000
    
    params = init_parameters(fi=fi) # initialize with frequences at last round
    chains = init_chains(num_rounds=n_rounds, num_chains=nchains, L=L, q=q, device=device, fi=fi)
    
    mask = torch.ones(size=(L, q, L, q), dtype=torch.bool, device=device)
    mask[torch.arange(L), :, torch.arange(L), :] = 0
    # Mask for saving only the upper diagnal part of the weight_matrix
    mask_save = get_mask_save(L=L, q=q, device=device)
    history = init_history()
    log_weights = torch.zeros(n_rounds, nchains, device=device, dtype=dtype)
    
    nsweeps = 10
    lr = 0.01
    
    ch, par, history = train(
        sampler=sampler,
        chains=chains,
        fi=fi,
        fij=fij,
        total_reads=total_reads,
        params=params,
        mask=mask,
        nsweeps=nsweeps,
        lr=lr,
        max_epochs=max_epochs[experiment_idx],
        target_pearson=0.98,
        history=history,
        log_weights=log_weights,
        progress_bar=True
        )

    pearson_final = history["pearson"][-1]

    pc_str = format(pseudocount, '.2f')
    filename = experiment_id + "pseudocount" + pc_str + ".pkl"
    filepath = "saved/pseudocount/" + experiment_id + filename
    
    data = [experiment_id, round_ids, params, pseudocount, pearson_final]
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    experiment_id = int(sys.argv[1])
    pseudocount = float(sys.argv[2])
    run_with_pseudocount(experiment_id, pseudocount)