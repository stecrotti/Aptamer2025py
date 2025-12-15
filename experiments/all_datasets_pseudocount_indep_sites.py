import torch
import numpy as np
import matplotlib.pyplot as plt
import adabmDCA

from adabmDCA.utils import get_device, get_dtype, get_mask_save
from adabmDCA.sampling import get_sampler

import sys
sys.path.append('..')
import indep_sites
import utils

import pickle

def run_with_pseudocount(experiment_idx, pseudocount):
    experiment_ids = ['Dop8V030', 'Dop8V930', 'Dop8V2430'] 
    
    experiment_id = experiment_ids[experiment_idx]
    round_ids = ["ARN", "R01", "R02N"]
    
    device = get_device("")
    dtype = get_dtype("float32")
    
    pc_str = format(pseudocount, '.8f')
    
    filename = experiment_id + "freq_pseudocount" + pc_str + ".pkl"
    filepath = "saved/freq_pseudocount/" + filename
    
    with open(filepath, 'rb') as f:
        fi, fij, total_reads = pickle.load(f)   
  
    lr = 0.01
    max_epochs = 10**5
    
    params=indep_sites.init_parameters(fi)
    
    params, history = indep_sites.train(
        fi=fi,
        total_reads=total_reads, 
        params=params,
        lr=lr,
        max_epochs=max_epochs,
        target_error=1e-6,
        progress_bar=False)

    filename = experiment_id + "_" + pc_str + ".pkl"
    filepath = "saved/pseudocount/indep_sites/" + filename
    
    data = [experiment_id, round_ids, params, pseudocount]
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    experiment_id = int(sys.argv[1])
    pseudocount = float(sys.argv[2])
    run_with_pseudocount(experiment_id, pseudocount)