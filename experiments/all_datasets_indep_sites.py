import torch
import numpy as np
import matplotlib.pyplot as plt
import adabmDCA

from adabmDCA.utils import get_device, get_dtype, get_mask_save
from adabmDCA.sampling import get_sampler

import sys
sys.path.append('..')
import indep_sites
import utils, selex_distribution, energy_models, tree, data_loading, training, callback, sampling

import pickle

def run(experiment_id_idx):
    experiment_ids = ['Dop8V030', 'Dop8V930', 'Dop8V2430'] 
    
    experiment_id = experiment_ids[experiment_id_idx]
    round_ids = ["ARN", "R01", "R02N"]
    
    device = get_device("")
    dtype = get_dtype("float32")

    fi, total_reads = torch.load(f'saved/fi_{experiment_id}.pt')
  
    max_epochs = 10**4
    
    params=indep_sites.init_parameters(fi)
    history=indep_sites.init_history()

    lrs = torch.logspace(-2, -15, 5)

    for lr in lrs:
        params, history = indep_sites.train(
            fi=fi,
            total_reads=total_reads, 
            params=params,
            lr=lr,
            max_epochs=max_epochs,
            target_error=1e-16,
            history=history,
            progress_bar=False)

    tr = tree.Tree()
    tr.add_node(-1, name = "R01")
    tr.add_node(0, name = "R02N")
    
    selected_modes = torch.BoolTensor(
        [[1], [1]]
    )
    
    k = params['bias_Ns0']
    h = params['bias_ps']  
    Ns0 = energy_models.IndepSites(k)
    potts = energy_models.IndepSites(h)
    ps = selex_distribution.MultiModeDistribution(potts, normalized=False)
    model = selex_distribution.MultiRoundDistribution(Ns0, ps, tr, selected_modes)

    torch.save(model, f'saved/{experiment_id}_indep_sites.pt')

if __name__ == "__main__":
    experiment_id = int(sys.argv[1])
    run(experiment_id)