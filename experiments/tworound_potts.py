import sys
sys.path.append('..')
import selex_dca, utils

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import utils
import selex_dca
import selex_distribution, energy_models, tree, data_loading, training, callback, sampling, diagnostic

from matplotlib import cm
import pickle

def train_tworound_potts(
    sequences_oh, 
    total_reads, 
    log_multinomial_factors,
    max_epochs,
    batch_size = 10**6,
    n_chains = None,
    weight_decay = 0.0,
    lr = 1e-2,
    n_sweeps = 10,
    dtype = torch.float32,
    device = torch.device('cpu'),
    checkpoint_filename = None,
    checkpoint_every = 500,
    save_params_every = 100
):
    if n_chains is None:
        if device == torch.device('cpu'):
            n_chains = 10**4
        else:
            n_chains = 10**5

    total_reads = total_reads.to(device)
    log_multinomial_factors = log_multinomial_factors.to(device)

    tr = tree.Tree()
    tr.add_node(-1, name = "R01")
    tr.add_node(0, name = "R02N")
    
    selected_modes = torch.BoolTensor(
        [[1], [1]]
    )

    L, q = sequences_oh[0][0].shape
    n_rounds = len(sequences_oh) 

    k, h = training.init_from_indep_sites(sequences_oh, total_reads)
    J = torch.zeros(L, q, L, q, dtype=dtype)
    Ns0 = energy_models.IndepSites(k)
    potts = energy_models.Potts(J, h)
    ps = selex_distribution.MultiModeDistribution(potts, normalized=False)
    model = selex_distribution.MultiRoundDistribution(Ns0, ps, tr, selected_modes).to(device)

    data_loaders = [data_loading.SelexRoundDataLoader(seq_oh, batch_size=batch_size, device=device) for seq_oh in sequences_oh]
    chains = training.init_chains(n_rounds, n_chains, L, q, dtype=dtype, device=device)
    log_weights = torch.zeros(n_rounds, n_chains, dtype=dtype, device=device)
    
    optimizer = torch.optim.SGD([
        {'params': (model.round_zero.h), 'lr': 10*lr},
        {'params': (model.selection.modes[0].J,), 'weight_decay': weight_decay},
        {'params': (model.selection.modes[0].h,)}
    ], lr=lr, weight_decay=0.0)

    callbacks = [callback.ConvergenceMetricsCallback(), callback.PearsonCovarianceCallback(), 
                 callback.CheckpointCallback(save_every=checkpoint_every, checkpoint_filename=checkpoint_filename, delete_old_checkpoints=True),
                 callback.ParamsCallback(save_every=save_params_every)]

    training.train(model, data_loaders, total_reads, chains, n_sweeps, max_epochs,
               optimizer=optimizer, callbacks=callbacks, log_weights=log_weights,
               log_multinomial_factors=log_multinomial_factors)

    return model, data_loaders, chains, optimizer, log_weights, callbacks

def resume_training(model, data_loaders, chains, optimizer, log_weights, callbacks, 
                    max_epochs, total_reads, log_multinomial_factors, n_sweeps = 10):
    
    training.train(model, data_loaders, total_reads, chains, n_sweeps, max_epochs,
           optimizer=optimizer, callbacks=callbacks, log_weights=log_weights,
           log_multinomial_factors=log_multinomial_factors)
    
    return model, data_loaders, chains, optimizer, log_weights, callbacks