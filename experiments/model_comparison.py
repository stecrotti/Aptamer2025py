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

def log_likelihood_indep_sites(model_indep, sequences_oh, total_reads, log_multinomial_factors):
    models_t = [energy_models.IndepSites(model_indep.round_zero.h + t * model_indep.selection.modes[0].h) for t in range(model_indep.get_n_rounds())]
    log_likelihood_normaliz = total_reads.sum().item()
    
    log_likelihood = 0.0
    for t in range(model_indep.get_n_rounds()):
        Lt = models_t[t].log_prob(sequences_oh[t]).mean()
        log_likelihood += (log_multinomial_factors[t] + total_reads[t] * Lt).item()
    
    return log_likelihood / log_likelihood_normaliz

def estimate_nll_potts(sequences_oh, model, total_reads, log_multinomial_factors, 
                       n_chains = 10**3, n_sweeps = 1, step = 1e-4):
    batches = sequences_oh
    nll_potts = - training.estimate_log_likelihood_AIS(model, batches, total_reads, log_multinomial_factors, n_chains, n_sweeps, step)
    return nll_potts

def sequences_at_distance_1(wt, q):
    seq = []
    L = len(wt)
    for i in range(L):
        delta = torch.zeros(L)
        # use itertools.product here to generalize to distance d
        for a in range(1, q):
            delta[i] = a
            s = torch.fmod(delta+wt, q)
            seq.append(s)
    neigs = torch.stack(seq).to(dtype=torch.int)
    assert torch.all(utils.hamming(wt, neigs) == 1)
    return neigs

def sequences_at_distance_2(wt, q):
    seq = []
    L = len(wt)
    for i in range(L):
        for j in range(L):
            if j != i:
                delta = torch.zeros(L)
                for a in range(1, q):
                    delta[i] = a
                    for b in range(1, q):
                        delta[j] = b
                        s = torch.fmod(delta+wt, q)
                        seq.append(s)
    neigs = torch.stack(seq).to(dtype=torch.int)
    assert torch.all(utils.hamming(wt, neigs) == 2)
    return neigs

def sequences_at_distance_3(wt, q):
    seq = []
    L = len(wt)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if j != i and j != k and i != k:
                    delta = torch.zeros(L)
                    for a in range(1, q):
                        delta[i] = a
                        for b in range(1, q):
                            delta[j] = b
                            for c in range(1, q):
                                delta[k] = c
                                s = torch.fmod(delta+wt, q)
                                seq.append(s)
    neigs = torch.stack(seq).to(dtype=torch.int)
    assert torch.all(utils.hamming(wt, neigs) == 3)
    return neigs

def plot_hist(model, wt, neigs, **kwargs):
    neighbors_oh = utils.one_hot(neigs)
    wt_oh = utils.one_hot(wt)
    logps_neigs = - model.selection_energy_at_round(neighbors_oh, 1).detach()
    logps_wt = - model.selection_energy_at_round(wt_oh, 1).detach()
    fig, ax = plt.subplots(**kwargs)
    ax.hist(logps_neigs, bins=100, density=True)
    ax.axvline(x=logps_wt.item(), color='red', label="wildtype")
    ax.set_xlabel('logps'); ax.set_ylabel('frequency')
    ax.legend()
    return logps_neigs, logps_wt, fig, ax

def best_mutations(model, wt, neigs):
    neighbors_oh = utils.one_hot(neigs)
    wt_oh = utils.one_hot(wt)
    logps_neigs = - model.selection_energy_at_round(neighbors_oh, 1).detach()
    logps_wt = - model.selection_energy_at_round(wt_oh, 1).detach()
    idx_best = logps_neigs >= logps_wt
    logps_best = (logps_neigs - logps_wt)[idx_best]
    perm = torch.argsort(logps_best)
    return torch.argmax(neighbors_oh[idx_best], dim=-1), perm, logps_best

def letter_at_index(seq, idx=None):
    if isinstance(idx, int):
        idx = [idx]
    elif idx is None:
        idx = []
    letters = [utils.TOKENS_DNA[s] for s in seq]
    out = []
    for a in range(len(letters)):
        l = letters[a]
        if a in idx:
            out.append(l)
        else:
            out.append('-')
    return ''.join(out)

def highlight_mutation_and_print(seq, idx=None):
    s = letter_at_index(seq, idx=idx)
    print(s)
    return s

def get_strongest_contact_indices(F, k):
    X = F.view(-1)
    L = F.size(1)
    _, indices = X.topk(2*k)
    idx = torch.cat(((indices // L).unsqueeze(1), (indices % L).unsqueeze(1)), dim=1)
    return idx[torch.arange(0, len(idx), 2)]