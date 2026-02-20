import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import utils

class SelexRoundDataset(Dataset):
    def __init__(
        self, 
        seq_oh: torch.Tensor  # n_seq * L * q
    ):
        super().__init__()
        assert seq_oh.dim() == 3
        self.seq_oh = seq_oh

    def __len__(self):
        return self.seq_oh.shape[0]

    def __getitem__(self, idx):
        return self.seq_oh[idx]
    
class SelexRoundDataLoader:
    def __init__(
        self,
        seq_oh: torch.Tensor,
        batch_size = None,
        device = None,
        generator = None
    ):
        if device is None:
            device = seq_oh.device
        if generator is None:
            generator = torch.Generator()
        if batch_size is None:
            batch_size = len(seq_oh)
        elif batch_size > len(seq_oh):
            raise ValueError(f"Batch size cannot be larger than number of sequences, got {batch_size} and {len(seq_oh)}.")
        self.seq_oh = seq_oh
        self.batch_size = batch_size
        self.device = device
        self.generator = generator

    def get_batch(
        self,
        generator = None 
    ):
        if generator is None:
            generator = self.generator
        nseq = self.seq_oh.shape[0]
        # generate random indices on the same device as data
        perm = torch.randperm(nseq, generator=generator).to(self.seq_oh.device)
        idx = perm[:self.batch_size]
        return self.seq_oh[idx].to(self.device)
    
    def __iter__(self):
        return SelexRoundDataLoaderIterator(data_loader=self)

class SelexRoundDataLoaderIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        nseq = self.data_loader.seq_oh.shape[0]
        self.perm = torch.randperm(nseq, generator=data_loader.generator).to(self.data_loader.seq_oh.device)
        self.count = 0
        self.max_count = nseq // self.data_loader.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.max_count:
            raise StopIteration
        bs = self.data_loader.batch_size
        idx = self.perm[self.count * bs : (self.count + 1) * bs]
        batch = self.data_loader.seq_oh[idx].to(self.data_loader.device)
        self.count += 1
        return batch
    
# def train_test_split(sequences_oh, batch_size, split, device=torch.device('cpu')):
#     seq_train = []; seq_valid = []
#     for seq_oh in sequences_oh:
#         t, v = torch.utils.data.random_split(seq_oh, split)
#         seq_train.append(t.dataset[t.indices])
#         seq_valid.append(v.dataset[v.indices])
#     data_loaders_train = [SelexRoundDataLoader(seq, batch_size=batch_size, device=device) for seq in seq_train]
#     data_loaders_valid = [SelexRoundDataLoader(seq, batch_size=batch_size, device=device) for seq in seq_valid]

#     return data_loaders_train, data_loaders_valid

def train_test_split_plot(counts_round, lims_high, lims_low, n_high, n_low):
    n_seq_unique = len(counts_round)
    
    valid_idx_high_bool = (lims_high[0] <= counts_round) * (counts_round <= lims_high[1])
    valid_idx_low_bool = (lims_low[0] <= counts_round) * (counts_round <= lims_low[1])
    nhigh = valid_idx_high_bool.sum().item()
    nlow = valid_idx_low_bool.sum().item()
    valid_idx_high = torch.arange(n_seq_unique)[valid_idx_high_bool].tolist()
    valid_idx_low = torch.arange(n_seq_unique)[valid_idx_low_bool].tolist()
    
    print(f'Found {nlow} unique sequences with low count and {nhigh} unique sequences with high count')

    idx_high = random.sample(valid_idx_high, n_high)
    idx_low = random.sample(valid_idx_low, n_low)
    
    idx_valid = torch.tensor(idx_high + idx_low)
    assert len(idx_valid) == n_high + n_low

    idx_train_bool = torch.ones(n_seq_unique, dtype=bool)
    idx_train_bool[idx_valid] = 0
    idx_train = torch.arange(n_seq_unique)[idx_train_bool]
    
    n_train = len(idx_train)
    n_valid = len(idx_valid)
    
    print(f'Training sequences (unique): {n_train}')
    print(f'Validation sequences (unique): {n_valid}')
    
    fig, ax = plt.subplots(figsize=(4,2))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.hist(counts_round[valid_idx_low], bins=torch.logspace(0, 4, 100), color=colors[0], alpha=0.1)
    ax.hist(counts_round[idx_low], bins=torch.logspace(0, 4, 100), label='low count', color=colors[0])
    ax.hist(counts_round[valid_idx_high], bins=torch.logspace(0, 4, 100), color=colors[1], alpha=0.1)
    ax.hist(counts_round[idx_high], bins=torch.logspace(0, 4, 100), label='high count', color=colors[1])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('count')
    ax.legend()
    ax.set_title('Counts of unique sequences at round 0', fontsize=9)

    return idx_train, idx_valid

def split_train_test(sequences_unique_all, counts_unique, idx_train, idx_valid, 
                     device=torch.device('cpu')):
    n_rounds = len(counts_unique)
    sequences_train = []
    sequences_valid = []
    log_multinomial_factors_train = []
    log_multinomial_factors_valid = []
    total_reads_train = []
    total_reads_valid = []
    
    for t in range(n_rounds):
        sequences_train.append(torch.repeat_interleave(sequences_unique_all[idx_train], counts_unique[t][idx_train], dim=0))
        sequences_valid.append(torch.repeat_interleave(sequences_unique_all[idx_valid], counts_unique[t][idx_valid], dim=0))
        log_multinomial_factors_train.append(utils.log_multinomial(counts_unique[t][idx_train]))
        log_multinomial_factors_valid.append(utils.log_multinomial(counts_unique[t][idx_valid]))
        total_reads_train.append(counts_unique[t][idx_train].sum().item())
        total_reads_valid.append(counts_unique[t][idx_valid].sum().item())
    
    log_multinomial_factors_train = torch.tensor(log_multinomial_factors_train).to(device)
    log_multinomial_factors_valid = torch.tensor(log_multinomial_factors_valid).to(device)
    total_reads_train = torch.tensor(total_reads_train).to(device)
    total_reads_valid = torch.tensor(total_reads_valid).to(device)
    
    sequences_train_oh = [utils.one_hot(s) for s in sequences_train]
    sequences_valid_oh = [utils.one_hot(s) for s in sequences_valid]

    return (sequences_train_oh, total_reads_train, log_multinomial_factors_train), (sequences_valid_oh, total_reads_valid, log_multinomial_factors_valid)