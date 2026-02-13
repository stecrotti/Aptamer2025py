import torch
from torch.utils.data import Dataset

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
    
def train_test_split(sequences_oh, batch_size, split, device=torch.device('cpu')):
    seq_train = []; seq_valid = []
    for seq_oh in sequences_oh:
        t, v = torch.utils.data.random_split(seq_oh, split)
        seq_train.append(t.dataset[t.indices])
        seq_valid.append(v.dataset[v.indices])
    data_loaders_train = [SelexRoundDataLoader(seq, batch_size=batch_size, device=device) for seq in seq_train]
    data_loaders_valid = [SelexRoundDataLoader(seq, batch_size=batch_size, device=device) for seq in seq_valid]

    return data_loaders_train, data_loaders_valid