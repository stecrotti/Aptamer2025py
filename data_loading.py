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
        batch_size,
        device = None,
        generator = None
    ):
        if device is None:
            device = seq_oh.device
        if generator is None:
            generator = torch.Generator()
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
    