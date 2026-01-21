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
    
def get_batch(
    sequences_oh: torch.Tensor,
    batch_size: int,
    device = None,
    generator = None 
):
    if device is None:
        device = sequences_oh.device
    if generator is None:
        generator = torch.Generator()
    nseq = sequences_oh.shape[0]
    perm = torch.randperm(nseq, generator=generator)
    idx = perm[:batch_size]
    return sequences_oh[idx].to(device)
    