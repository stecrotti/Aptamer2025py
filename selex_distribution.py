import torch
from torch.nn import Module
from tree import Tree
from utils import one_hot

class EnergyModel(Module):
    def __init__(self):
        super().__init__()
    
class MultiModeDistribution(torch.nn.Module):
    def __init__(
        self,
        *modes,
        normalized: bool = True
    ):
        super().__init__()
        for mode in modes:
            assert isinstance(mode, EnergyModel), f"Expected mode of type `EnergyModel`, got {type(mode)}"
        self.modes = torch.nn.ModuleList(modes)
        self.normalized = normalized

    def get_n_modes(self):
        return len(self.modes) 
    
    def compute_energy(
            self,
            x: torch.Tensor, # batch_size * L * q
            selected: torch.BoolTensor, # n_rounds * n_modes
    ):
        minus_en_tuple = tuple(- mode.compute_energy(x) for mode in self.modes)
        minus_en = torch.stack(minus_en_tuple, dim=1)
        if self.normalized == True:
            minus_en = minus_en - minus_en.logsumexp(dim=1, keepdim=True)
        # pick only the selected rounds 
        en_selected =  - torch.where(selected, minus_en.unsqueeze(1), torch.inf)
        # (log)sum(exp) over modes, then sum over rounds
        en = en_selected.logsumexp(dim=-1).sum(-1)
        return en
        
    
class MultiRoundDistribution(torch.nn.Module):
    def __init__(
        self,
        round_zero: EnergyModel,
        selection: MultiModeDistribution,
        tree: Tree,
        selected_modes: torch.BoolTensor,   # (n_rounds * n_modes) modes selected for at each round
    ):
        if selection.get_n_modes() != selected_modes.size(1):
            raise ValueError(f"Number of modes must coincide for selection probability, got {selection.get_n_modes()} and {selected_modes.size(1)}")
        super().__init__()
        self.round_zero = round_zero
        self.selection = selection
        self.tree = tree
        self.selected_modes = selected_modes

    # compute $\log p_{st}
    def selection_energy_at_round(self, x, t):
        if t == 0:
            return torch.zeros(x.size(0))
        return self.selection.compute_energy(x, selected=self.selected_modes[t-1])

    # compute $\sum_{\tau \in \mathcal A(t)} \log p_{s,\tau}
    def selection_energy_up_to_round(self, x, t):
        if t == 0:
            return torch.zeros(x.size(0), device=x.device)
        ancestors = self.tree.ancestors_of(t-1)
        return self.selection.compute_energy(x, selected=self.selected_modes[ancestors])

    # compute $\sum_{\tau \in \mathcal A(t)} \log p_{s,\tau} + logNs0
    def compute_energy_up_to_round(self, x, t):
        if t == -1:
            return torch.zeros(x.size(0), device=x.device)
        logNs0 = - self.round_zero.compute_energy(x)
        logps = - self.selection_energy_up_to_round(x, t)
        return - (logps + logNs0)

    def get_n_rounds(self):
        return self.tree.get_n_nodes()
    
    def _apply(self, fn):
        super()._apply(fn)
        
        self.tree.parent = fn(self.tree.parent)
        self.tree.ancestors_flat = fn(self.tree.ancestors_flat)
        self.tree.offset = fn(self.tree.offset)
        self.tree.length = fn(self.tree.length)
        self.selected_modes = fn(self.selected_modes)
        # self.round_zero = self.round_zero._apply(fn)
        # self.selection = self.selection._apply(fn)
        
        return self