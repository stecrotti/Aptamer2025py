import torch
from torch.nn import Module
from tree import Tree
import sampling


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
            selection_strength: torch.Tensor = None
    ):
        if selection_strength is None:
            selection_strength = torch.ones(selected.size(0), dtype=x.dtype, device=x.device)
        minus_en_tuple = tuple(- mode.compute_energy(x) for mode in self.modes)
        minus_en = torch.stack(minus_en_tuple, dim=1)
        if self.normalized == True:
            minus_en = minus_en - minus_en.logsumexp(dim=1, keepdim=True)
        # pick only the selected rounds 
        en_selected =  - torch.where(selected, minus_en.unsqueeze(1), torch.inf)
        # (log)sum(exp) over modes, then sum over rounds
        en = en_selected.logsumexp(dim=-1)
        return (en * selection_strength[None,:]).sum(1)
        
    
class MultiRoundDistribution(torch.nn.Module):
    def __init__(
        self,
        round_zero: EnergyModel,
        selection: MultiModeDistribution,
        tree: Tree,
        selected_modes: torch.BoolTensor,   # (n_rounds * n_modes) modes selected for at each round
        selection_strength: torch.Tensor | None = None,
        learn_selection_strength: bool = False,
        dtype = torch.float32
    ):
        if selection.get_n_modes() != selected_modes.size(1):
            raise ValueError(f"Number of modes must coincide for selection probability and selected modes, got {selection.get_n_modes()} and {selected_modes.size(1)}")
        n_selection_rounds = selected_modes.size(0)
        if tree.get_n_nodes() != n_selection_rounds:
            raise ValueError(f"Number of selection rounds must coincide for tree and selected modes, got {tree.get_n_nodes()} and {n_selection_rounds}")
        super().__init__()
        self.round_zero = round_zero
        self.selection = selection
        self.tree = tree
        self.selected_modes = selected_modes
        if selection_strength is None:
            selection_strength = torch.ones(n_selection_rounds, device=selected_modes.device, dtype=dtype)
        elif selection_strength.size(0) != n_selection_rounds:
            raise ValueError(f"Length of selection strength vector must coincide with number of selection rounds, got {selection_strength.size(0)} and {n_selection_rounds}")
        if learn_selection_strength:
            self.selection_strength = torch.nn.Parameter(selection_strength)
        else:
            self.selection_strength = selection_strength
        self.learn_selection_strength = learn_selection_strength

    # compute $\log p_{st}
    def selection_energy_at_round(self, x, t):
        if t == 0:
            return torch.zeros(x.size(0))
        abs_selection_strength = torch.square(self.selection_strength)
        normalized_selection_strength = abs_selection_strength / abs_selection_strength.sum(0, keepdim=True)
        return self.selection.compute_energy(x, selected=self.selected_modes[t-1:t], 
                                             selection_strength=normalized_selection_strength[t-1:t])

    # compute $\sum_{\tau \in \mathcal A(t)} \log p_{s,\tau}
    def selection_energy_up_to_round(self, x, t):
        if t == 0:
            return torch.zeros(x.size(0), device=x.device)
        ancestors = self.tree.ancestors_of(t-1)
        normalized_selection_strength = self.selection_strength / self.selection_strength.sum(0, keepdim=True)
        return self.selection.compute_energy(x, selected=self.selected_modes[ancestors],
                                             selection_strength=normalized_selection_strength[ancestors])

    # compute $\sum_{\tau \in \mathcal A(t)} \log p_{s,\tau} + logNs0
    def compute_energy_up_to_round(self, x, t):
        if t == -1:
            return torch.zeros(x.size(0), device=x.device)
        logNs0 = - self.round_zero.compute_energy(x)
        logps = - self.selection_energy_up_to_round(x, t)
        return - (logps + logNs0)

    def get_n_rounds(self):
        return self.get_n_selection_rounds() + 1
    
    def get_n_selection_rounds(self):
        return self.tree.get_n_nodes()
    
    def _apply(self, fn):
        super()._apply(fn)
        
        self.tree.parent = fn(self.tree.parent)
        self.tree.ancestors_flat = fn(self.tree.ancestors_flat)
        self.tree.offset = fn(self.tree.offset)
        self.tree.length = fn(self.tree.length)
        self.selected_modes = fn(self.selected_modes)
        if not self.learn_selection_strength:
            self.selection_strength = fn(self.selection_strength)
        
        return self

    def sample(self, chains, n_sweeps, beta=1.0):
        n_rounds, n_chains, L, q = chains.size()
        energies = torch.zeros((n_rounds, n_chains), device=chains.device, dtype=chains.dtype)
        for t in torch.arange(n_rounds):
            with torch.no_grad():
                energy_t = sampling.sample_metropolis(self, chains, t, n_sweeps, beta=beta)
                energies[t] = energy_t
        
        return energies
