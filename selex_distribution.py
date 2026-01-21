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
        minus_en_ = tuple(mode.compute_energy(x) for mode in self.modes)
        minus_en = torch.stack(minus_en_, dim=1)
        if self.normalized == True:
            minus_en = minus_en - minus_en.logsumexp(dim=1, keepdim=True)

        # first pick only the selected rounds, then (log)sum(exp) over modes, then sum over rounds
        return -(-minus_en[:,None,:] + torch.log(selected)).logsumexp(dim=-1).sum(-1)
        
    
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

    # compute $\sum_{\tau \in \mathcal A(a(t))} \log p_{s,\tau} + logNs0
    def compute_energy_up_to_parent_round(self, x, t):
        if t == 0:
            return torch.zeros(x.size(0), device=x.device)
        at = self.tree.get_parent(t-1)
        return self.compute_energy_up_to_round(x, at+1)

    def get_n_rounds(self):
        return self.tree.get_n_nodes()

    def metropolis_step_uniform_sites(
        self,
        chains: torch.Tensor,
        t: int,
        beta: float = 1.0
        # random number generator!
    ):
        energy_curr = self.compute_energy_up_to_round(chains, t)
        device = chains.device
        dtype = chains.dtype
        n_chains, L, q = chains.shape

        idx = torch.randint(0, L, (1,), device=device)[0]
        res_new = one_hot(torch.randint(0, q, (n_chains,), device=device), num_classes=q).to(dtype)
        chains_new = chains.clone()
        chains_new[:, idx, :] = res_new

        energy_new = self.compute_energy_up_to_round(chains_new, t)
        delta_energy = energy_new - energy_curr
        accept_prob = torch.exp(-beta * delta_energy)
        rand_vals = torch.rand(n_chains, device=device, dtype=dtype)
        accepted = accept_prob > rand_vals

        # Only update accepted chains at the chosen site
        chains[accepted, idx, :] = res_new[accepted]

    def sample_metropolis_uniform_sites(
        self,
        chains: torch.Tensor,
        t: int,
        n_sweeps: int,
        beta: float = 1.0,
        # random number generator!
    ):
        L = chains.size(1)
        n_steps = n_sweeps * L
        with torch.no_grad():
            for step in torch.arange(n_steps):
                self.metropolis_step_uniform_sites(chains.select(0, t), t, beta)