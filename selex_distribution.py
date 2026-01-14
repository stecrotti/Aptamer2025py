import torch
from torch.nn import Module
from round_tree import RoundTree

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

    def compute_logprobabilities(
        self,
        x: torch.Tensor,  # batch_size * L * q
    ) -> torch.Tensor: # batch_size * n_modes
        minusE = torch.stack(
            [-mode.compute_energy(x) for mode in self.modes],
            dim=1
        )
        logp = minusE 
        if self.normalized == True:
            logp = logp - minusE.logsumexp(dim=1, keepdim=True)
        return logp   
        
    

class MultiRoundDistribution(torch.nn.Module):
     def __init__(
        self,
        round_zero: EnergyModel,
        selection: MultiModeDistribution,
        round_tree: RoundTree
     ):
        if selection.get_n_modes() != round_tree.get_n_modes():
            raise ValueError(f"Number of modes must coincide for selection probability, got {selection.get_n_modes()} and {round_tree.get_n_modes()}")
        super().__init__()
        self.round_zero = round_zero
        self.selection = selection
        self.round_tree = round_tree

     def _selection_energy(self, x, t_or_ts):
        logps_modes = self.selection.compute_logprobabilities(x)
        selected = self.round_tree.selected_modes[t_or_ts].clone()
        # first pick only the selected rounds, then (log)sum(exp) over modes, then sum over rounds
        return - (logps_modes[:,None,:] + torch.log(selected)).logsumexp(dim=-1).sum(-1)
     
     def selection_energy_at_round(self, x, t):
        if t == -1:
            return torch.zeros(x.size(0))
        return self._selection_energy(x, t)

     # compute $\sum_{\tau \in \mathcal A(t)} \log p_{s,\tau}
     def selection_energy_up_to_round(self, x, t):
        if t == -1:
            return torch.zeros(x.size(0))
        ancestors = self.round_tree.ancestors_of(t)
        return self._selection_energy(x, ancestors)

     # compute sum_tau log p_{s,tau}
    #  @torch.compile
     def compute_energy_up_to_round(self, x, t):
         if t == -2:
            return torch.zeros(x.size(0))
         logNs0 = - self.round_zero.compute_energy(x)
         logps = - self.selection_energy_up_to_round(x, t)
         return - (logps + logNs0)
     
    #  @torch.compile
     def compute_energy_up_to_parent_round(self, x, t):
         at = self.round_tree.tree.parent[t]
         return self.compute_energy_up_to_round(x, at)


     def get_n_rounds(self):
         return self.round_tree.get_n_rounds()