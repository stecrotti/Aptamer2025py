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
        minusE = torch.Tensor()
        for mode in self.modes:
            minusEw = - mode.compute_energy(x)
            minusE = torch.cat((minusE, minusEw[:,None]), dim=1)
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

     # compute $\sum_{\tau \in \mathcal A(t)} \log p_{s,\tau}
     def selection_energy_up_to_round(self, x, t):
        if t == -1:
            return torch.zeros(x.size(0))
        ancestors = self.round_tree.ancestors_of(t)
        logps_modes = self.selection.compute_logprobabilities(x)
        ancestors = self.round_tree.ancestors_of(t)
        selected = self.round_tree.selected_modes[ancestors]
        # first pick only the selected rounds, then (log)sum(exp) over modes, then sum over rounds
        return - (logps_modes[:,None,:] + torch.log(selected)).logsumexp(dim=-1).sum(1)

     # compute sum_tau log p_{s,tau}
     @torch.compile
     def compute_energy_up_to_round(self, x, t):
         logNs0 = - self.round_zero.compute_energy(x)
         logps = - self.selection_energy_up_to_round(x, t)
         return - (logps + logNs0)

     def get_n_rounds(self):
         return self.round_tree.get_n_rounds()