import torch
from torch.nn import Module
from tree import Tree

class EnergyModel(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_energy_stacked(x, *params):
        """
        Default fallback: assumes first dimension of each param is the stacking dimension.
        Calls the class's compute_energy for each set of parameters.
        """
        N = params[0].shape[0]
        results = []
        for i in range(N):
            # Extract i-th set of parameters for this model
            param_i = [p[i] for p in params]
            # Create a temporary model instance with these parameters
            model = type(params[0]).__self_class__(*param_i)
            results.append(model.compute_energy(x))
        return torch.stack(results, dim=1)  # (B, N)
    
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
    
    def _compute_energy_modes(self, x: torch.Tensor):
        return torch.stack(
            [mode.compute_energy(x) for mode in self.modes],
            dim=1
        )
    
    def compute_energy(
            self,
            x: torch.Tensor, # batch_size * L * q
            selected: torch.BoolTensor, # n_rounds * n_modes
    ):
        minus_en = self._compute_energy_modes(x)
        if self.normalized == True:
            minus_en = minus_en - minus_en.logsumexp(dim=1, keepdim=True)

        # first pick only the selected rounds, then (log)sum(exp) over modes, then sum over rounds
        return -(-minus_en[:,None,:] + torch.log(selected)).logsumexp(dim=-1).sum(-1)
    
class HomogeneousMultiModeDistribution(MultiModeDistribution):
    def __init__(
        self,
        *modes,
        normalized: bool = True
    ):
        super().__init__(*modes, normalized=normalized)
        mode_type = type(modes[0])
        for mode in modes:
            assert type(mode) == mode_type
        self.mode_type = mode_type
        self.tensors_stacked = tuple(torch.stack(params) for params in zip(*[mode.parameters() for mode in modes]))

    def _compute_energy_modes(self, x: torch.Tensor):
        return self.mode_type.compute_energy_stacked(x, *self.tensors_stacked)
    
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
            return torch.zeros(x.size(0))
        ancestors = self.tree.ancestors_of(t-1)
        return self.selection.compute_energy(x, selected=self.selected_modes[ancestors])

    # compute $\sum_{\tau \in \mathcal A(t)} \log p_{s,\tau} + logNs0
    def compute_energy_up_to_round(self, x, t):
        if t == -1:
            return torch.zeros(x.size(0))
        logNs0 = - self.round_zero.compute_energy(x)
        logps = - self.selection_energy_up_to_round(x, t-1)
        return - (logps + logNs0)

    # compute $\sum_{\tau \in \mathcal A(a(t))} \log p_{s,\tau} + logNs0
    def compute_energy_up_to_parent_round(self, x, t):
        at = self.tree.parent[t-1]
        return self.compute_energy_up_to_round(x, at+1)

    def get_n_rounds(self):
        return self.tree.get_n_nodes()