import torch
from selex_distribution import EnergyModel

class IndepSites(EnergyModel):
    def __init__(
        self,
        h: torch.Tensor # Lxq tensor
    ):
        super().__init__()
        n_dim = h.dim()
        if n_dim != 2:
            raise ValueError(f"Expected tensor with 2 dimensions, got {n_dim}.")

        self.h = torch.nn.Parameter(h)

    def get_n_states(self):
        return self.h.size(1)

    def get_sequence_length(self):
        return self.h.size(0)

    def compute_energy(
        self,
        x: torch.Tensor
    ):

        L = self.get_sequence_length()
        q = self.get_n_states()
        x_flat = x.view(-1, L * q)
        bias_flat = self.h.view(L * q)

        return - x_flat @ bias_flat

    def forward(self, x):
        return self.compute_energy(x)
    
    def set_zerosum_gauge(self):
        h = self.h.detach().clone()
        h = h - h.mean(dim=1, keepdim=True)
        return IndepSites(h)

class Potts(EnergyModel):
    def __init__(
        self,
        J: torch.Tensor, # LxqxLxq tensor
        h: torch.Tensor, # Lxq tensor
    ):
        super().__init__()
        sz_h = h.size()
        sz_J = J.size()
        if len(sz_h) != 2:
            raise ValueError(f"Expected tensor with 2 dimensions, got {len(sz_h)}.")
        if len(sz_J) != 4:
            raise ValueError(f"Expected tensor with 2 dimensions, got {len(sz_J)}.")
        if not (sz_J[0:2] == sz_J[2:4] == sz_h):
            raise ValueError("Wrong tensor dimensions")
        
        self.h = torch.nn.Parameter(h)

        L, q = sz_h
        mask = torch.ones(L, q, L, q)
        # set the (i,i) blocks to zero
        mask[torch.arange(L), :, torch.arange(L), :] = 0
        self.J = torch.nn.Parameter(J)
        self.mask = mask

    def get_n_states(self):
        return self.h.size(1)

    def get_sequence_length(self):
        return self.h.size(0)

    def compute_energy(
        self,
        x: torch.Tensor
    ):
        L = self.get_sequence_length()
        q = self.get_n_states()
        # the -1 accounts for possible batch index along dimension 0
        x_flat = x.view(-1, L * q)
        bias_flat = self.h.view(L * q)
        couplings_flat = (self.J * self.mask).reshape(L * q, L * q)
        bias_term = x_flat @ bias_flat
        coupling_term = torch.sum(x_flat * (x_flat @ couplings_flat), dim=1)
        return - bias_term - 0.5 * coupling_term

    def forward(self, x):
        return self.compute_energy(x)

    def set_zerosum_gauge(self):
        h = self.h.detach().clone()
        h = h - h.mean(dim=1, keepdim=True)

        J = self.J.detach() * self.mask.detach()
        J = J - (
            J.mean(dim=1, keepdim=True)
            + J.mean(dim=3, keepdim=True)
            - J.mean(dim=(1, 3), keepdim=True)
        )

        return Potts(J, h)
        
# used as dummy for checks
class InfiniteEnergy(EnergyModel):
    def __init__(self):
        super().__init__()

    def compute_energy(
        self,
        x: torch.Tensor
    ):
        if x.dim() == 2:
            return torch.full((1,), torch.inf)
        elif x.dim() == 3:
            return torch.full((x.size(0),), torch.inf)
        else:
            raise ValueError(f"Expected tensor `x` of dimension either 2 or 3, got {x.dim()}")

    def forward(self, x):
        return self.compute_energy(x)
    
# wrapper around any torch.nn.Module. the module's `forward` is assumed to compute the energy
class GenericEnergyModel(EnergyModel):
    def __init__(
        self,
        model: torch.nn.Module
    ):
        super().__init__()
        self.model = model

    def compute_energy(
        self,
        x: torch.Tensor
    ):
        batch_size, L, q = x.size()
        x_flat = x.view(-1, L * q)

        return self.model(x_flat).squeeze()

    def forward(self, x):
        return self.compute_energy(x)