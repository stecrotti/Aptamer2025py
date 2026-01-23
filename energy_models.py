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
    
    @staticmethod
    def compute_energy_stacked(x: torch.Tensor, h_stacked):
        """
        x: (B, L, q) input batch
        Returns: (B, N) tensor of energies
        """
        return -torch.einsum('blq,nlq->bn', x, h_stacked)

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
    
    @staticmethod
    def compute_energy_stacked(x: torch.Tensor, h: torch.Tensor, J: torch.Tensor):
        """
        x: (B, L, q)
        J: (N, L, q, L, q)
        h: (N, L, q)
        Returns: (B, N)
        """
        N, L, q = h.shape
        B = x.shape[0]

        # Create mask (same as in __init__)
        mask = torch.ones(L, q, L, q, device=x.device)
        mask[torch.arange(L), :, torch.arange(L), :] = 0

        # Apply mask to all J
        J_masked = J * mask  # (N, L, q, L, q)

        # Flatten
        x_flat = x.view(B, L * q)         # (B, L*q)
        h_flat = h.view(N, L * q)         # (N, L*q)
        J_flat = J_masked.view(N, L * q, L * q)  # (N, L*q, L*q)

        # Bias term: (B, N)
        bias_term = torch.einsum('bi,ni->bn', x_flat, h_flat)

        # Coupling term: (B, N)
        # (B, L*q) @ (N, L*q, L*q) -> (B, N, L*q)
        xJ = torch.einsum('bi,nij->bnj', x_flat, J_flat)
        # (B, N, L*q) * (B, 1, L*q) -> (B, N, L*q)
        coupling_term = torch.sum(xJ * x_flat.unsqueeze(1), dim=2)  # (B, N)

        # Final energy: (B, N)
        return -bias_term - 0.5 * coupling_term

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