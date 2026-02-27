import torch
import energy_models, tree
import selex_distribution

def is_simple_indep_sites(model: selex_distribution.MultiRoundDistribution):
    if not isinstance(model.round_zero, energy_models.IndepSites):
        return False
    modes = model.selection.modes
    if len(modes) > 1:
        return False
    for mode in modes:
        if not isinstance(mode, energy_models.IndepSites):
            return False
            
    return True

class IndepSitesMultiRoundDistribution(selex_distribution.MultiRoundDistribution):
    def __init__(
        self,
        round_zero_field: torch.tensor,
        selection_field: torch.tensor,
        n_rounds: int
    ):
        tr = tree.Tree()
        for t in range(n_rounds):
            tr.add_node(t-1)

        selected_modes = torch.ones((n_rounds-1, 1), dtype=bool)

        Ns0 = energy_models.IndepSites(round_zero_field)
        ps = selex_distribution.MultiModeDistribution(energy_models.IndepSites(selection_field), 
                                                      normalized=False)
        super.__init__(Ns0, ps, tr, selected_modes)
