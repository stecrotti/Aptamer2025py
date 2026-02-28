import torch
import energy_models, tree
import selex_distribution
import utils

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
        for t in range(n_rounds-1):
            tr.add_node(t-1)

        selected_modes = torch.ones((n_rounds-1, 1), dtype=bool)

        Ns0 = energy_models.IndepSites(round_zero_field)
        ps = selex_distribution.MultiModeDistribution(energy_models.IndepSites(selection_field), 
                                                      normalized=False)
        super().__init__(Ns0, ps, tr, selected_modes)
        self.fi = None

    def selection_field(self):
        return self.selection.modes[0].h 
    
    def round_zero_field(self):
        return self.round_zero.h
    
    def cache_site_frequencies(self, sequences_oh):
        fi_list = [utils.get_freq_single_point(s) for s in sequences_oh]
        self.fi = torch.stack(fi_list)
    
    @torch.no_grad
    def grad_loglikelihood_model(self, chains, total_reads, **kwargs):
        n_rounds = len(total_reads)
        normaliz = total_reads.sum()
        
        grad_selection = torch.zeros_like(self.selection_field())
        grad_round_zero = torch.zeros_like(self.round_zero_field())

        for t in range(n_rounds):
            logpit = utils.normalize_to_logprob(t * self.selection_field() + self.round_zero_field())
            pit = torch.exp(logpit)
            grad_selection += t * total_reads[t] * pit / normaliz
            grad_round_zero += total_reads[t] * pit / normaliz


        return (grad_round_zero, grad_selection)
    
    @torch.no_grad
    def grad_loglikelihood_data(self, batches, total_reads, **kwargs):
        if self.fi is None:
            fi_list = [utils.get_freq_single_point(s) for s in batches]
            fi = torch.stack(fi_list)
        else:
            fi = self.fi
        n_rounds = len(total_reads)
        normaliz = total_reads.sum()
        
        grad_selection = torch.zeros_like(self.selection_field())
        grad_round_zero = torch.zeros_like(self.round_zero_field())

        for t in range(n_rounds):
            grad_selection += t * total_reads[t] * fi[t] / normaliz
            grad_round_zero += total_reads[t] * fi[t] / normaliz


        return (grad_round_zero, grad_selection)
        
    @torch.no_grad
    def estimate_log_likelihood(self, batches, total_reads, 
                                log_multinomial_factors=None, use_cached_fi = True, **kwargs):
        if use_cached_fi:
            if self.fi is None:
                raise ValueError('Expected to find cached single-site frequencies.')
            fi = self.fi
        else:
            fi_list = [utils.get_freq_single_point(s) for s in batches]
            fi = torch.stack(fi_list)
        
        if log_multinomial_factors is None:
            log_multinomial_factors = torch.zeros(len(fi), dtype=fi.dtype, device=fi.device)

        n_rounds = len(total_reads)
        normaliz = total_reads.sum()

        ll = 0.0
        for t in range(n_rounds):
            logpit = utils.normalize_to_logprob(t * self.selection_field() + self.round_zero_field())
            ll += (log_multinomial_factors[t] + total_reads[t] * fi[t] * logpit).sum()
    
        return (ll / normaliz).item()

