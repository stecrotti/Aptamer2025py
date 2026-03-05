import sys
sys.path.append('./src')

import torch

import selex_distribution, specialized_models, utils, tree, energy_models

L, q = 10, 3
n_rounds = 4
n_modes = 6

tr = tree.Tree()
for t in range(n_rounds-1):
    # tr.add_node(torch.randint(t-1, (1,)).item())
    tr.add_node(t-1)

n_selection_rounds = n_rounds - 1
selected_modes = torch.randint(1, (n_selection_rounds, n_modes)).to(torch.bool)
k = torch.randn(L, q)
h = torch.randn(L, q)
Ns0 = energy_models.IndepSites(k)
mode = energy_models.IndepSites(h)
modes = [mode for _ in range(n_modes)]
ps = selex_distribution.MultiModeDistribution(*modes)
model = specialized_models.IndepSitesMultiRoundDistribution(k, h, n_rounds)
chemical_potential = torch.rand(n_selection_rounds, n_modes)
selection_strength = torch.rand(n_selection_rounds)
model = selex_distribution.MultiRoundDistribution(
    Ns0, ps, tr, selected_modes=selected_modes,
    chemical_potential=chemical_potential,
    selection_strength=selection_strength
    )

x = utils.random_data(5, L, q)

def dot(h, x):
    return (x * h).sum((-1,-2))

t = 0
print(model.tree.ancestors_of(t-1))
print(model.selection_energy_up_to_round(x, t))

# for t in range(n_rounds):
#     en_model = model.compute_energy_up_to_round(x, t)
#     en_true = dot(x, k)
#     for tau in range(t):
#         en_round = 0
#         for m in range(n_modes):
#             en_round += (dot(x, h) - chemical_potential[tau, m])
            
#         en_true += en_round * selection_strength[tau]

#     print(en_model.mean(), en_true.mean())