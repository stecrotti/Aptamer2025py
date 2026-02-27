import torch
import utils
import training
import sampling
import matplotlib.pyplot as plt

def check_equilibration(model, data_loaders, total_reads, log_multinomial_factors=None,
                        n_max_sweeps = 10**3, n_chains_equil = 10**3, 
                        fi = None,
                        dtype=torch.float32, device=torch.device('cpu')):
    batches = [next(iter(dl)) for dl in data_loaders]
    if fi is None:
        fi = torch.stack([utils.get_freq_single_point(b) for b in batches])
    n_rounds = len(batches)
    L, q = batches[0][0].size()
    chains_fi = training.init_chains(n_rounds, n_chains_equil, L, q, dtype=dtype, device=device, fi=fi)
    chains_indep = training.init_chains(n_rounds, n_chains_equil, L, q, dtype=dtype, device=device)

    print('Computing mixing time for chains initialized at site frequencies')
    mixing_times_fi, fig_fi = sampling.compute_and_plot_mixing_time(
        model, chains_fi, n_max_sweeps, title="Mixing time - chains initialized at site frequences")
    print('\nComputing mixing time for chains initialized uniformly')
    mixing_times_indep, fig_indep = sampling.compute_and_plot_mixing_time(
        model, chains_indep, n_max_sweeps, title="Mixing time - chains initialized uniformly")
    
    model.sample(chains_fi, n_sweeps=max(mixing_times_fi)*2)
    model.sample(chains_indep, n_sweeps=max(mixing_times_indep)*2)

    grad_data = training.grad_loglikelihood_component(model, batches, total_reads, log_multinomial_factors)
    grad_model_fi =  training.grad_loglikelihood_component(model, chains_fi, total_reads, log_multinomial_factors)
    grad_model_indep =  training.grad_loglikelihood_component(model, chains_indep, total_reads, log_multinomial_factors)

    n_param = len(list(model.named_parameters()))

    fig, axes = plt.subplots(2, n_param, figsize=(12,6))
    for (i, (param_name, _)) in enumerate(model.named_parameters()):
        ax = axes[0, i]
        ax.scatter(grad_model_fi[i].cpu(), grad_data[i].cpu(), label='Initializ fi')
        ax.scatter(grad_model_indep[i].cpu(), grad_data[i].cpu(), label='Initializ indep')
        ax.plot(grad_data[i].cpu().reshape(-1), grad_data[i].cpu().reshape(-1), label='identity', color='gray', ls='--')
        ax.legend()
        ax.set_xlabel('Moments model')
        ax.set_ylabel('Moments data')
        ax.set_title(param_name)

        ax = axes[1, i]
        ax.scatter(grad_model_fi[i].cpu(), grad_model_indep[i].cpu())
        ax.plot(grad_model_fi[i].cpu().reshape(-1), grad_model_fi[i].cpu().reshape(-1), label='identity', color='gray', ls='--')
        ax.legend()
        ax.set_xlabel('Moments model initializ fi')
        ax.set_ylabel('Moments model initializ indep')
        ax.set_title(param_name)
    fig.suptitle("Equilibration check")
    fig.tight_layout()

    return fig_fi, fig_indep, fig