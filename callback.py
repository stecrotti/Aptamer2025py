import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import selex_dca

def compute_pearson(grad_model, grad_data):
    x = torch.nn.utils.parameters_to_vector(grad_model)
    y = torch.nn.utils.parameters_to_vector(grad_data)
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()

def compute_slope(grad_model, grad_data):
    x = torch.nn.utils.parameters_to_vector(grad_model)
    y = torch.nn.utils.parameters_to_vector(grad_data)
    n = len(x)
    num = n * (x @ y) - y.sum() * x.sum()
    den = n * (x @ x) - torch.square(x.sum())
    return torch.abs(num / den)

class ConvergenceMetricsCallback:
    def __init__(self, progress_bar=True):
        self.pearson = []
        self.slope = []
        self.grad_norm = []
        self.log_likelihood = []
        self.grad_data = []
        self.grad_model = []
        self.progress_bar = progress_bar

    def before_training(self, max_epochs, *args, **kwargs):
        if self.progress_bar:
            pbar = tqdm(
                initial=0,
                total=max_epochs,
                colour="red",
                dynamic_ncols=True,
                leave=False,
                ascii="-#",
                bar_format="{desc} {percentage:.2f}%[{bar}] Epoch: {n}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
            )
            self.pbar = pbar

    def after_step(self, grad_model, grad_data, grad_total, log_likelihood, epochs, target_pearson, thresh_slope, *args, **kwargs):
        pearson = compute_pearson(grad_model, grad_data)
        slope = compute_slope(grad_model, grad_data)
        grad_vec = torch.nn.utils.parameters_to_vector(grad_total)
        grad_norm = torch.sqrt(torch.square(grad_vec).sum()) / len(grad_vec)
        self.pearson.append(pearson)
        self.slope.append(slope)
        self.grad_norm.append(grad_norm)
        self.log_likelihood.append(log_likelihood)
        self.grad_model.append(grad_model)
        self.grad_data.append(grad_data)

        if self.progress_bar:
            self.pbar.n = epochs
            self.pbar.set_description(f"Epoch {epochs}, Pearson = {pearson:.4e}, Gradient norm = {grad_norm:.4e}, NLL = {-log_likelihood:.4e}")
        
        c1 = pearson > target_pearson
        c2 = abs(slope - 1.) < thresh_slope

        return c1 and c2
    
    def plot(self, figsize=(10,3)):
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        ax = axes[0]
        ax.plot([abs(1-p) for p in self.pearson])
        ax.set_yscale('log')
        ax.set_xlabel('iter')
        ax.set_ylabel('|1-pearson|')

        ax = axes[1]
        ax.plot([abs(1-p) for p in self.slope])
        ax.set_yscale('log')
        ax.set_xlabel('iter')
        ax.set_ylabel('|1-slope|')

        ax = axes[2]
        ax.plot(self.grad_norm)
        ax.set_yscale('log')
        ax.set_xlabel('iter')
        ax.set_ylabel('|| grad logL ||')

        ax = axes[3]
        ax.plot([-nll for nll in self.log_likelihood])
        ax.set_xlabel('iter')
        ax.set_ylabel('NLL')

        fig.tight_layout()
        
        return fig, axes
    
class PearsonCovarianceCallback:
    def __init__(self):
        self.pearson = []
        self.grad_model = []
        self.grad_data = []

    def before_training(self, *args, **kwargs):
        pass

    def after_step(self, model, grad_model, grad_data, total_reads, *args, **kwargs):
        fi = grad_data[1]
        fij = grad_data[2]
        pi = grad_model[1]
        pij = grad_model[2]
        mask = model.selection.modes[0].mask
        pij = pij * mask * 2
        fij = fij * mask * 2
        pearson, slope = selex_dca.get_correlation_two_points(fij, pij, fi, pi, total_reads)
        self.pearson.append(pearson)
        self.grad_model.append(grad_model)
        self.grad_data.append(grad_data)

        return False
    
    def plot(self, figsize=(6, 3)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.pearson)
        ax.set_xlabel('iter')
        ax.set_ylabel('Pearson $C_{ij}$')

        return fig, ax