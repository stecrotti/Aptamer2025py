import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import energy_models


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
    return torch.abs(num / den).item()

class Callback:
    pass

    def before_training(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs):
        return False

class ConvergenceMetricsCallback(Callback):
    def __init__(self, progress_bar=True):
        super().__init__()
        self.pearson = []
        self.slope = []
        self.grad_norm = []
        self.log_likelihood = []
        self.grad_model = []
        self.grad_data = []
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
        grad_norm = (torch.sqrt(torch.square(grad_vec).sum()) / len(grad_vec)).item()
        self.pearson.append(pearson)
        self.slope.append(slope)
        self.grad_norm.append(grad_norm)
        self.log_likelihood.append(log_likelihood)
        self.grad_model.append(grad_model.detach().clone())
        self.grad_data.append(grad_data.detach().clone())

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
    
def compute_potts_covariance(model, grad_model, grad_data):
    fi = grad_data[1]
    fij = grad_data[2]
    pi = grad_model[1]
    pij = grad_model[2]
    mask = model.selection.modes[0].mask
    pij = pij * mask * 2
    fij = fij * mask * 2
    cov_data = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    cov_chains = pij - torch.einsum('ij,kl->ijkl', pi, pi)
    L = fi.size(0)
    idx_row, idx_col = torch.tril_indices(L, L, offset=-1)
    fij_extract = cov_data[idx_row, :, idx_col, :].reshape(-1)
    pij_extract = cov_chains[idx_row, :, idx_col, :].reshape(-1)
    pearson = torch.corrcoef(torch.stack([fij_extract.float(), pij_extract.float()]))[0, 1].item()
    
    return pearson
    
class PearsonCovarianceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.pearson = []

    def after_step(self, model, grad_model, grad_data,  *args, **kwargs):
        pearson = compute_potts_covariance(model, grad_model, grad_data)
        self.pearson.append(pearson)

        return False
    
    def plot(self, figsize=(6, 3)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.pearson)
        ax.set_xlabel('iter')
        ax.set_ylabel('Pearson $C_{ij}$')

        return fig, ax
    
    
class TeacherStudentCallback(Callback):
    def __init__(self, model_teacher):
        super().__init__()
        self.model_teacher = model_teacher
        self.pearson_Ns0 = []
        self.pearson_ps = []

    def after_step(self, model, *args, **kwargs):
        model_teacher = self.model_teacher
        model_student = model
        Ns0_teacher = model_teacher.round_zero
        Ns0_student = model_student.round_zero
        if isinstance(Ns0_teacher, energy_models.IndepSites) or isinstance(Ns0_teacher, energy_models.Potts):
            Ns0_teacher = Ns0_teacher.set_zerosum_gauge()
            Ns0_student = Ns0_student.set_zerosum_gauge()
        pearson_Ns0_round = []
        for (param_teacher, param_student) in zip(Ns0_teacher.parameters(), Ns0_student.parameters()):
            x = param_teacher.detach().clone().cpu().reshape(-1)
            y = param_student.detach().clone().cpu().reshape(-1)
            p = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
            pearson_Ns0_round.append(p)
        self.pearson_Ns0.append(pearson_Ns0_round)
        
        pearson_ps_round = []
        for (mode_teacher, mode_student) in zip(model_teacher.selection.modes, model_student.selection.modes):
            if isinstance(mode_teacher, energy_models.IndepSites) or isinstance(mode_teacher, energy_models.Potts):
                mode_teacher = mode_teacher.set_zerosum_gauge()
                mode_student = mode_student.set_zerosum_gauge()
            pearson_ps_mode = []
            for (param_teacher, param_student) in zip(mode_teacher.parameters(), mode_student.parameters()):
                x = param_teacher.detach().clone().cpu().reshape(-1)
                y = param_student.detach().clone().cpu().reshape(-1)
                p = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
                pearson_ps_mode.append(p)
            pearson_ps_round.append(pearson_ps_mode)
        self.pearson_ps.append(pearson_ps_round)

    def plot(self, figsize=(10, 4)):
        n_selection_modes = self.model_teacher.selection.get_n_modes()
        fig, axes = plt.subplots(1, n_selection_modes + 1, figsize=figsize)

        ax = axes[0]
        ax.set_title('Ns0')
        pearson_Ns0 = zip(*self.pearson_Ns0)
        for (pearson, np) in zip(pearson_Ns0, self.model_teacher.round_zero.named_parameters()):
            ax.plot(pearson, label=np[0])
            ax.set_xlabel('iter'); ax.set_ylabel('Pearson')
            ax.legend()
        
        
        pearson_ps = zip(*self.pearson_ps)
        for (i, pearson_mode) in enumerate(pearson_ps):
            ax = axes[i+1]
            ax.set_title(f'ps - mode #{i}')
            for (pearson, np) in zip(zip(*pearson_mode), self.model_teacher.selection.modes[i].named_parameters()):
                ax.plot(pearson, label=np[0])
                ax.set_xlabel('iter'); ax.set_ylabel('Pearson')
                ax.legend()
        fig.suptitle('Correlation between teacher and student parameters')
        fig.tight_layout()
        