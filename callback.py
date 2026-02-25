import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm.autonotebook import tqdm
import energy_models
import IPython
from utils import compute_pearson, compute_slope
import utils
import numpy
import pathlib
import numpy
import copy
import shutil

def relative_error(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    return torch.linalg.norm(x - y) / torch.linalg.norm(x)

class Callback:
    pass

    def before_training(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs):
        return False

# TODO: save also some info about gradient persistance (e.g. cosine similarity t,t+1 over a window) and how parameters change
class ConvergenceMetricsCallback(Callback):
    def __init__(self, progress_bar=True, progress_plot=False):
        super().__init__()
        self.pearson = []
        self.slope = []
        self.pearson_detail = []
        self.slope_detail = []
        self.grad_norm = []
        self.grad_max = []
        self.grad_median = []
        self.log_likelihood = []
        self.log_likelihood_valid = []
        self.grad_max_params = []
        self.grad_param_ratio = []
        self.param_names = None
        self.progress_bar = progress_bar
        self.progress_plot = progress_plot

    def before_training(self, model, max_epochs, *args, **kwargs):
        if self.progress_bar:
            pbar = tqdm(
                initial=0,
                total=max_epochs,
                colour="red",
                dynamic_ncols=True,
                leave=False,
                ascii="-#",
                bar_format="{desc} {percentage:.2f}%[{bar}] Epoch: {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
            self.pbar = pbar
        self.param_names = [name for (name, param) in model.named_parameters()]
        if self.progress_plot:
            fig, axes = plt.subplots(1, 4)
            self.fig = fig
            self.axes = axes
            
            ax = axes[0]
            ax.set_yscale('log')
            ax.set_xlabel('iter')
            ax.set_ylabel('|1-pearson|')

            ax = axes[1]
            ax.set_yscale('log')
            ax.set_xlabel('iter')
            ax.set_ylabel('|1-slope|')

            ax = axes[2]
            ax.set_yscale('log')
            ax.set_xlabel('iter')
            ax.set_ylabel('|| grad logL ||')

            ax = axes[3]
            ax.set_xlabel('iter')
            ax.set_ylabel('NLL')

    def after_step(self, model, grad_model, grad_data, grad_total, log_likelihood, log_likelihood_valid,
                   epochs, target_pearson, thresh_slope, *args, **kwargs):
        
        x = torch.nn.utils.parameters_to_vector(grad_model)
        y = torch.nn.utils.parameters_to_vector(grad_data)
        pearson = compute_pearson(x, y)
        slope = compute_slope(x, y)
        grad_vec = torch.nn.utils.parameters_to_vector(grad_total).cpu()
        grad_norm = (torch.sqrt(torch.square(grad_vec).sum()) / len(grad_vec)).item()
        abs_grad_vec = torch.abs(grad_vec)
        grad_max = torch.max(abs_grad_vec).item()
        self.pearson.append(pearson)
        self.slope.append(slope)
        self.grad_norm.append(grad_norm)
        self.log_likelihood.append(log_likelihood)
        self.grad_max.append(grad_max)
        self.grad_median.append(torch.median(abs_grad_vec).item())
        if log_likelihood_valid:
            self.log_likelihood_valid.append(log_likelihood_valid)

        pearson_detail = []
        slope_detail = []
        for (gm, gd) in zip(grad_model, grad_data):
            x = gm.detach().clone().cpu()
            y = gd.detach().clone().cpu()
            assert torch.numel(x) == torch.numel(y)
            if torch.numel(x) > 1:
                p = compute_pearson(x, y)
                s = compute_slope(x, y)
                pearson_detail.append(p)
                slope_detail.append(s)
        self.pearson_detail.append(pearson_detail)
        self.slope_detail.append(slope_detail)

        g = []
        r = []
        for ((name, param), grad) in zip(model.named_parameters(), grad_total):
            param = param.detach().clone().cpu()
            grad = grad.detach().clone().cpu()
            if name.endswith('J'):
                param = utils.off_diagonal_terms(param)
                grad = utils.off_diagonal_terms(grad)
            grad_abs = torch.abs(grad)
            param_abs = torch.abs(param)
            ratio = grad_abs / (param_abs + 1e-8)
            ratio_abs = torch.max(ratio).item()
            r.append(ratio_abs)
            max = torch.max(grad_abs).item()
            g.append(max)
        self.grad_param_ratio.append(r)
        self.grad_max_params.append(g)

        if self.progress_bar:
            self.pbar.n = epochs
            desc = f"Pearson = {pearson:.5f}, Max abs grad = {grad_max:.4e}, NLL = {-log_likelihood:.4f}"
            if log_likelihood_valid:
                desc += f", NLL valid = {-log_likelihood_valid:.4f}"
            self.pbar.set_description(desc)
        
        if self.progress_plot:
            self.axes[0].plot([abs(1-p) for p in self.pearson])
            self.axes[1].plot([abs(1-p) for p in self.slope])
            self.axes[2].plot(self.grad_norm)
            self.axes[3].plot([-nll for nll in self.log_likelihood])
            self.fig.tight_layout()
            IPython.display.display(self.fig, clear=True)

        c1 = pearson > target_pearson
        c2 = abs(slope - 1.) < thresh_slope

        return c1 and c2
    
    def plot(self, figsize=(12,3)):
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
        ax.plot(self.grad_norm, label='mean ' + '$|\\nabla logL|$')
        ax.plot(self.grad_max, label='max ' + '$|\\nabla logL|$')
        ax.plot(self.grad_median, label='median ' + '$|\\nabla logL|$')
        ax.set_yscale('log')
        ax.set_xlabel('iter')
        # ax.set_title('grad logL')
        ax.legend()

        ax = axes[3]
        ax.plot([-ll for ll in self.log_likelihood], label='training')
        if self.log_likelihood_valid:
            ax.plot([-ll for ll in self.log_likelihood_valid], label='validation')
        ax.set_xlabel('iter')
        ax.set_ylabel('NLL')
        ax.legend()

        fig.tight_layout()
        
        return fig, axes
    
    def plot_pearson_detail(self, figsize=(10, 3)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for (name, pearson, slope) in zip(self.param_names, zip(*self.pearson_detail), zip(*self.slope_detail)):
            ax = axes[0]
            ax.plot([abs(1-p) for p in pearson], label=name)
            ax.set_xlabel('iter'); ax.set_ylabel('|1 - pearson|')
            ax.set_yscale('log')
            ax.legend()

            ax = axes[1]
            ax.plot([abs(1-s) for s in slope], label=name)
            ax.set_xlabel('iter'); ax.set_ylabel('|1 - slope|')
            ax.set_yscale('log')
            ax.legend()

        fig.tight_layout()
        
        return fig, ax
    
    def plot_gradient_parameter_ratio(self, figsize=(10,3)):
        fig, ax = plt.subplots(figsize=figsize)
        names = self.param_names

        for (name, norm) in zip(names, zip(*self.grad_param_ratio)):
            ax.plot(norm, label=name)
        ax.legend()
        ax.set_ylabel('$\\max |\\nabla p / p|$')
        ax.set_xlabel('iter')
        ax.set_title('Max abs gradient divided by parameter value')
        fig.tight_layout()
        ax.set_yscale('log')
        return fig, ax
    
    def plot_grad_detail(self, figsize=(10,3)):
        fig, ax = plt.subplots(figsize=figsize)
        names = self.param_names

        for (name, mx) in zip(names, zip(*self.grad_max_params)):
            ax.plot(mx, label=name)
        ax.legend()
        ax.set_ylabel('$\\max |\\nabla p|$')
        ax.set_xlabel('iter')
        ax.set_title('Max abs gradient')
        fig.tight_layout()
        ax.set_yscale('log')
        return fig, ax
    
def compute_potts_covariance(fi, fij, pi, pij):
    L, q = fi.size()
    assert fij.size() == (L, q, L, q), f"{fij.size()}"
    cov_data = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    cov_chains = pij - torch.einsum('ij,kl->ijkl', pi, pi)

    pearson = utils.compute_pearson(utils.off_diagonal_terms(cov_data), utils.off_diagonal_terms(cov_chains))
    
    return pearson
    
class PearsonCovarianceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.pearson_Ns0 = []
        self.pearson_ps = []

    def after_step(self, model, grad_model, grad_data, *args, **kwargs):
        offset = 0
        if isinstance(model.round_zero, energy_models.Potts):
            pearson_Ns0 = compute_potts_covariance(
                grad_data[1+offset], grad_data[0+offset]*2, grad_model[1+offset], grad_model[0+offset]*2)
            self.pearson_Ns0.append(pearson_Ns0)
        offset += len(list(model.round_zero.parameters()))
        
        pearson_ps_round = []
        for (mode) in model.selection.modes:
            pearson_ps_mode = []
            if isinstance(mode, energy_models.Potts):
                pearson_ps_mode = compute_potts_covariance(
                    grad_data[1+offset], grad_data[0+offset]*2, grad_model[1+offset], grad_model[0+offset]*2)
                pearson_ps_round.append(pearson_ps_mode)
            offset += len(list(mode.parameters()))
        self.pearson_ps.append(pearson_ps_round)

        return False
    
    def plot(self, figsize=(10, 4)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.set_title('Ns0')
        x = [abs(1-p) for p in self.pearson_Ns0]
        ax.plot(x)
        # ax.plot(self.pearson_Ns0)
        # ax.axhline(y=1, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('iter'); 
        # ax.set_ylabel('Pearson $C_{ij}$')
        ax.set_title('Pearson $C_{ij}$ Ns0')
        ax.set_ylabel('$|1-\\rho|$')
        ax.set_yscale('log')
        
        pearson_ps = zip(*self.pearson_ps)
        ax = axes[1]
        ax.set_xlabel('iter'); 
        # ax.set_ylabel('Pearson $C_{ij}$')
        ax.set_ylabel('$|1-\\rho|$')
        ax.set_title('Pearson $C_{ij}$ ps')
        ax.set_yscale('log')
        for (i, pearson_mode) in enumerate(pearson_ps):
            x = [abs(1-p) for p in pearson_mode]
            ax.plot(x, label=f'Potts mode #{i}')
            # ax.plot(pearson_mode, label=f'Potts mode #{i}')
            # ax.axhline(y=1, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        fig.suptitle('Pearson on covariances for Potts modes')
        fig.tight_layout()
        
        return fig, axes
    
    
class TeacherStudentCallback(Callback):
    def __init__(self, model_teacher):
        super().__init__()
        self.model_teacher = model_teacher
        self.pearson_Ns0 = []
        self.pearson_ps = []
        self.pearson_energies = []
        self.slope_energies = []

    def after_step(self, model, data_loaders, grad_total, *args, **kwargs):
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
            assert len(x) == len(y)
            if len(x) > 0:
                p = compute_pearson(x, y)
                pearson_Ns0_round.append(p)
        self.pearson_Ns0.append(pearson_Ns0_round)
        
        pearson_ps_round = []
        for (mode_teacher, mode_student) in zip(model_teacher.selection.modes, model_student.selection.modes):
            if isinstance(mode_teacher, energy_models.IndepSites) or isinstance(mode_teacher, energy_models.Potts):
                mode_teacher = mode_teacher.set_zerosum_gauge()
                mode_student = mode_student.set_zerosum_gauge()
            pearson_ps_mode = []
            for (np_teacher, np_student) in zip(mode_teacher.named_parameters(), mode_student.named_parameters()):
                name_teacher, param_teacher = np_teacher
                name_student, param_student = np_student
                assert name_teacher == name_student
                x = param_teacher.detach().clone().cpu()
                y = param_student.detach().clone().cpu()
                # if it's a Potts model, only consider the lower diagonal when computing pearson
                if isinstance(mode_teacher, energy_models.Potts) and name_teacher.endswith('J'):
                    x = utils.off_diagonal_terms(x)
                    y = utils.off_diagonal_terms(y)
                assert len(x) == len(y)
                if torch.numel(x) > 1:
                    p = compute_pearson(x, y)
                    pearson_ps_mode.append(p)
            pearson_ps_round.append(pearson_ps_mode)
        self.pearson_ps.append(pearson_ps_round)

        pearson_energy = []
        slope_energy = []
        for t in range(model.get_n_rounds()):
            x = data_loaders[t].get_batch()
            en_teacher = self.model_teacher.compute_energy_up_to_round(x, t).detach().cpu()
            en_student = model.compute_energy_up_to_round(x, t).detach().cpu()
            p = compute_pearson(en_teacher, en_student)
            s = compute_slope(en_teacher, en_student)
            pearson_energy.append(p)
            slope_energy.append(s)
        self.pearson_energies.append(pearson_energy)
        self.slope_energies.append(slope_energy)


    def plot_pearson_parameters(self, figsize=(10, 4)):
        n_selection_modes = self.model_teacher.selection.get_n_modes()
        fig, axes = plt.subplots(1, n_selection_modes + 1, figsize=figsize)
        if type(axes) != numpy.ndarray: axes = [axes]

        ax = axes[0]
        ax.set_title('Ns0')
        pearson_Ns0 = zip(*self.pearson_Ns0)
        for (pearson, np) in zip(pearson_Ns0, self.model_teacher.round_zero.named_parameters()):
            ax.plot(pearson, label=np[0])
            ax.axhline(y=1, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('iter'); ax.set_ylabel('Pearson')
            ax.legend()
        
        
        pearson_ps = zip(*self.pearson_ps)
        for (i, pearson_mode) in enumerate(pearson_ps):
            ax = axes[i+1]
            ax.set_title(f'ps - mode #{i}')
            for (pearson, np) in zip(zip(*pearson_mode), self.model_teacher.selection.modes[i].named_parameters()):
                ax.plot(pearson, label=np[0])
                ax.axhline(y=1, color='r', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_xlabel('iter'); ax.set_ylabel('Pearson')
                ax.legend()
        fig.suptitle('Correlation between teacher and student parameters')
        fig.tight_layout()
        
        return fig, axes

    def plot_pearson_energies(self, figsize=(10, 4)):
        n_rounds = self.model_teacher.get_n_rounds()
        fig, axes = plt.subplots(1, n_rounds, figsize=figsize)
        if type(axes) != numpy.ndarray: axes = [axes]

        pearson_energy = list(zip(*self.pearson_energies))
        slope_energy = list(zip(*self.slope_energies))
        for t in range(n_rounds):
            ax = axes[t]
            ax.axhline(y=1, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.plot(pearson_energy[t], label='Pearson')
            ax.plot(slope_energy[t], label='Slope')
            ax.set_title(f'Round t={t}')
            ax.set_xlabel('iter')
            ax.legend()
        fig.suptitle('Correlation between teacher and student logNst on a random batch')
        fig.tight_layout()

        return fig, axes
        
    def plot(self, **kwargs):
        return self.plot_pearson_parameters(**kwargs)
    
class ConstEnergyCallback(Callback):
    def __init__(self, model_teacher):
        super().__init__()
        self.model_teacher = model_teacher
        self.err = []

    def after_step(self, model, *args, **kwargs):
        en_teacher = self.model_teacher.selection.modes[-1].en
        en_student = model.selection.modes[-1].en
        self.err.append(relative_error(en_teacher, en_student).item())

    def plot(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        ax.plot(self.err)
        ax.set_xlabel('iter')
        ax.set_ylabel('relative err')
        ax.set_title('Relative error on constant energy for unbounded mode')
        fig.tight_layout()

        return fig, ax
    
def save_checkpoint(checkpoint_filename, epochs, **kwargs):
    fn = checkpoint_filename + '_' + str(epochs) + '.pt'
    dirpath = pathlib.Path(__file__).parent.resolve() / f'experiments/checkpoints/{checkpoint_filename}' 
    torch.save(kwargs, dirpath / fn)
    
class CheckpointCallback(Callback):
    def __init__(self, save_every = torch.inf, checkpoint_filename = 'model', delete_old_checkpoints = True):
        super().__init__()
        self.save_every = save_every
        self.checkpoint_filename = checkpoint_filename
        self.total_epochs = 0

        if delete_old_checkpoints:
            dirpath = pathlib.Path(__file__).parent.resolve() / f'experiments/checkpoints/{checkpoint_filename}' 
            if dirpath.is_dir():
                shutil.rmtree(dirpath, ignore_errors=True)

        dirpath = pathlib.Path(__file__).parent.resolve() / f'experiments/checkpoints/{checkpoint_filename}' 
        pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)

    def before_training(self, *args, **kwargs):
        super().before_training(*args, **kwargs)

    def after_step(self, model, optimizer, log_weights, epochs, *args, **kwargs):
        self.total_epochs += 1
        if epochs % self.save_every == 0:
            cpu = torch.device('cpu')

            save_checkpoint(self.checkpoint_filename, self.total_epochs,
                            model=copy.deepcopy(model).to(cpu), optimizer=optimizer, log_weights=copy.deepcopy(log_weights).to(cpu))

class ParamsCallback(Callback):
    def __init__(self, save_every=torch.inf):
        super().__init__()
        self.save_every = save_every
        self.param_names = None
        self.params = []
        self.epochs = []
        self.last_epoch = 1 - self.save_every

    def after_step(self, model, epochs, *args, **kwargs):
        if self.param_names is None:
            self.param_names = [n for (n, p) in model.named_parameters()]
        if (epochs-1) % self.save_every == 0:
            self.params.append([p.detach().cpu().clone() for p in model.parameters()])
            curr_epoch = self.last_epoch + self.save_every
            self.epochs.append(curr_epoch)
            self.last_epoch = curr_epoch
        return False
    
    def plot(self, figsize=(10, 4), plot_every:int = 1, cmap=matplotlib.cm.viridis):
        params = list(zip(*self.params))
        n_params = len(self.param_names)
        n_points = len(self.params)
        assert len(params) == n_params
        fig, axes = plt.subplots(1, n_params, figsize=figsize)
        cgrad = [cmap(x) for x in numpy.linspace(0, 1, n_points)]

        for i in range(n_params):
            ax = axes[i]
            param_name = self.param_names[i]
            if param_name.endswith('.J'):
                transform = utils.off_diagonal_terms
            else:
                transform = lambda x: x
            param = params[i]
            p_final = transform(param[-1])
            ax.plot(p_final.reshape(-1), p_final.reshape(-1), color='gray', label='identity', ls='--')
            for n in range(0, n_points, plot_every):
                p_current = transform(param[n])
                ax.scatter(p_current, p_final,
                    label=f'Epoch {self.epochs[n]}', color=cgrad[n])
                ax.set_xlabel('Param at epoch n')
                ax.set_ylabel(f'Param at epoch {self.epochs[-1]}')
                ax.set_title(param_name)
                ax.legend()
        fig.tight_layout()
        return fig, axes
    
    def plot_diff(self, **kwargs):
        n_steps = len(self.params)
        n_params = len(self.param_names)
        diff = list(zip(*[[torch.median(torch.abs(self.params[n][p] - self.params[n-1][p])).item() for p in range(n_params)] for n in range(n_steps)]))

        fig, ax = plt.subplots( **kwargs)
        for i in range(n_params):
            ax.plot(self.epochs, diff[i], label=self.param_names[i])
            ax.set_xlabel('epoch')
            ax.set_yscale('log')
            ax.legend()

        ax.set_title('Abs median difference of parameter value wrt previous measure')
        fig.tight_layout()

        return fig, ax
