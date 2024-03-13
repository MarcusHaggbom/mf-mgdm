import torch
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
import energies
from models import Model


class GeneratorABC(ABC):
    def __init__(self, energy_fn: energies.Energy, target_energy: torch.Tensor, grad_steps: int, grad_step_size: float,
                 gpu_bs_gen: int | None = None, gpu_bs_logdet: int | None = None, project: bool = False):
        self.energy_fn = energy_fn
        self.target_energy = target_energy
        self.loss_normalization = (target_energy ** 2).sum()

        self.grad_steps = grad_steps
        self.grad_step_size = grad_step_size
        self.project = project

        self.gpu_bs_logdet = gpu_bs_logdet
        self.gpu_bs_gen = gpu_bs_gen

    @abstractmethod
    def __repr__(self):
        pass

    def __call__(self, x0: torch.Tensor, include_log_det: bool = True, steps: int | None = None,
                 include_errors: bool = False):
        return self.descend(x0, include_log_det, steps, include_errors)

    @abstractmethod
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def descent_step(self, x: torch.Tensor, include_log_det: bool = True) -> (torch.Tensor, torch.Tensor):
        pass

    def descend(self, x0: torch.Tensor, include_log_det: bool, steps: int | None = None, include_errors: bool = False):
        steps = steps or self.grad_steps
        x = x0
        errors = []
        if include_errors:
            errors.append(self.loss(x).mean().item())
        log_det_sum = torch.zeros(1, device=x.device)
        if steps > 1:
            print('generating samples by gradient descent:')
        for t in (range if steps == 1 else trange)(steps):
            x, log_det_sum_t = self.descent_step(x, include_log_det)
            log_det_sum += log_det_sum_t
            if torch.isnan(x).any():
                raise RuntimeError(f'NaN encountered in descend at iteration {t}')
            if include_errors:
                errors.append(self.loss(x).mean().item())
        rets = [x]
        if include_log_det:
            rets.append(log_det_sum)
        if include_errors:
            rets.append(errors)
        return tuple(rets) if len(rets) > 1 else rets[0]

    def project_op(self, x: torch.Tensor, x_next: torch.Tensor):
        if self.project:
            x = torch.where(x_next <= 0., x, x_next)  # Step only if next iterate is positive
        else:
            x = x_next
        return x

    def to(self, device):
        self.energy_fn.to(device)
        self.target_energy = self.target_energy.to(device)
        self.loss_normalization = self.loss_normalization.to(device)
        return self

    def cpu(self):
        self.energy_fn.cpu()
        self.target_energy = self.target_energy.cpu()
        self.loss_normalization = self.loss_normalization.cpu()
        return self


class Generator(GeneratorABC):
    def __init__(self, energy_fn: energies.Energy, target_energy: torch.Tensor, grad_steps: int, grad_step_size: float,
                 gpu_bs_gen: int | None = None, gpu_bs_logdet: int | None = None, project: bool = False):
        super().__init__(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_gen, gpu_bs_logdet, project)

    def __repr__(self):
        return f'MGDM-{repr(self.energy_fn)}-{self.grad_steps}-{self.grad_step_size}'

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((self.energy_fn(x) - self.target_energy) ** 2).sum() / self.loss_normalization

    loss = torch.func.vmap(_loss, in_dims=(None, 0))
    grad_loss = torch.func.vmap(torch.func.grad(_loss, argnums=1), in_dims=(None, 0))
    hess_loss = torch.func.vmap(torch.func.hessian(_loss, argnums=1), in_dims=(None, 0))

    def _log_det_contr(self, x, x_next):
        mask = (x_next > 0.)[..., None] if self.project else 1.
        gd_jacs = torch.eye(x.shape[-1], device=x.device) - self.grad_step_size * mask * self.hess_loss(x)
        gd_jac_prod = gd_jacs[0]
        for i in range(1, gd_jacs.shape[0]):
            gd_jac_prod @= gd_jacs[i]  # TODO check if necessary
        log_det_contr = torch.linalg.slogdet(gd_jac_prod)[1]
        if torch.isnan(log_det_contr):
            raise RuntimeError('NaN encountered in log_det_contr')
        return torch.linalg.slogdet(gd_jac_prod)[1]

    def compute_log_det_contr(self, x, x_next):
        N = x.shape[0]
        bs = self.gpu_bs_logdet or N
        log_det_contr = torch.zeros(1, device=x.device)
        for i in range(0, N, bs):
            x_i = x[i:i + bs]
            x_next_i = x_next[i:i + bs]
            log_det_contr += self._log_det_contr(x_i, x_next_i)
        return log_det_contr

    def descent_step(self, x: torch.Tensor, include_log_det: bool = True):
        N = x.shape[0]
        bs = self.gpu_bs_gen or N
        x_next = torch.zeros_like(x)
        for i in range(0, N, bs):
            x_i = x[i:i + bs]
            x_next[i:i + bs] = x_i - self.grad_step_size * self.grad_loss(x_i)
        log_det_sum = torch.zeros(1, device=x.device)
        if include_log_det:
            log_det_sum += self.compute_log_det_contr(x, x_next)
        x = self.project_op(x, x_next)
        return x, log_det_sum


class GeneratorMF(GeneratorABC):
    def __init__(self, energy_fn: energies.Energy, target_energy: torch.Tensor, grad_steps: int, grad_step_size: float,
                 gpu_bs_gen: int | None = None, gpu_bs_logdet: int | None = None, project: bool = False,
                 grad_step_size_expand: float = 0.):
        super().__init__(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_gen, gpu_bs_logdet, project)
        self.energy_jac_fn = torch.func.vmap(torch.func.jacrev(self.energy_fn))
        self.grad_step_size_expand_rel = grad_step_size_expand / grad_step_size  # Expansion step size rel. to descent step size

    def __repr__(self):
        if self.grad_step_size_expand_rel != 0.:
            return f'eMF-MGDM-{repr(self.energy_fn)}-{self.grad_steps}-{self.grad_step_size}-{self.grad_step_size_expand_rel}'
        else:
            return f'MF-MGDM-{repr(self.energy_fn)}-{self.grad_steps}-{self.grad_step_size}'

    def _partial_loss(self, x_i: torch.Tensor, mean_energy: torch.Tensor) -> torch.Tensor:
        energy_i_centered = self.energy_fn(x_i) - self.target_energy
        mean_energy_centered = mean_energy - self.target_energy
        if self.grad_step_size_expand_rel:  # Don't know if there is a performance gain in doing this
            loss = torch.dot(mean_energy_centered - self.grad_step_size_expand_rel / 2. * energy_i_centered, energy_i_centered)
        else:
            loss = torch.dot(mean_energy_centered, energy_i_centered)
        return loss / self.loss_normalization

    grad_partial_loss = torch.func.vmap(torch.func.grad(_partial_loss, argnums=1), in_dims=(None, 0, None))
    hess_partial_loss = torch.func.vmap(torch.func.hessian(_partial_loss, argnums=1), in_dims=(None, 0, None))

    def loss(self, x: torch.Tensor):
        mean_energy = self.energy_fn(x).mean(0)
        return 0.5 * x.shape[0] * ((mean_energy - self.target_energy) ** 2).sum() / self.loss_normalization

    def compute_log_det_contr(self, x, x_next, energy_mean, sequentially=False):
        N, T = x.shape
        if sequentially:
            bs = 1
        else:
            bs = self.gpu_bs_logdet or N

        log_det_contr = torch.zeros(1, device=x.device)
        J_Ainv_J_sum = torch.zeros((self.energy_fn.dim, self.energy_fn.dim), device=x.device)
        for i in range(0, N, bs):
            x_is = x[i:i + bs]
            x_next_is = x_next[i:i + bs]
            mask = (x_next_is > 0.)[..., None] if self.project else 1.
            energy_hess = self.hess_partial_loss(x_is, energy_mean)
            energy_jacs = self.energy_jac_fn(x_is)
            A = torch.eye(T, device=x.device) - self.grad_step_size * mask * energy_hess
            Ainv_J = torch.linalg.solve(A, mask * energy_jacs.mT)
            J_Ainv_J = (energy_jacs @ Ainv_J).sum(0)
            J_Ainv_J_sum += J_Ainv_J
            log_det_contr += torch.linalg.slogdet(A).logabsdet.sum(0)
        B = torch.eye(self.energy_fn.dim, device=x.device) - self.grad_step_size * J_Ainv_J_sum / N / self.loss_normalization
        log_det_contr += torch.linalg.slogdet(B).logabsdet
        return log_det_contr

    def descent_step(self, x: torch.Tensor, include_log_det: bool = True):
        energy_mean = self.energy_fn(x).mean(0)
        N = x.shape[0]
        bs = self.gpu_bs_gen or N
        x_next = torch.zeros_like(x)
        for i in range(0, N, bs):
            x_i = x[i:i + bs]
            x_next[i:i + bs] = x_i - self.grad_step_size * self.grad_partial_loss(x_i, energy_mean)
        log_det_sum = torch.zeros(1, device=x.device)
        if include_log_det:
            log_det_sum += self.compute_log_det_contr(x, x_next, energy_mean)
        x = self.project_op(x, x_next)
        return x, log_det_sum


class RevKLLoss:
    def __init__(self, true_model: Model, latent_model: Model, sample_size: int, T: int,
                 device: torch.device):
        self.true_model = true_model
        self.latent_model = latent_model
        self.sample_size = sample_size
        self.T = T
        self.device = device

    def __call__(self, generator, reduction='mean'):
        assert reduction in ('mean', 'sum')
        x0 = self.latent_model.generate_sample(self.T, N=self.sample_size).to(self.device)
        x, log_det_sum = generator(x0)
        loss = self.rev_loss(x0, x, log_det_sum, reduction)
        return loss

    def _reduce(self, loss, reduction):
        if reduction == 'mean':
            return loss / self.sample_size
        elif reduction == 'sum':
            return loss
        else:
            raise ValueError(f'Unknown reduction {reduction}')

    def rev_loss(self, x0, x, log_det_sum, reduction, return_entropy_and_ll=False):
        entropy = - (self.latent_model.loglikelihood(x0).sum(0) - log_det_sum)
        loglikelihood = self.true_model.loglikelihood(x).sum(0)
        loss = -entropy - loglikelihood
        if return_entropy_and_ll:
            loss = torch.tensor([loss, entropy, loglikelihood], device=loss.device)
        return self._reduce(loss, reduction)

    def through_descent(self, generator: GeneratorABC, include_entropy_and_ll=True):
        kl_losses = torch.zeros((generator.grad_steps + 1, 3) if include_entropy_and_ll else generator.grad_steps + 1)
        x0 = self.latent_model.generate_sample(self.T, N=self.sample_size).to(self.device)
        x = x0.clone()
        log_det_sum = torch.zeros(1, device=x.device)
        kl_losses[0] = self.rev_loss(x0, x, log_det_sum, reduction='mean', return_entropy_and_ll=include_entropy_and_ll).detach().cpu()
        bar = trange(generator.grad_steps, desc='KL: ')
        for k in bar:
            x, log_det_contr = generator.descend(x, include_log_det=True, steps=1)
            log_det_sum += log_det_contr
            kl_losses[k + 1] = self.rev_loss(x0, x, log_det_sum, reduction='mean', return_entropy_and_ll=include_entropy_and_ll).detach().cpu()
            bar.set_description(f'KL: {kl_losses[k + 1, 0 if include_entropy_and_ll else None]:.2f}')
            tqdm.write(f'KL loss {kl_losses[k + 1, 0 if include_entropy_and_ll else None]:.2f}')
        print(f'Minimum KL: {kl_losses[:, 0].min():.2f} at step {kl_losses[:, 0].argmin()}')
        print(f'Final KL: {kl_losses[-1, 0]:.2f}')
        return kl_losses

    def through_descent_batched(self, generator: GeneratorABC, batch_size: int, x0: torch.Tensor,
                                include_entropy_and_ll=True):
        x = x0.clone()
        kl_losses = torch.zeros((generator.grad_steps + 1, 3) if include_entropy_and_ll else generator.grad_steps + 1)
        N = x0.shape[0]
        assert N % batch_size == 0
        log_det_sums = torch.zeros(N // batch_size, device=x.device)
        bar = trange(generator.grad_steps + 1, desc='KL: ')
        for k in bar:
            for ii, i in enumerate(range(0, N, batch_size)):
                x_i = x[i:i + batch_size]
                kl_losses[k] += self.rev_loss(x0[i:i + batch_size], x_i, log_det_sums[ii], reduction='sum',
                                              return_entropy_and_ll=include_entropy_and_ll).detach().cpu()
                x_i, log_det_contr = generator.descend(x_i, include_log_det=True, steps=1)
                x[i:i + batch_size] = x_i
                log_det_sums[ii] += log_det_contr.squeeze()
            kl_losses[k] /= N
            bar.set_description(f'KL: {kl_losses[k, 0 if include_entropy_and_ll else None]:.2f}')
            tqdm.write(f'KL loss {kl_losses[k, 0 if include_entropy_and_ll else None]:.2f}')
        print(f'Minimum KL: {kl_losses[:, 0].min():.2f} at step {kl_losses[:, 0].argmin()}')
        print(f'Final KL: {kl_losses[-1, 0]:.2f}')
        return kl_losses

    def to(self, device):
        self.device = device
        return self

    def cpu(self):
        self.device = torch.device('cpu')
        return self
