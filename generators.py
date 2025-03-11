import torch
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
import energies
from models import Model
import numpy as np


class LR(ABC):
    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __call__(self, steps: int, i: int):
        pass


class GeneratorABC(ABC):
    def __init__(self, energy_fn: energies.Energy, target_energy: torch.Tensor, grad_steps: int, grad_step_size: float | LR,
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
                 include_errors: bool = False, step_nbr: int | None = 0, track_indices: tuple[int, ...] = ()):
        return self.descend(x0, include_log_det, steps, include_errors, step_nbr, track_indices)

    @abstractmethod
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def descent_step(self, x: torch.Tensor, step_nbr: int | None, include_log_det: bool = True) -> (torch.Tensor, torch.Tensor):
        pass

    def mean_loss(self, x: torch.Tensor) -> float:
        return self.loss(x).mean().item()

    def descend(self, x0: torch.Tensor, include_log_det: bool, steps: int | None = None, include_errors: bool = False,
                step_nbr: int | None = 0, track_indices: tuple[int, ...] = ()):
        steps = steps or self.grad_steps
        x = x0
        errors = []
        if include_errors:
            errors.append(self.mean_loss(x))
        if track_indices:
            track_indices = list(track_indices)
            tracked_x = torch.zeros((len(track_indices), steps + 1, *x0.shape[1:]), device=x0.device)
            tracked_x[:, 0] = x0[track_indices]
        log_det_sum = torch.zeros(1, device=x.device)
        if steps > 1:
            print('generating samples by gradient descent:')
        for t in (range if steps == 1 else trange)(steps):
            x, log_det_sum_t = self.descent_step(x, step_nbr + t, include_log_det)
            log_det_sum += log_det_sum_t
            if torch.isnan(x).any():
                raise RuntimeError(f'NaN encountered in descend at iteration {t}')
            if include_errors:
                errors.append(self.mean_loss(x))
            if track_indices:
                tracked_x[:, t + 1] = x[track_indices]
        rets = [x]
        if include_log_det:
            rets.append(log_det_sum)
        if include_errors:
            rets.append(np.array(errors))
        if track_indices:
            rets.append(tracked_x)
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
        return log_det_contr

    def compute_log_det_contr(self, x, x_next):
        N = x.shape[0]
        bs = self.gpu_bs_logdet or N
        log_det_contr = torch.zeros(1, device=x.device)
        for i in range(0, N, bs):
            x_i = x[i:i + bs]
            x_next_i = x_next[i:i + bs]
            log_det_contr += self._log_det_contr(x_i, x_next_i)
        return log_det_contr

    def descent_step(self, x: torch.Tensor, step_nbr: int | None, include_log_det: bool = True):
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

    def descent_step(self, x: torch.Tensor, step_nbr: int | None, include_log_det: bool = True):
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


class LRConst(LR):
    def __init__(self, lr: float):
        self._lr = lr

    def __repr__(self):
        return repr(self._lr)

    def __call__(self, steps: int, i: int):
        return self._lr


class LRStep(LR):
    def __init__(self, start, stop, levels=2):
        self.start = start
        self.stop = stop
        self.levels = levels
        if self.levels == 1:
            assert self.start == self.stop
        if self.levels < 1:
            raise ValueError('Number of levels must be at least 1')

    def __repr__(self):
        if self.levels == 1:
            return repr(self.start)
        return f'LRStep({self.start:.1E},{self.stop:.1E},{self.levels})'

    def __call__(self, steps: int, i: int):
        if self.levels == 1:
            return self.start
        return self.start + (self.stop - self.start) / (self.levels - 1) * (i // (steps / self.levels))


class LRLinear(LR):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __repr__(self):
        if self.start == self.stop:
            return repr(self.start)
        return f'LRLinear({self.start:.1E},{self.stop:.1E})'

    def __call__(self, steps: int, i: int):
        return self.start + (self.stop - self.start) * i / (steps - 1)


class LRGeom(LR):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __repr__(self):
        if self.start == self.stop:
            return repr(self.start)
        return f'LRGeom({self.start:.1E},{self.stop:.1E})'

    def __call__(self, steps: int, i: int):
        return self.start * (self.stop / self.start) ** (i / (steps - 1))


class LRNSteps(LR):
    def __init__(self, levels: list[float], break_points: list[float]):
        self.levels = levels
        self.break_points = break_points
        assert all(0. < bp < 1. for bp in break_points)
        assert len(levels) == len(break_points) + 1

    def __repr__(self):
        return f'LRNSteps({self.levels},{self.break_points})'.replace(' ', '')

    def __call__(self, steps: int, i: int):
        for j, bp in enumerate(self.break_points):
            if i / steps < bp:
                return self.levels[j]
        return self.levels[-1]


class GeneratorConstrained(GeneratorABC):
    def __init__(self, energy_fn: energies.Energy, target_energy: torch.Tensor, grad_steps: int,
                 grad_step_size: float | LR, gpu_bs_gen: int | None = None, gpu_bs_logdet: int | None = None,
                 gpu_bs_loss: int | None = None,
                 project: bool = False, is_mf: bool = False,
                 constr_efn: energies.Energy | None = None,
                 constr_target: torch.Tensor | None = None,
                 constr_grad_step_size: float | LR | None = None,
                 final_fn: callable = None,
                 final_fn_applications: int = 1,
                 approx_logdet: bool = False,
                 hutchinson_seed: tuple[int, int] | None = None,
                 hutchinson_mc_count: int = 2 ** 8,
                 hutchinson_min_terms: int = 10,
                 hutchinson_geom_param: float = .5):
        grad_step_size = grad_step_size if isinstance(grad_step_size, LR) else LRConst(grad_step_size)
        super().__init__(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_gen, gpu_bs_logdet, project)
        self.gpu_bs_loss = gpu_bs_loss
        self.is_mf = is_mf
        self.vjac_energy = torch.func.vmap(torch.func.jacrev(self.energy_fn))
        self.final_fn_applications = final_fn_applications
        self.final_fn = final_fn
        self.final_fn_vgrad = torch.func.vmap(torch.func.grad(final_fn)) if final_fn is not None else None
        self.final_fn_vjac = torch.func.vmap(torch.func.jacrev(final_fn)) if final_fn is not None else None
        self.constr_efn = constr_efn
        self.constr_target = constr_target
        if isinstance(constr_grad_step_size, float):
            self.constr_grad_step_size = LRConst(constr_grad_step_size)
        else:
            self.constr_grad_step_size = constr_grad_step_size

        self.approx_logdet = approx_logdet
        self.hutchinson_mc_count = None
        self.hutchinson_min_terms = None
        self.hutchinson_seed = None
        self.hutchinson_rng = None
        self.hutchinson_geom_param = None
        if approx_logdet:
            assert hutchinson_seed is not None
            self.hutchinson_seed = hutchinson_seed
            self.hutchinson_rng = np.random.default_rng(hutchinson_seed)
            self.hutchinson_mc_count = hutchinson_mc_count
            self.hutchinson_min_terms = hutchinson_min_terms
            assert 0. < hutchinson_geom_param < 1.
            self.hutchinson_geom_param = hutchinson_geom_param

        constr_args_not_none = [constr_efn is not None, constr_target is not None, constr_grad_step_size is not None]
        if any(constr_args_not_none):
            assert all(constr_args_not_none), 'All constraint arguments must be provided'

    def __repr__(self):
        r = f'{"MF-" if self.is_mf else ""}CMGDM-{repr(self.energy_fn)}-{self.grad_steps}-{repr(self.grad_step_size)}'
        if self.constr_efn is not None:
            r += f'-{repr(self.constr_efn)}-{repr(self.constr_grad_step_size)}'
        if self.approx_logdet:
            r += f'-H({self.hutchinson_seed},{self.hutchinson_mc_count},{self.hutchinson_min_terms},{self.hutchinson_geom_param})'
        return r

    @property
    def constrained(self) -> bool:
        return self.constr_efn is not None

    def mean_energy(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        bs = self.gpu_bs_loss or N
        mean_energy = 0.
        for i in range(0, N, bs):
            mean_energy += self.energy_fn(x[i:i + bs]).sum(0) / N
        return mean_energy

    def mean_loss(self, x: torch.Tensor) -> float | list[float]:
        loss = self.vloss(x).mean().item()
        if self.constrained:
            return [loss, self.vconstr_loss(x).mean().item()]
        else:
            return loss

    def vloss(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_mf:
            return self.vloss_mf(x)
        else:
            return self.vloss_reg(x)

    loss = vloss

    def vgrad_loss(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.is_mf:
            return self.vgrad_loss_mf_partial(x, *args)
        else:
            return self.vgrad_loss_reg(x, *args)

    def vhess_loss(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.is_mf:
            return self.vhess_loss_mf_partial(x, *args)
        else:
            return self.vhess_loss_reg(x, *args)

    def _loss_reg(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((self.energy_fn(x) - self.target_energy) ** 2).sum()

    _vloss_reg = torch.func.vmap(_loss_reg, in_dims=(None, 0))
    grad_loss_reg = torch.func.grad(_loss_reg, argnums=1)
    vgrad_loss_reg = torch.func.vmap(grad_loss_reg, in_dims=(None, 0))
    vhess_loss_reg = torch.func.vmap(torch.func.hessian(_loss_reg, argnums=1), in_dims=(None, 0))

    def vloss_reg(self, x: torch.Tensor) -> torch.Tensor:
        losses = []
        N = x.shape[0]
        batch_size = self.gpu_bs_loss or N
        for i in range(0, N, batch_size):
            losses.append(self._vloss_reg(x[i:i + batch_size]))
        return torch.cat(losses)

    def vloss_mf(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.shape[0] * ((self.mean_energy(x) - self.target_energy) ** 2).sum()

    def _loss_mf_partial(self, x_i: torch.Tensor, mean_energy: torch.Tensor) -> torch.Tensor:
        # Only for computing gradients
        energy_i_centered = self.energy_fn(x_i) - self.target_energy
        mean_energy_centered = mean_energy - self.target_energy
        loss = torch.dot(mean_energy_centered, energy_i_centered)
        return loss

    grad_loss_mf_partial = torch.func.grad(_loss_mf_partial, argnums=1)
    vgrad_loss_mf_partial = torch.func.vmap(grad_loss_mf_partial, in_dims=(None, 0, None))
    vhess_loss_mf_partial = torch.func.vmap(torch.func.hessian(_loss_mf_partial, argnums=1), in_dims=(None, 0, None))

    def _constr_loss(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((self.constr_efn(x) - self.constr_target) ** 2).sum()

    vconstr_loss = torch.func.vmap(_constr_loss, in_dims=(None, 0))
    vgrad_constr_loss = torch.func.vmap(torch.func.grad(_constr_loss, argnums=1), in_dims=(None, 0))
    vhess_constr_loss = torch.func.vmap(torch.func.hessian(_constr_loss, argnums=1), in_dims=(None, 0))

    def _get_hutchinson_factors(self, N_terms: int, device: torch.device):
        return torch.cat([torch.ones(self.hutchinson_min_terms, device=device),
                          (1 - self.hutchinson_geom_param) ** (torch.arange(1, N_terms - self.hutchinson_min_terms + 1, device=device) - 1)])

    def hutchinson_est_reg(self, x: torch.Tensor, vs: torch.Tensor, N_terms: int, step_nbr: int):
        # x is not batched here, use vmap outside instead
        rr_factors = self._get_hutchinson_factors(N_terms, x.device)
        N_vs = vs.shape[0]
        # def flat_grad_loss_reg(_x):
        #     return self.grad_loss_reg(_x).flatten()
        _, vjp = torch.func.vjp(self.grad_loss_reg, x)
        vvjp = torch.func.vmap(vjp)
        log_det_contr = 0.
        u_i = vs
        for i in range(N_terms):
            u_i = self.grad_step_size(self.grad_steps, step_nbr) * vvjp(u_i)[0]
            contr = (u_i.reshape(N_vs, 1, -1) @ vs.reshape(N_vs, -1, 1)).squeeze(-1).squeeze(-1).mean() / ((i + 1) * rr_factors[i])
            log_det_contr += contr
        return log_det_contr

    vhutchinson_est_reg = torch.func.vmap(hutchinson_est_reg, in_dims=(None, 0, 0, None, None))

    def hutchinson_est_mf(self, x: torch.Tensor, mean_energy: torch.Tensor, vs: torch.Tensor, N_terms: int, step_nbr: int):
        """
        Compute the Hutchinson trace estimation of J_{\bar{f}}^p for exponents p = 1, ..., N_terms, where
        \bar{g}(x) = x - \bar{f}(x) is the mean-field gradient step, i.e.
        x^{(n)}_{t+1} = x^{(n)}_t - \gamma J_\phi(x^{(n)}_t) (\bar{\Phi}(x_t) - \alpha).
                                    |<-----           this is \bar{f}           ----->|
        We need to compute vjp with the Jacobian J_{\bar{f}} (N_terms times). This is divided into two terms
        (ignoring gamma):
            * J1(x^n) = sum_k H_k(x^n) (\bar(Phi)(x) - \alpha), and
            * J2(x) = 1 / N * \mathcal{J}_{\Phi}(x)^T \mathcal{J}_{\Phi}(x)
        """
        N = x.shape[0]  # batch size
        assert vs.shape[0] == N
        N_vs = vs.shape[1]
        signal_shape = x.shape[1:]
        signal_size = np.prod(signal_shape)

        rr_factors = self._get_hutchinson_factors(N_terms, x.device)

        def wrapped_grad_loss(_x):
            return self.grad_loss_mf_partial(_x, mean_energy)

        def jvpJ1(_x, _v):
            return torch.func.jvp(wrapped_grad_loss, (_x,), (_v,))[1]

        vjvpJ1 = torch.func.vmap(torch.func.vmap(jvpJ1, in_dims=(None, 0)), in_dims=(0, 0))
        J2s = self.vjac_energy(x).reshape(N, self.energy_fn.dim, signal_size).unsqueeze(1)  # (N, 1, K, signal_size)

        log_det_contr = 0.
        u_i = vs
        for i in range(N_terms):
            u_i1 = vjvpJ1(x, u_i)
            u_i2 = 1 / N * (J2s.mT @ ((J2s @ u_i.reshape(N, N_vs, -1, 1)).sum(0))).squeeze(-1).reshape(u_i.shape)
            u_i = u_i1 + u_i2
            u_i = u_i * self.grad_step_size(self.grad_steps, step_nbr)
            log_det_contr += (vs * u_i).mean(1).sum() / ((i + 1) * rr_factors[i])
        return log_det_contr

    def compute_log_det_contr(self, x, x_next, step_nbr: int, loss_args: tuple, sequentially=False):
        if self.approx_logdet:
            return self.compute_approx_log_det_contr(x, x_next, step_nbr, loss_args, sequentially)
        else:
            return self.compute_exact_log_det_contr(x, x_next, step_nbr, loss_args, sequentially)

    def compute_approx_log_det_contr(self, x, x_next, step_nbr: int, loss_args: tuple, sequentially: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        signal_shape = x.shape[1:]
        if len(signal_shape) != 2:
            raise NotImplementedError("Need to check that code doesn't assume exactly 2D signals")
        N_terms = self.hutchinson_min_terms + self.hutchinson_rng.geometric(self.hutchinson_geom_param)
        vs = torch.from_numpy(self.hutchinson_rng.standard_normal((batch_size, self.hutchinson_mc_count) + signal_shape)).to(torch.float32).to(x.device)
        if self.constrained:
            raise NotImplementedError()
        if self.project:
            raise NotImplementedError()
        if self.is_mf:
            log_det_contr = self.hutchinson_est_mf(x, *loss_args, vs, N_terms, step_nbr)
        else:
            log_det_contr = self.vhutchinson_est_reg(x, vs, N_terms, step_nbr)
            assert log_det_contr.shape == (batch_size,)
            log_det_contr = log_det_contr.sum()
        return log_det_contr

    def compute_exact_log_det_contr(self, x, x_next, step_nbr: int, loss_args: tuple, sequentially: bool) -> torch.Tensor:
        N = x.shape[0]
        signal_shape = x.shape[1:]
        signal_size = np.prod(signal_shape)

        if sequentially:
            bs = 1
        else:
            bs = self.gpu_bs_logdet or N

        log_det_contr = torch.zeros(1, device=x.device)

        if self.is_mf:
            J_Ainv_J_sum = torch.zeros((self.energy_fn.dim, self.energy_fn.dim), device=x.device)
        else:
            J_Ainv_J_sum = torch.zeros(1, device=x.device)

        for i in range(0, N, bs):
            x_is = x[i:i + bs]
            x_next_is = x_next[i:i + bs]
            mask = (x_next_is > 0.).reshape(x_next_is.shape[0], signal_size, 1) if self.project else 1.

            energy_hess = self.vhess_loss(x_is, *loss_args)
            hess_sum = self.grad_step_size(self.grad_steps, step_nbr) * energy_hess
            if self.constrained:
                constr_hess = self.vhess_constr_loss(x_is)
                hess_sum += self.constr_grad_step_size(self.grad_steps, step_nbr) * constr_hess
            hess_sum = hess_sum.reshape(hess_sum.shape[0], signal_size, signal_size)

            A = torch.eye(signal_size, device=x.device) - mask * hess_sum
            log_det_contr += torch.linalg.slogdet(A).logabsdet.sum(0)

            if self.is_mf:
                energy_jacs = self.vjac_energy(x_is).reshape(x_is.shape[0], self.energy_fn.dim, signal_size)
                Ainv_J = torch.linalg.solve(A, mask * energy_jacs.mT)
                J_Ainv_J = (energy_jacs @ Ainv_J).sum(0)
                J_Ainv_J_sum += J_Ainv_J

        if self.is_mf:
            B = torch.eye(self.energy_fn.dim, device=x.device) - self.grad_step_size(self.grad_steps, step_nbr) * J_Ainv_J_sum / N
            log_det_contr += torch.linalg.slogdet(B).logabsdet

        return log_det_contr

    def descent_step(self, x: torch.Tensor, step_nbr: int | None, include_log_det: bool = True):
        log_det_sum = torch.zeros(1, device=x.device)
        if step_nbr >= self.grad_steps - self.final_fn_applications and self.final_fn is not None:
            if include_log_det:
                # log_det_sum = torch.log(torch.abs(self.final_fn_vgrad(x.reshape(-1)))).sum()
                N = x.shape[0]
                signal_shape = x.shape[1:]
                signal_size = np.prod(signal_shape)
                jacs = self.final_fn_vjac(x)
                log_det_sum = torch.linalg.slogdet(jacs.reshape(N, signal_size, signal_size)).logabsdet.sum()
            x = self.final_fn(x)
            return x, log_det_sum

        loss_args = (self.mean_energy(x),) if self.is_mf else ()
        N = x.shape[0]
        bs = self.gpu_bs_gen or N
        x_next = torch.zeros_like(x)
        for i in range(0, N, bs):
            x_i = x[i:i + bs]
            x_next_i = x_i - self.grad_step_size(self.grad_steps, step_nbr) * self.vgrad_loss(x_i, *loss_args)
            if self.constrained:
                x_next_i -= self.constr_grad_step_size(self.grad_steps, step_nbr) * self.vgrad_constr_loss(x_i)
            x_next[i:i + bs] = x_next_i
        if include_log_det:
            log_det_sum += self.compute_log_det_contr(x, x_next, step_nbr, loss_args)
        x = self.project_op(x, x_next)
        return x, log_det_sum

    def to(self, device):
        super().to(device)
        if self.constrained:
            self.constr_efn.to(device)
            self.constr_target = self.constr_target.to(device)
        try:
            self.final_fn = self.final_fn.to(device)
        except AttributeError:
            pass
        return self

    def cpu(self):
        super().cpu()
        if self.constrained:
            self.constr_efn.cpu()
            self.constr_target = self.constr_target.cpu()
        try:
            self.final_fn = self.final_fn.cpu()
        except AttributeError:
            pass
        return self


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
        # batch_size must not be less than mean-field parameter N
        x = x0.clone()
        kl_losses = torch.zeros((generator.grad_steps + 1, 3) if include_entropy_and_ll else generator.grad_steps + 1)
        N = x0.shape[0]
        assert N == batch_size  # required for MF-MGDM
        log_det_sums = torch.zeros(N // batch_size, device=x.device)
        bar = trange(generator.grad_steps + 1, desc='KL: ')
        for k in bar:
            for ii, i in enumerate(range(0, N, batch_size)):
                x_i = x[i:i + batch_size]
                kl_losses[k] += self.rev_loss(x0[i:i + batch_size], x_i, log_det_sums[ii], reduction='sum',
                                              return_entropy_and_ll=include_entropy_and_ll).detach().cpu()
                x_i, log_det_contr = generator.descend(x_i, include_log_det=True, steps=1, step_nbr=k)
                x[i:i + batch_size] = x_i
                log_det_sums[ii] += log_det_contr.squeeze()
            kl_losses[k] /= N
            bar.set_description(f'KL: {kl_losses[k, 0 if include_entropy_and_ll else None]:.2f}')
            tqdm.write(f'KL loss {kl_losses[k, 0 if include_entropy_and_ll else None]:.2f}')
        print(f'Minimum KL: {kl_losses[:, 0 if include_entropy_and_ll else None].min():.2f} at step {kl_losses[:, 0].argmin()}')
        print(f'Final KL: {kl_losses[-1, 0 if include_entropy_and_ll else None]:.2f}')
        return kl_losses

    def to(self, device):
        self.device = device
        return self

    def cpu(self):
        self.device = torch.device('cpu')
        return self


class EntropyCalculator:
    def __init__(self, latent_model: Model):
        self.latent_model = latent_model

    def through_descent(self, generator: GeneratorABC, batch_size: int, x0: torch.Tensor):
        # batch_size must not be less than mean-field parameter N
        x = x0.clone()
        entropies = torch.zeros(generator.grad_steps + 1)
        N = x0.shape[0]
        assert N == batch_size  # required for MF-MGDM
        entropies[0] = - self.latent_model.loglikelihood(x0).mean(0).detach().cpu()
        bar = trange(generator.grad_steps, desc='Entropy: ')
        for k in bar:
            log_det_contrs = 0.
            for ii, i in enumerate(range(0, N, batch_size)):
                x_i = x[i:i + batch_size]
                x_i, log_det_contr = generator.descend(x_i, include_log_det=True, steps=1, step_nbr=k)
                x[i:i + batch_size] = x_i
                log_det_contrs += log_det_contr.squeeze().sum().detach().cpu()
            entropies[k + 1] = entropies[k] + log_det_contrs / N
            bar.set_description(f'Entropy: {entropies[k]:.2f}')
            tqdm.write(f'Entropy {entropies[k]:.2f}')
        print(f'Initial entropy: {entropies[0]:.2f}')
        print(f'Maximum entropy: {entropies.max():.2f} at step {entropies.argmax()}')
        print(f'Minimum entropy: {entropies.min():.2f} at step {entropies.argmin()}')
        print(f'Final entropy: {entropies[-1]:.2f}')
        return entropies
