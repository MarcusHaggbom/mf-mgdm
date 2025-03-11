from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import scipy
from kymatio.scattering1d.filter_bank import scattering_filter_factory as filter_bank_1d
from kymatio.scattering2d.filter_bank import filter_bank as filter_bank_2d
from kymatio.torch import Scattering1D, Scattering2D
from phaseharmonics2d.phase_harmonics_k_bump_isotropic_noreflect_norml import PhaseHarmonics2d
import utils
import scatspectra
from models import Model


class Energy(ABC):
    @ abstractmethod
    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    def to(self, device: torch.device):
        return self

    def cpu(self):
        return self


class LpNorm(Energy):
    def __init__(self, p: float, dim=1):
        self.p = p
        self.signal_dim = dim

    def __call__(self, x):
        return (x.abs() ** self.p).mean(axis=tuple(range(-self.signal_dim, 0)))[..., None]

    def __repr__(self):
        return f'L{self.p}Norm'

    @property
    def dim(self):
        return 1


class IsingStat2d(Energy):
    def __call__(self, x):
        return (x * (torch.roll(x, 1, dims=-1) + torch.roll(x, 1, dims=-2))).mean((-1, -2))[..., None]

    def __repr__(self):
        return 'IsingStat2d'

    @property
    def dim(self):
        return 1


class Square(Energy):
    def __init__(self, signal_size, dim=1):
        self.signal_size = signal_size
        self.signal_dim = dim

    def __call__(self, x):
        return (x.abs() ** 2).reshape(*x.shape[:-self.dim], self.signal_size)

    def __repr__(self):
        return 'Square'

    @property
    def dim(self):
        return self.signal_size


class Abs(Energy):
    def __init__(self, signal_size, dim=1):
        self.signal_size = signal_size
        self.signal_dim = dim

    def __call__(self, x):
        return (x.abs()).reshape(*x.shape[:-self.dim], self.signal_size)

    def __repr__(self):
        return 'Abs'

    @property
    def dim(self):
        return self.signal_size


class ACF(Energy):
    def __init__(self, logT):
        self.T = 2 ** logT

    def __call__(self, x):
        lag = 1
        return torch.stack([(x[..., lag:] * x[..., :-lag]).sum(-1),
                                    (x[..., lag:] ** 2).sum(-1)], dim=-1) / self.T

    def __repr__(self):
        return 'ACF(1)'

    @property
    def dim(self):
        return 2


class ACF2(Energy):
    def __init__(self, logT, lags=10):
        self.T = 2 ** logT
        self.lags = lags

    def __call__(self, x):
        x2 = x ** 2
        return torch.stack([(x ** 2).sum(-1), (x[..., 1:] * x[..., :-1]).sum(-1)]
                           + [(x2[..., lag:] * x2[..., :-lag]).sum(-1) for lag in range(1, self.lags + 1)], dim=-1) / self.T

    def __repr__(self):
        return f'ACF2({self.lags})'

    @property
    def dim(self):
        return 2 + self.lags


class ScatABC(Energy):
    def __init__(self, logT, j1, j2, dim=1, Q=1):
        assert j1 is not None and j2 is not None
        self.j1 = j1
        self.j2 = j2
        self.logT = logT
        if dim > 2:
            raise NotImplementedError('Only 1D and 2D scattering is supported')
        self.signal_dim = dim
        self.Q = Q
        self.phi, self.psi1, self.psi2 = self.get_filters(logT, J=max(j1, j2, 8 if dim == 1 else 0), Q=self.Q, dim=self.signal_dim)
        if dim == 1:
            self.psi1 = self.psi1[:j1*self.Q]
            self.psi2 = self.psi2[:j2*self.Q]
        self.phi = torch.from_numpy(self.phi)
        self.psi1 = torch.from_numpy(self.psi1)
        self.psi2 = torch.from_numpy(self.psi2)
        self._dim = None

    @staticmethod
    def get_filters(logT, J, Q, dim):
        if dim == 1:
            phi_f, psi1_f, psi2_f = filter_bank_1d(2 ** logT, J, (Q, 1), 2 ** J)
            phi_f = phi_f['levels'][0]
            psi1_f = np.stack([psi['levels'][0] for psi in psi1_f])
            psi2_f = np.stack([psi['levels'][0] for psi in psi2_f])
        elif dim == 2:
            filters = filter_bank_2d(2 ** logT, 2 ** logT, J, L=Q)
            phi_f = filters['phi']['levels'][0]
            psi1_f = np.stack([psi['levels'][0] for psi in filters['psi']])
            psi2_f = psi1_f.copy()
        else:
            raise ValueError('Invalid signal dimension')
        return phi_f.astype(np.float32), psi1_f.astype(np.float32), psi2_f.astype(np.float32)

    def convolve(self, signal, fourier_filter):
        d = self.signal_dim
        if utils.is_torch_else_numpy(signal):
            fft = torch.fft.fft2 if d == 2 else torch.fft.fft
            ifft = torch.fft.ifft2 if d == 2 else torch.fft.ifft
            fourier_signal = fft(signal)
            fourier_conv = fourier_signal.unsqueeze(-(d + 1)) * fourier_filter
            return ifft(fourier_conv)
        else:
            fft = scipy.fft.fft2 if d == 2 else scipy.fft.fft
            ifft = scipy.fft.ifft2 if d == 2 else scipy.fft.ifft
            fourier_signal = fft(signal, axis=-1)  # Uses scipy due to https://github.com/numpy/numpy/issues/17801
            fourier_conv = np.expand_dims(fourier_signal, -(d + 1)) * fourier_filter
            return ifft(fourier_conv)

    @staticmethod
    @abstractmethod
    def nl(x):
        pass

    def to(self, device: torch.device):
        self.phi = self.phi.to(device)
        self.psi1 = self.psi1.to(device)
        self.psi2 = self.psi2.to(device)
        return self

    def cpu(self):
        self.phi = self.phi.cpu()
        self.psi1 = self.psi1.cpu()
        self.psi2 = self.psi2.cpu()
        return self

    @property
    def device(self):
        return self.psi1.device

    @property
    def dim(self):
        if self._dim is None:
            signal_shape = 2 ** self.logT if self.signal_dim == 1 else (2 ** self.logT, 2 ** self.logT)
            self._dim = self(torch.zeros(signal_shape, device=self.device)).shape[0]
        return self._dim


class ScatMean(ScatABC):
    def __call__(self, x):
        expanded = False
        if x.dim() == 1:
            x = x[None, :]
            expanded = True

        N, T = x.shape
        s1 = self.nl(self.convolve(x, self.psi1))
        s2 = self.nl(self.convolve(s1, self.psi2))
        phi = torch.cat([s1.mean(-1), s2.mean(-1).view((N, -1))], dim=1)

        if expanded:
            assert phi.shape[0] == 1
            phi = phi.squeeze(0)

        return phi

    def __repr__(self):
        return f'ScatMean({self.j1},{self.j2})'

    @staticmethod
    def nl(x):
        return x.abs()


class ScatMean2d(ScatABC):
    def __call__(self, x):
        expanded = False
        if x.dim() == self.signal_dim:
            x = x[None, ...]
            expanded = True

        N = x.shape[0]
        Q = self.Q
        s1 = self.nl(self.convolve(x, self.psi1))
        assert len(self.psi1) == len(self.psi2) == Q * self.j1
        axis = tuple(range(-self.signal_dim, 0))
        s2 = torch.cat([self.nl(self.convolve(s1[:, j * Q:(j + 1) * Q], self.psi2[(j + 1) * Q:])).mean(axis=axis).view((N, -1)) for j in range(self.j1 - 1)], dim=1)
        s_list = [s1.mean(axis=axis), s2]
        # if self.signal_dim == 2:
        #     s0 = self.nl(self.convolve(x, self.phi))
        #     s_list = [s0.mean(axis=axis)] + s_list
        phi = torch.cat(s_list, dim=1)

        if expanded:
            assert phi.shape[0] == 1
            phi = phi.squeeze(0)

        return phi

    def __repr__(self):
        d = '' if self.signal_dim == 1 else f'{self.signal_dim}d'
        return f'ScatMean{d}({self.j1},{self.j2},{self.Q})'

    @staticmethod
    def nl(x):
        return x.abs()


class ScatKymatio(Energy):
    def __init__(self, logT, J, Q, dim=1):
        self.logT = logT
        self.J = J
        self.Q = Q
        self.signal_dim = dim
        if dim == 1:
            self.scat = Scattering1D(J=J, shape=2 ** logT, Q=Q)
        elif dim == 2:
            self.scat = Scattering2D(J=J, shape=(2 ** logT, 2 ** logT), L=Q)
        else:
            raise ValueError('Invalid signal dimension')

    def __repr__(self):
        return f'ScatKymatio({self.J},{self.Q})'

    def __call__(self, x):
        expanded = False
        if x.dim() == self.signal_dim:
            x = x[None, ...]
            expanded = True
        phi = self.scat(x).mean(axis=tuple(range(-self.signal_dim, 0)))
        if expanded:
            assert phi.shape[0] == 1
            phi = phi.squeeze(0)

        return phi

    @property
    def dim(self):
        T = 2 ** self.logT
        x = torch.zeros(tuple(T for _ in range(self.signal_dim)))
        return self(x).shape[-1]

    def to(self, device: torch.device):
        self.scat = self.scat.to(device)
        return self

    def cpu(self):
        self.scat = self.scat.cpu()
        return self


class WaveletL1(ScatABC):
    def __call__(self, x):
        expanded = False
        if x.dim() == self.signal_dim:
            x = x[None, ...]
            expanded = True

        N = x.shape[0]
        w1 = self.nl(self.convolve(x, self.psi1))
        axis = tuple(range(-self.signal_dim, 0))
        phi = w1.mean(axis=axis)

        if expanded:
            assert phi.shape[0] == 1
            phi = phi.squeeze(0)

        return phi

    def __repr__(self):
        d = '' if self.signal_dim == 1 else f'{self.signal_dim}d'
        return f'WaveletL1{d}({self.j1},{self.Q})'

    @staticmethod
    def nl(x):
        return x.abs()


class ScatCov(ScatABC):
    def __init__(self, logT: int, j1: int, j2: int, phase_shifts: int):
        super().__init__(logT, j1, j2)
        self.phase_shifts = phase_shifts
        self.psi1 = self.shift_phases(self.psi1, torch.arange(phase_shifts) * torch.pi / 3)
        self.triu_idx = torch.triu_indices(self.scat_dim, self.scat_dim)

    def __repr__(self):
        return f'ScatCov({self.j1},{self.j2},{self.phase_shifts})'

    @staticmethod
    def shift_phases(filters, phases):
        shifts = torch.exp(-1.j * phases)
        shifted_filters = filters[:, None, :] * shifts[None, :, None]
        return shifted_filters.reshape((-1, shifted_filters.shape[-1]))

    def __call__(self, x):
        expanded = False
        if x.dim() == 1:
            x = x[None, :]
            expanded = True

        N, T = x.shape
        s1 = self.nl(self.convolve(x, self.psi1))
        s2 = self.nl(self.convolve(s1, self.psi2))
        s = torch.cat([s1, s2.view((N, -1, T))], dim=1)  # (N, K, T)

        phi = self.cov(s)  # (N, K, K)
        phi = phi[:, self.triu_idx[0], self.triu_idx[1]]  # (N, K * (K + 1) / 2)

        if expanded:
            assert phi.shape[0] == 1
            phi = phi.squeeze(0)

        return phi

    @staticmethod
    def nl(x):
        return torch.relu(x.real)

    @property
    def scat_dim(self):
        return self.psi1.shape[0] + self.psi1.shape[0] * self.psi2.shape[0]

    @staticmethod
    def cov(x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        x_mean = x.mean(-1, keepdim=True)
        x_centered = x - x_mean
        cov = x_centered @ x_centered.transpose(-2, -1) / (T - 1)
        return cov

    @staticmethod
    def cross_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        T = x.shape[-1]
        x_mean = x.mean(-1, keepdim=True)
        y_mean = y.mean(-1, keepdim=True)
        cross_cov = ((x - x_mean)[..., None, :] @ (y - y_mean)[..., None]).squeeze(-2, -1) / (T - 1)
        return cross_cov

    def _infer_cov_coeff(self, idx: int) -> tuple[int, int]:
        return self.triu_idx[0, idx], self.triu_idx[1, idx]


class ScatCovPCA(ScatCov):
    def __init__(self, logT: int, j1: int, j2: int, phase_shifts: int, pca_model: Model,
                 projector: torch.Tensor | None = None):
        super().__init__(logT, j1, j2, phase_shifts)
        self.pca_model = pca_model
        if projector is None:
            self.pc_projector = None
            self.pc_projector = self._load_projector()
        else:
            self.pc_projector = projector

    def __repr__(self):
        return f'ScatCovPCA({self.j1},{self.j2},{self.phase_shifts},{repr(self.pca_model)})'

    def _load_projector(self) -> torch.Tensor:
        log10_sample_size = 5
        projector_dir = os.path.join('data', 'pca-projectors')
        os.makedirs(projector_dir, exist_ok=True)
        filename = f'{repr(self)}-{self.logT}-1e{log10_sample_size}.pt'
        projector_path = os.path.join(projector_dir, filename)
        if os.path.exists(projector_path):
            print(f'loading pca projector from {projector_path}')
            projector = torch.load(projector_path)
        else:
            print(f'Projector not found for {repr(self)}, generating samples...')
            samples = self.pca_model.generate_sample(2 ** self.logT, 10 ** log10_sample_size)
            print(f'... computing projector...')
            projector = self._compute_projector(samples)
            print(f'... done, saving projector to {projector_path}')
            torch.save(projector, projector_path)
        return projector

    def _compute_projector(self, x: torch.Tensor) -> torch.Tensor:
        from scipy.sparse.linalg import eigsh
        energies = self(x)
        cov = torch.cov(energies.T)
        eigvals, eigvecs = eigsh(cov.numpy(), k=20, which='LM')
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        projector = (eigvals ** -0.5 * eigvecs).T
        return torch.from_numpy(projector).to(x.dtype)

    def __call__(self, x):
        phi = super().__call__(x)
        if self.pc_projector is not None:
            phi = (self.pc_projector @ phi[..., None]).squeeze(-1)
        return phi

    def to(self, device: torch.device):
        super().to(device)
        self.pc_projector = self.pc_projector.to(device)
        return self

    def cpu(self):
        super().cpu()
        self.pc_projector = self.pc_projector.cpu()
        return self


class PhaseHarmonicCovD(Energy):
    def __init__(self, M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, stdnorm=1, kmax=None):
        self._J = J
        self._L = L
        self._delta_j = delta_j
        self._delta_l = delta_l
        self._delta_k = delta_k
        self.phase_harmonics = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, stdnorm, kmax=kmax)

    def __call__(self, x):
        return self.phase_harmonics(x)

    def __repr__(self):
        return f'PhaseHarmonicCovD({self._J},{self._L},{self._delta_j},{self._delta_l},{self._delta_k})'

    @property
    def dim(self):
        return 1 + 2 * len(self.phase_harmonics.this_wph['la1'])

    def to(self, device: torch.device):
        self.phase_harmonics = self.phase_harmonics.to(device)
        return self


class ScatSpectra(Energy):
    def __init__(self, logT, true_model_or_data: Model | torch.Tensor | None, include_phase: bool = True,
                 sigma2: torch.Tensor | None = None):
        self.include_phase = include_phase
        self.J = 7
        Q = 1
        wav_type = 'battle_lemarie'
        wav_norm = 'l1'
        high_freq = 0.425
        rpad = True

        if sigma2 is None:
            if isinstance(true_model_or_data, Model):
                true_samples = true_model_or_data.generate_sample(2 ** logT, 10000).unsqueeze(-2)  # (B, 1, T)
            elif isinstance(true_model_or_data, torch.Tensor):
                true_samples = true_model_or_data.clone()[None, None, :]
            else:
                raise ValueError(f'Invalid type {type(true_model_or_data)} for true_model_or_data')
            assert true_samples.ndim == 3
            sigma2 = scatspectra.frontend.compute_sigma2(true_samples, self.J, Q, wav_type, wav_norm, high_freq, rpad,
                                                         False, 1).mean(0, keepdim=True)
        else:
            if true_model_or_data is not None:
                print('Warning: true_model_or_data is ignored when sigma2 is given')
        self.sigma2 = sigma2
        self.model = scatspectra.Model(model_type='scat_spectra',
                                       T=2 ** logT,
                                       r=2,  # order
                                       N=1,
                                       J=self.J,
                                       Q=Q,
                                       wav_type=wav_type, wav_norm=wav_norm,
                                       high_freq=high_freq, rpad=rpad,
                                       A=None, channel_transforms=None,
                                       Ns=None,
                                       diago_n=True, cross_params=None, sigma2=sigma2, norm_on_the_fly=False,
                                       estim_operator=None,
                                       qs=None,
                                       histogram_moments=False,
                                       coeff_types=None, dtype=sigma2.dtype, skew_redundance=False, nchunks=1,
                                       sigma2_L1=None, sigma2_L2=None, sigma2_Lphix=None)
        self._dim = self(torch.zeros(2 ** logT)).shape[-1]

    def __repr__(self):
        energy_type = 'ScatSpectra' if self.include_phase else 'ScatSpectraAbs'
        return f'{energy_type}({self.J})'

    def __call__(self, x):
        expanded = x.dim() == 1
        if expanded:
            x = x.unsqueeze(0)
        x = x.unsqueeze(-2)
        Rx = self.model(x)
        reals = Rx.query(is_real=True).y
        imags = Rx.query(is_real=False).y
        y = torch.cat([reals.real, imags.abs()], dim=-2)  # Related issue as below, expected -1 but use -2
        if self.include_phase:
            y = torch.cat([y, imags.angle()], dim=-2)
        y = y.squeeze(-1)  # Expected it to be -2, might be a bug in package
        if expanded:
            assert y.shape[0] == 1
            y = y.squeeze(0)
        return y

    @property
    def dim(self):
        return self._dim

    def to(self, device: torch.device):
        if device.type == 'cuda':
            self.model = self.model.cuda()
            self.model.norm_layer.sigma = self.model.norm_layer.sigma.cuda()
        else:
            self.model = self.model.cpu()
            self.model.norm_layer.sigma = self.model.norm_layer.sigma.cpu()
        return self

    def cpu(self):
        self.model = self.model.cpu()
        self.model.norm_layer.sigma = self.model.norm_layer.sigma.cpu()
        return self


class Combined(Energy):
    def __init__(self, energies: list[Energy], weights: torch.Tensor | None = None):
        if weights is None:
            weights = torch.ones(len(energies))
        else:
            assert len(weights) == len(energies)
        self.energies = energies
        self.weights = weights
        self._dim = sum(energy.dim for energy in energies)

    def __call__(self, x):
        return torch.cat([energy(x) * w for (energy, w) in zip(self.energies, self.weights)], dim=-1)

    def __repr__(self):
        r = f'Combined({",".join(repr(energy) for energy in self.energies)})'
        if not all(self.weights == 1):
            sw = [f'{w.item():.1E}' for w in self.weights]
            r += f'({",".join(sw)})'
        return r

    @property
    def dim(self):
        return self._dim

    def to(self, device: torch.device):
        for energy in self.energies:
            energy.to(device)
        return self

    def cpu(self):
        for energy in self.energies:
            energy.cpu()
        return self
