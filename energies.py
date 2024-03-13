from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import scipy
from kymatio.scattering1d.filter_bank import scattering_filter_factory
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
    def __init__(self, logT, j1, j2):
        assert j1 is not None and j2 is not None
        self.j1 = j1
        self.j2 = j2
        self.logT = logT
        self.phi, self.psi1, self.psi2 = self.get_filters(logT, J=max(j1, j2, 8), Q=1)
        self.psi1 = self.psi1[:j1]
        self.psi2 = self.psi2[:j2]
        self.phi = torch.from_numpy(self.phi)
        self.psi1 = torch.from_numpy(self.psi1)
        self.psi2 = torch.from_numpy(self.psi2)
        self._dim = None

    @staticmethod
    def get_filters(logT, J, Q):
        phi_f, psi1_f, psi2_f = scattering_filter_factory(2 ** logT, J, (Q, 1), 2 ** J)
        phi_f = phi_f['levels'][0]
        psi1_f = np.stack([psi['levels'][0] for psi in psi1_f])
        psi2_f = np.stack([psi['levels'][0] for psi in psi2_f])
        return phi_f.astype(np.float32), psi1_f.astype(np.float32), psi2_f.astype(np.float32)

    @staticmethod
    def convolve(signal, fourier_filter):
        if utils.is_torch_else_numpy(signal):
            fourier_signal = torch.fft.fft(signal, dim=-1)
            if signal.ndim <= 1 or fourier_filter.ndim <= 1:
                fourier_conv = fourier_signal * fourier_filter
            else:
                fourier_conv = torch.einsum('...t,jt->...jt', fourier_signal, fourier_filter)
            return torch.fft.ifft(fourier_conv)
        else:
            fourier_signal = scipy.fft.fft(signal, axis=-1)  # Uses scipy due to https://github.com/numpy/numpy/issues/17801
            if signal.ndim <= 1 or fourier_filter.ndim <= 1:
                fourier_conv = fourier_signal * fourier_filter
            else:
                fourier_conv = np.einsum('...t,jt->...jt', fourier_signal, fourier_filter)
            return scipy.fft.ifft(fourier_conv)

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
            self._dim = self(torch.zeros(2 ** self.logT, device=self.device)).shape[0]
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
                                                         False, 1).mean(0, keepdims=True)
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
