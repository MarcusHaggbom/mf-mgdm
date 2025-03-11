import os
import numpy as np
import scipy
import pandas as pd
import torch
import statsmodels.api as sm
import arch
from abc import ABC, abstractmethod
import utils
import re
from tqdm import trange
import numba


def float_f(x: float) -> str:
    s = f'{x:.4f}'
    return s.rstrip('0').rstrip('.') if '.' in s else s


class Model(ABC):
    def __init__(self, seed: (int, int)):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.non_negative = False

    @abstractmethod
    def __repr__(self):
        pass

    @property
    def seed_f(self) -> str:
        return f'{self.seed}'.replace(' ', '')

    @abstractmethod
    def generate_sample(self, T, N=None):
        pass

    @abstractmethod
    def loglikelihood(self, x):
        pass

    @property
    @abstractmethod
    def mean(self):
        pass

    @property
    @abstractmethod
    def std(self):
        pass

    @staticmethod
    def formatted(x: np.ndarray | float) -> torch.Tensor:
        if isinstance(x, float):
            return torch.tensor(x).to(torch.float32)
        return torch.from_numpy(x.astype(np.float32))

    @abstractmethod
    def clone(self):
        pass


class Gaussian(Model):
    def __init__(self, seed: (int, int), mean: float = 0., std: float = 1.):
        super().__init__(seed)
        self._mean = mean
        self._std = std

    def __repr__(self):
        return f'{self.__class__.__name__}({self.mean:.2g},{self.std:.2g})-{self.seed_f}'

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        return self.formatted(self.rng.standard_normal(shape)) * self.std + self.mean

    def loglikelihood(self, x):
        return -0.5 * (((x - self.mean) ** 2 / self.std ** 2).sum(-1) + x.shape[-1] * np.log(2. * np.pi * self.std ** 2))

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def clone(self):
        return Gaussian(self.seed, self._mean, self._std)


class Gaussian2D(Gaussian):
    def generate_sample(self, T, N=None):
        shape = (T, T) if N is None else (N, T, T)
        return self.formatted(self.rng.standard_normal(shape)) * self.std + self.mean

    def loglikelihood(self, x):
        return -0.5 * (((x - self.mean) ** 2 / self.std ** 2).sum((-1, -2)) + x.shape[-1] * x.shape[-2] * np.log(2. * np.pi * self.std ** 2))


class StudentT(Model):
    def __init__(self, seed: (int, int), df: float = 6., mean: float = 0., std: float = 1.):
        super().__init__(seed)
        self._df = df
        self._mean = mean
        self._std = std

    def __repr__(self):
        return f'{self.__class__.__name__}({self._df},{self.mean},{self.std})-{self.seed_f}'

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        return self.formatted(self.rng.standard_t(self._df, shape)) * ((self._df - 2) / self._df) ** 0.5 * self.std + self.mean

    def loglikelihood(self, x):
        raise NotImplementedError()  # TODO

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def clone(self):
        return StudentT(self.seed, self._df, self._mean, self._std)


class GaussianTruncated(Model):
    def __init__(self, seed: (int, int), trunc_mean: float = 0., trunc_std: float = 1.,
                 a: float = 0., b: float = float('inf')):
        assert a < b
        super().__init__(seed)
        self.mu, self.sigma = self.solve_for_mu_and_sigma(trunc_mean, trunc_std, a, b)
        self.a = a
        self.b = b
        self._trunc_mean = trunc_mean
        self._trunc_std = trunc_std

        assert np.allclose(self.mean, trunc_mean)
        assert np.allclose(self.std, trunc_std)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.mu:.2f},{self.sigma:.2f},{self.a},{self.b})-{self.seed_f}'

    @classmethod
    def solve_for_mu_and_sigma(cls, trunc_mean, trunc_std, a, b):
        def eqns(_mu_and_sigma):
            _mu, _sigma = _mu_and_sigma
            return cls._mean(_mu, _sigma, a, b) - trunc_mean, cls._std(_mu, _sigma, a, b) - trunc_std

        res = scipy.optimize.root(eqns, np.r_[trunc_mean, trunc_std])
        assert res.success
        return res.x

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        s = self.rng.standard_normal(shape) * self.sigma + self.mu
        while True:
            idx = (s <= self.a) + (s >= self.b)
            count = idx.sum()
            if count == 0:
                break
            s[idx] = self.rng.standard_normal(count) * self.sigma + self.mu
        return self.formatted(s)

    @staticmethod
    def normal_cdf(x):
        erf = torch.special.erf if utils.is_torch_else_numpy(x) else scipy.special.erf
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    @staticmethod
    def normal_pdf(x):
        exp = torch.exp if utils.is_torch_else_numpy(x) else np.exp
        return exp(-0.5 * x ** 2) / (2 * np.pi) ** 0.5

    def loglikelihood(self, x):
        return -0.5 * (((x - self.mu) / self.sigma) ** 2).sum(-1) - x.shape[-1] * (0.5 * np.log(2. * np.pi) + np.log(self.sigma) + np.log(self.Z))

    @staticmethod
    def _alpha(mu, sigma, a):
        return (a - mu) / sigma

    @staticmethod
    def _beta(mu, sigma, b):
        return (b - mu) / sigma

    @classmethod
    def _Z(cls, mu, sigma, a, b):
        alpha = cls._alpha(mu, sigma, a)
        beta = cls._beta(mu, sigma, b)
        return cls.normal_cdf(beta) - cls.normal_cdf(alpha)

    @property
    def Z(self):
        return self._Z(self.mu, self.sigma, self.a, self.b)

    @classmethod
    def _mean(cls, mu, sigma, a, b):
        alpha = cls._alpha(mu, sigma, a)
        beta = cls._beta(mu, sigma, b)
        Z = cls._Z(mu, sigma, a, b)
        return mu + sigma * (cls.normal_pdf(alpha) - cls.normal_pdf(beta)) / Z

    @classmethod
    def _std(cls, mu, sigma, a, b):
        alpha = cls._alpha(mu, sigma, a)
        beta = cls._beta(mu, sigma, b)
        Z = cls._Z(mu, sigma, a, b)
        beta_times_phi_beta = 0. if np.isposinf(b) else beta * cls.normal_pdf(beta)
        alpha_times_phi_beta = 0. if np.isneginf(a) else alpha * cls.normal_pdf(alpha)
        return (1. - (beta_times_phi_beta - alpha_times_phi_beta) / Z
                - (cls.normal_pdf(alpha) - cls.normal_pdf(beta)) ** 2 / Z ** 2) ** 0.5 * sigma

    @property
    def mean(self):
        return self._mean(self.mu, self.sigma, self.a, self.b)

    @property
    def std(self):
        return self._std(self.mu, self.sigma, self.a, self.b)

    def clone(self):
        return GaussianTruncated(self.seed, self._trunc_mean, self._trunc_std, self.a, self.b)


class Exponential(Model):
    def __init__(self, seed: (int, int), scale: float = 1.):
        super().__init__(seed)
        self.scale = scale

    def __repr__(self):
        return f'{self.__class__.__name__}({self.scale:.2g})-{self.seed_f}'

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        return self.formatted(self.rng.exponential(self.scale, shape))

    def loglikelihood(self, x):
        return -x.sum(-1) / self.scale - x.shape[-1] * np.log(self.scale)

    @property
    def mean(self):
        return self.scale

    @property
    def std(self):
        return self.scale

    def clone(self):
        return Exponential(self.seed, self.scale)


class Gamma(Model):
    def __init__(self, seed: (int, int), alpha: float = 1., beta: float = 1.):
        super().__init__(seed)
        self.alpha = alpha
        self.beta = beta
        self._log_normalizer = scipy.special.loggamma(alpha) - alpha * np.log(beta)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.alpha:.2g},{self.beta:.2g})-{self.seed_f}'

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        return self.formatted(self.rng.gamma(self.alpha, 1. / self.beta, shape))

    def loglikelihood(self, x):
        return -x.shape[-1] * self._log_normalizer + (self.alpha - 1) * utils.lib(x).log(x).sum(-1) - self.beta * x.sum(-1)

    @property
    def mean(self):
        return self.alpha / self.beta

    @property
    def std(self):
        return self.alpha ** 0.5 / self.beta

    def clone(self):
        return Gamma(self.seed, self.alpha, self.beta)


class AR1(Model):
    def __init__(self, seed: (int, int), gamma: float = 0.1):
        super().__init__(seed)
        self.gamma = gamma
        self.sigma = np.sqrt(1 - self.gamma ** 2)
        self.arma_process = sm.tsa.ArmaProcess.from_coeffs([self.gamma], None)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.gamma})-{self.seed_f}'

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        sample = self.arma_process.generate_sample(shape, self.sigma, distrvs=self.rng.standard_normal, axis=-1)
        return self.formatted(sample)

    def loglikelihood(self, x):
        marginal_var = self.sigma ** 2 / (1 - self.gamma ** 2)
        return (- 0.5 * (np.log(2 * np.pi * marginal_var) + x[..., 0] ** 2 / marginal_var)
                - 0.5 * ((x.shape[-1] - 1) * np.log(2 * np.pi * self.sigma ** 2)
                         + ((x[..., 1:] - self.gamma * x[..., :-1]) ** 2).sum(-1) / self.sigma ** 2)
                )

    @property
    def mean(self):
        return 0.

    @property
    def std(self):
        return self.sigma / (1 - self.gamma ** 2) ** 0.5

    def clone(self):
        return AR1(self.seed, self.gamma)


class ARMA(Model):
    def __init__(self, seed: (int, int), sigma: int | str = 'standardize', arcoefs: tuple | tuple[float] = (0.1,),
                 macoefs: tuple | tuple[float] = ()):
        super().__init__(seed)
        self.arma_process = sm.tsa.ArmaProcess.from_coeffs(arcoefs, macoefs)
        assert self.arma_process.isstationary, 'Provided coeffs do not result in a stationary process'
        if sigma == 'standardize':
            self.sigma = 1. / self.arma_process.acovf(1)[0] ** 0.5
        else:
            assert isinstance(sigma, (int, float)) and sigma > 0.
            self.sigma = sigma

    def __repr__(self):
        return f'{self.__class__.__name__}({self.sigma:.2f},{self.arma_process.arcoefs},{self.arma_process.macoefs})-{self.seed_f}'

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        sample = self.arma_process.generate_sample(shape, self.sigma, distrvs=self.rng.standard_normal, axis=-1,
                                                   burnin=T // 10)
        return self.formatted(sample)

    def loglikelihood(self, x):
        # Approximation, assuming first value is stationary and preceding zero
        p = len(self.arma_process.arcoefs)
        if len(self.arma_process.macoefs) > 0:
            raise NotImplementedError()
        marginal_var = self.sigma ** 2 * self.arma_process.acovf(1)[0]
        llh = - 0.5 * (np.log(2 * np.pi * marginal_var) + x[..., 0] ** 2 / marginal_var)
        if utils.is_torch_else_numpy(x):
            x_padded = torch.concatenate([torch.zeros((*x.shape[:-1], p), device=x.device, dtype=x.dtype), x], dim=-1)
        else:
            x_padded = np.concatenate([np.zeros((*x.shape[:-1], p)), x], axis=-1)
        residuals = x_padded[..., p + 1:] * 1.  # Times 1 to make a copy
        for shift in range(1, p + 1):
            residuals -= self.arma_process.arcoefs[shift - 1] * x_padded[..., p + 1 - shift:-shift]
        llh += -0.5 * ((residuals ** 2).sum(-1) / self.sigma ** 2 + (x.shape[-1] - 1) * np.log(2 * np.pi * self.sigma ** 2))
        return llh

    @property
    def mean(self):
        return 0.

    @property
    def std(self):
        return self.sigma * self.arma_process.acovf(1)[0] ** 0.5

    @classmethod
    def from_str(cls, seed: (int, int), s: str):
        # match pattern "arma([x, y, z], [a, b, c])"
        pattern = r'arma\(\[([-\d., ]*)\], \[([-\d., ]*)\]\)'

        match = re.search(pattern, s)

        if match:
            def to_floats(_s):
                return tuple([float(num.strip()) for num in _s.split(',') if num.strip()])

            arcoefs = to_floats(match.group(1))
            macoefs = to_floats(match.group(2))

            return cls(seed, arcoefs=arcoefs, macoefs=macoefs)
        else:
            raise ValueError("Input string does not match the required format")

    def clone(self):
        return ARMA(self.seed, self.sigma, self.arma_process.arcoefs, self.arma_process.macoefs)


class CIR0(Model):  # Uses I0 (modified Bessel function of the first kind of order 0)
    """dx = kappa * (theta - x) * dt + sigma * sqrt(x) * dW"""

    def __init__(self, seed: (int, int), kappa: float = 0.5, theta: float = 1., sigma: float = 1.):
        super().__init__(seed)
        self.non_negative = True
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        _alpha = 2 * self.kappa * self.theta / self.sigma ** 2
        _beta = 2 * self.kappa / self.sigma ** 2
        self.sample_stationary_distribution = lambda _n: self.rng.gamma(_alpha, 1/_beta, _n)
        self.stationary_distribution = torch.distributions.gamma.Gamma(_alpha, _beta)
        self.stationary_distribution_cuda = None
        self.dt = 1.  # / 256

        # for sampling
        self.c = 2 * self.kappa / (self.sigma ** 2 * (1 - np.exp(-self.kappa * self.dt)))
        self.df = 4 * self.kappa * self.theta / self.sigma ** 2

    def __repr__(self):
        return f'{self.__class__.__name__}({self.kappa:.2f},{self.theta:.2f},{self.sigma:.2f})-{self.seed_f}'

    def generate_sample(self, T, N=None):
        shape = T if N is None else (N, T)
        x = np.zeros(shape)
        x[..., 0] = self.sample_stationary_distribution(N or 1)
        for t in range(1, T):
            x_prev = x[..., t - 1]
            nc = 2 * self.c * x_prev * np.exp(-self.kappa * self.dt)
            x[..., t] = self.rng.noncentral_chisquare(self.df, nc) / (2 * self.c)
        return self.formatted(x)

    def bessel(self, x):
        assert abs((2 * self.kappa * self.theta / self.sigma ** 2 - 1) - 0) < 1e-6
        if utils.is_torch_else_numpy(x):
            return torch.special.i0(x)
        else:
            return scipy.special.i0(x)

    def stationary_log_prob(self, x):
        is_torch = utils.is_torch_else_numpy(x)
        if not is_torch:
            x = torch.from_numpy(x)

        if x.device.type == 'cuda':
            if self.stationary_distribution_cuda is None:
                distr = torch.distributions.gamma.Gamma(self.stationary_distribution.concentration.to(x.device),
                                                        self.stationary_distribution.rate.to(x.device))
                self.stationary_distribution_cuda = distr
            res = self.stationary_distribution_cuda.log_prob(x)
        else:
            res = self.stationary_distribution.log_prob(x)

        if not is_torch:
            res = res.numpy()

        return res

    def loglikelihood(self, x):
        llh = self.stationary_log_prob(x[..., 0])
        q = 2 * self.kappa * self.theta / self.sigma ** 2 - 1
        us = self.c * np.exp(-self.kappa * self.dt) * x[..., :-1]
        vs = self.c * x[..., 1:]
        log = utils.lib(x).log
        llh += (np.log(self.c) - (us + vs) + q / 2 * (log(vs) - log(us)) + log(self.bessel(2 * (us * vs) ** 0.5))).sum(-1)
        return llh

    @property
    def mean(self):
        return self.stationary_distribution.mean.item()

    @property
    def std(self):  # TODO This is incorrect if dt too small
        return self.stationary_distribution.stddev.item()

    def clone(self):
        return self.__class__(self.seed, self.kappa, self.theta, self.sigma)


class CIR0ZeroMean(Model):
    def __init__(self, seed: (int, int), kappa: float = 0.5, theta: float = 1., sigma: float = 1.):
        super().__init__(seed)
        self.cir0 = CIR0(seed, kappa, theta, sigma)

    def __repr__(self):
        return 'ZeroMean' + repr(self.cir0)

    def generate_sample(self, T, N=None):
        return self.cir0.generate_sample(T, N) - self.cir0.mean

    def loglikelihood(self, x):
        return self.cir0.loglikelihood(x + self.cir0.mean)

    @property
    def mean(self):
        return 0.

    @property
    def std(self):
        return self.cir0.std

    def clone(self):
        return CIR0ZeroMean(self.seed, self.cir0.kappa, self.cir0.theta, self.cir0.sigma)


class CIR1(CIR0):  # Uses I1 (modified Bessel function of the first kind of order 1)
    def __init__(self, seed: (int, int), kappa: float = 2**(-0.5), theta: float = 2**0.5, sigma: float = 1.):
        super().__init__(seed, kappa, theta, sigma)

    def bessel(self, x):
        assert abs((2 * self.kappa * self.theta / self.sigma ** 2 - 1) - 1) < 1e-6
        if utils.is_torch_else_numpy(x):
            return torch.special.i1(x)
        else:
            return scipy.special.i1(x)


class CIR1ZeroMean(Model):
    def __init__(self, seed: (int, int), kappa: float = 2**(-0.5), theta: float = 2**0.5, sigma: float = 1.):
        super().__init__(seed)
        self.cir1 = CIR1(seed, kappa, theta, sigma)

    def __repr__(self):
        return 'ZeroMean' + repr(self.cir1)

    def generate_sample(self, T, N=None):
        return self.cir1.generate_sample(T, N) - self.cir1.mean

    def loglikelihood(self, x):
        return self.cir1.loglikelihood(x + self.cir1.mean)

    @property
    def mean(self):
        return 0.

    @property
    def std(self):
        return self.cir1.std

    def clone(self):
        return CIR1ZeroMean(self.seed, self.cir1.kappa, self.cir1.theta, self.cir1.sigma)


class GARCH(Model):
    def __init__(self, seed: (int, int), omega: float = 0.03, alpha: float = 0.1, beta: float = 0.87):
        super().__init__(seed)
        assert alpha + beta < 1, 'Stationarity condition violated'
        self.params = pd.Series({'omega': omega, 'alpha[1]': alpha, 'beta[1]': beta})
        self.model = self.get_model(self.rng)

    @staticmethod
    def get_model(rng: np.random.Generator, data=None):
        distribution = arch.univariate.Normal(seed=rng)
        model = arch.univariate.ZeroMean(data, volatility=arch.univariate.GARCH(p=1, o=0, q=1),
                                         distribution=distribution)
        return model

    def __repr__(self):
        return (f'{self.__class__.__name__}({float_f(self.params["omega"])},'
                f'{float_f(self.params["alpha[1]"])},{float_f(self.params["beta[1]"])})-{self.seed_f}')

    def generate_sample(self, T, N=None):
        samples = np.stack([self.model.simulate(self.params, T, burn=1000).data.values for _ in range(N or 1)], axis=0)
        if N is None:
            samples = samples.squeeze(0)
        return self.formatted(samples)

    def loglikelihood(self, x, sigma2_0=1., burnin=0):
        lib = utils.lib(x)
        sigma2 = lib.zeros_like(x)
        sigma2[..., 0] = sigma2_0
        for t in range(1, sigma2.shape[-1]):
            sigma2[..., t] = self.params['omega'] + self.params['alpha[1]'] * x[..., t - 1] ** 2 + self.params['beta[1]'] * sigma2[..., t - 1]
        arr = -0.5 * (lib.log(2 * lib.pi * sigma2) + x ** 2 / sigma2)
        return arr[..., burnin:].sum(-1)

    @property
    def mean(self):
        return 0.

    @property
    def std(self):
        return (self.params['omega'] / (1 - self.params['alpha[1]'] - self.params['beta[1]'])) ** 0.5

    @classmethod
    def from_str(cls, seed: (int, int), s: str):
        # match pattern "GARCH(x, y, z)"
        pattern = r'GARCH\(([-\d., ]*)\)'

        match = re.search(pattern, s)

        if match:
            def to_floats(_s):
                return tuple([float(num.strip() or 0) for num in _s.split(',')])

            args = to_floats(match.group(1))
            return cls(seed, *args)
        else:
            raise ValueError("Input string does not match the required format")

    @classmethod
    def from_true_data(cls, seed: (int, int), true_data: np.ndarray):
        model = cls.get_model(None, true_data)
        res = model.fit(disp='off')
        assert res.optimization_result.success
        params = res.params
        # print('GARCH fitted params', params)
        return cls(seed, params['omega'], params['alpha[1]'], params['beta[1]'])

    def clone(self):
        return GARCH(self.seed, self.params['omega'], self.params['alpha[1]'], self.params['beta[1]'])


class AR1GARCH(Model):
    def __init__(self, seed: (int, int), gamma: float = 0.1, omega: float = 0.03, alpha: float = 0.1, beta: float = 0.87,
                 df: float = float('inf')):
        super().__init__(seed)
        assert alpha + beta < 1, 'Stationarity condition violated'
        assert abs(gamma) < 1, 'Stationarity condition violated'
        self.params = pd.Series({'y[1]': gamma, 'omega': omega, 'alpha[1]': alpha, 'beta[1]': beta})
        _studentsT = not np.isposinf(df)
        if _studentsT:
            self.params['nu'] = df
        self.model = self.get_model(self.rng, studentsT=_studentsT)

    @staticmethod
    def get_model(rng: np.random.Generator, data=None, studentsT=False):
        distribution = (arch.univariate.StudentsT if studentsT else arch.univariate.Normal)(seed=rng)
        model = arch.univariate.ARX(data, constant=False, lags=1, volatility=arch.univariate.GARCH(p=1, o=0, q=1),
                                    distribution=distribution)
        return model

    def __repr__(self):
        return (f'AR({self.params["y[1]"]})'
                f'-GARCH({float_f(self.params["omega"])},{float_f(self.params["alpha[1]"])},'
                f'{float_f(self.params["beta[1]"])})'
                + ('-N' if 'nu' not in self.params else f'-T({float_f(self.params["nu"])})') +
                f'-{self.seed_f}')

    def generate_sample(self, T, N=None):
        samples = np.stack([self.model.simulate(self.params, T, burn=1000).data.values for _ in range(N or 1)], axis=0)
        if N is None:
            samples = samples.squeeze(0)
        return self.formatted(samples)

    def loglikelihood(self, x, sigma2_0=1., burnin=0):
        raise NotImplementedError()

    @property
    def mean(self):
        return 0.

    @property
    def std(self):
        if 'nu' in self.params:
            raise NotImplementedError()
        vol_var = self.params['omega'] / (1 - self.params['alpha[1]'] - self.params['beta[1]'])
        return (vol_var / (1 - self.params['y[1]'] ** 2)) ** 0.5

    @classmethod
    def from_true_data(cls, seed: (int, int), true_data: np.ndarray):
        model = cls.get_model(None, true_data, True)
        res = model.fit(disp='off')
        assert res.optimization_result.success
        params = res.params
        # print('GARCH fitted params', params)
        return cls(seed, params['y[1]'], params['omega'], params['alpha[1]'], params['beta[1]'], params['nu'])

    def clone(self):
        return AR1GARCH(self.seed, self.params['y[1]'], self.params['omega'], self.params['alpha[1]'],
                        self.params['beta[1]'], self.params['nu'])


@numba.jit(nopython=True)
def ising_loop(x, N, T, steps, beta, rows, cols, thresholds):
    for i in range(steps):
        for n in range(N):
            r, c = rows[i, n], cols[i, n]
            dE = 2 * x[n, r, c] * (x[n, (r - 1) % T, c] + x[n, (r + 1) % T, c]
                                   + x[n, r, (c - 1) % T] + x[n, r, (c + 1) % T]) * beta
            if dE < 0 or thresholds[i, n] < np.exp(-dE):
                x[n, r, c] *= -1
    return x


class Ising(Model):
    """2d Ising model"""

    def __init__(self, seed: (int, int), temperature: float = 2.2, sweeps: int = 1000, use_cache: bool = True):
        super().__init__(seed)
        self.temperature = temperature
        self.sweeps = sweeps
        self.use_cache = use_cache
        self._sampled_from = False

    def __repr__(self):
        return f'{self.__class__.__name__}({self.temperature:.2g},{self.sweeps})-{self.seed_f}'

    @property
    def beta(self):
        return 1 / self.temperature  # beta = 1 / kT. J (exchange interaction) is set to 1.

    def generate_sample(self, T, N=None):
        if self.use_cache and self._sampled_from:
            raise ValueError('Sampled from cache. Cannot generate new samples')
        self._sampled_from = True
        file_dir = os.path.join('data', 'ising')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, repr(self) + f'-{T}x{T}x{N}.npy')

        if os.path.exists(file_path):
            x = np.load(file_path)
        else:
            x = self._generate_sample(T, N)
            np.save(file_path, x)
        return self.formatted(x)

    def _generate_sample(self, T, N):
        x = self.rng.choice([-1, 1], (N, T, T))

        total_steps = self.sweeps * T ** 2
        steps_per_batch = round(128 * 256 * 256 * 8 / N)
        assert steps_per_batch > 0

        for _ in trange(np.ceil(total_steps / steps_per_batch).astype(int)):
            rows = self.rng.choice(T, (steps_per_batch, N))
            cols = self.rng.choice(T, (steps_per_batch, N))
            thresholds = self.rng.uniform(size=(steps_per_batch, N))
            # t0 = time.time()
            x = ising_loop(x, N, T, steps_per_batch, self.beta, rows, cols, thresholds)
            # t1 = time.time()
            # print('Frequency:', steps_per_batch / (t1 - t0))

        return x

    @staticmethod
    def log_ising_partition_fn(beta, M, N):  # Kaufman 49 (also in Beale 96)
        K = beta
        k = np.arange(2 * N)
        gamma = np.arccosh(np.cosh(2 * K) ** 2 / np.sinh(2 * K) - np.cos(np.pi * k / N))
        gamma[0] = 2 * K + np.log(np.tanh(K))
        Y1 = np.cosh(M / 2 * gamma[1::2])
        Y2 = np.sinh(M / 2 * gamma[1::2])
        Y3 = np.cosh(M / 2 * gamma[::2])
        Y4 = np.sinh(M / 2 * gamma[::2])
        log_Z = (N * M / 2 + N - 1) * np.log(2) + N * M / 2 * np.log(np.sinh(2 * K))
        log_Z += np.log(Y1).sum() + np.log(1 + np.prod(Y2 / Y1) + np.prod(Y3 / Y1) + np.prod(Y4 / Y1))
        return log_Z

    def loglikelihood(self, x):
        log_energy = self.beta * (x * (np.roll(x, 1, axis=-1) + np.roll(x, 1, axis=-2))).sum((-1, -2))
        M, N = x.shape[-2:]
        log_Z = self.log_ising_partition_fn(self.beta, M, N)
        return log_energy - log_Z

    @property
    def mean(self):
        return 0.

    @property
    def std(self):
        return 1.

    def clone(self):
        return Ising(self.seed, self.temperature, self.sweeps, self.use_cache)


class IsingRelaxed(Ising):
    """Relaxed 2d Ising model"""

    def __init__(self, seed: (int, int), temperature: float = 2.2, sweeps: int = 1000, use_cache: bool = True, alpha: float = 1.):
        super().__init__(seed, temperature, sweeps, use_cache)
        self.alpha = alpha
        # self._adj_matrices_cache = {}
        self._cov_mat_cache = {}

    def __repr__(self):
        return f'{self.__class__.__name__}({self.temperature:.2g},{self.sweeps},{self.alpha})-{self.seed_f}'

    @property
    def beta(self):
        return 1 / self.temperature  # beta = 1 / kT. J (exchange interaction) is set to 1.

    def adj_matrix(self, M, N):
        adj_mat = torch.zeros((N * M, N * M))
        indices = torch.arange(N * M)
        adj_mat[indices, (indices + N) % (N * M)] = 1
        adj_mat[indices, (indices // N) * N + (indices + 1) % N] = 1
        adj_mat = adj_mat + adj_mat.T
        try:
            np.linalg.cholesky(adj_mat.numpy() * self.beta + self.alpha * np.eye(N * M))
        except np.linalg.LinAlgError:
            raise ValueError('Matrix is not positive definite; increase alpha')
        return adj_mat

    def covariance_matrix_inv_and_logdet(self, M, N, device):
        key = (M, N, device)
        if key not in self._cov_mat_cache:
            K = self.beta * self.adj_matrix(M, N)
            cov_mat = K + self.alpha * torch.eye(K.shape[-1], device=K.device)
            self._cov_mat_cache[key] = (torch.linalg.inv(cov_mat).to(device), torch.slogdet(cov_mat).logabsdet.item())
        return self._cov_mat_cache[key]

    def loglikelihood(self, x):
        M, N = x.shape[-2:]
        x_flat = x.reshape(*x.shape[:-2], M * N, 1)
        cov_inv, cov_logdet = self.covariance_matrix_inv_and_logdet(M, N, x.device)
        log_energy = -0.5 * (x_flat.mT @ cov_inv @ x_flat).squeeze((-1, -2))
        log_energy += torch.log(torch.cosh(x_flat)).sum((-1, -2))

        # Neural Network Renormalization group https://arxiv.org/pdf/1802.02840
        logZ = self.log_ising_partition_fn(self.beta, M, N) + 0.5 * cov_logdet - 0.5 * M * N * (np.log(2 / np.pi) - self.alpha)
        return log_energy - logZ

    def clone(self):
        return IsingRelaxed(self.seed, self.temperature, self.sweeps, self.use_cache, self.alpha)


def get_maximum_entropy_model(model: Model, seed: (int, int)) -> Model:
    if model.non_negative:
        if model.mean == model.std:
            latent_model = Exponential(seed, model.std)
        else:
            latent_model = GaussianTruncated(seed, model.mean, model.std)
    else:
        latent_model = Gaussian(seed, model.mean, model.std)
    print('Latent model:', latent_model)
    return latent_model


def _test():

    # x = ising2d(10, 256, (12, 234), 1000, 1 / 2.2)
    N = 16
    # T = 256
    # ising = Ising((12, 234), 2.4, 1024)
    T = 32
    ising = Ising((12, 234), 4., 1024)
    x = ising.generate_sample(T, N)

    import matplotlib.pyplot as plt
    for y in x[:20]:
        plt.figure()
        plt.imshow(y)
    plt.show()


def _test_ising_partition():
    import itertools

    def logZ(M, N, K):
        k = np.arange(2 * N)
        gamma = np.arccosh(np.cosh(2 * K) ** 2 / np.sinh(2 * K) - np.cos(np.pi * k / N))
        gamma[0] = 2 * K + np.log(np.tanh(K))
        Y1 = np.cosh(M / 2 * gamma[1::2])
        Y2 = np.sinh(M / 2 * gamma[1::2])
        Y3 = np.cosh(M / 2 * gamma[::2])
        Y4 = np.sinh(M / 2 * gamma[::2])
        log_free_energy = (N * M / 2 + N - 1) * np.log(2) + N * M / 2 * np.log(np.sinh(2 * K))
        log_free_energy += np.log(Y1).sum() + np.log(1 + np.prod(Y2 / Y1) + np.prod(Y3 / Y1) + np.prod(Y4 / Y1))
        # log_free_energy += np.log(np.prod(Y1) + np.prod(Y2) + np.prod(Y3) + np.prod(Y4))
        return log_free_energy

    def logZ2(M, N, K):
        def efn(x):
            return - K * (x * (np.roll(x, 1, axis=-1) + np.roll(x, 1, axis=-2))).sum((-1, -2))
        sample = np.random.choice([-1, 1], (N, M))
        C = 1 #np.exp(-efn(sample))
        s = 0
        for i, a in enumerate(itertools.product([-1, 1], repeat=N*M)):
            x = np.array(a).reshape(M, N)
            s += np.exp(-efn(x)) / C
        return np.log(s) + np.log(C)

    for n in (2, 3, 4):
        for temp in (1., 2., 3., 4., 5.):
            Z1 = logZ(n, n, 1 / temp)
            Z2 = logZ2(n, n, 1 / temp)
            print(n, temp, Z1, Z2, Z1 - Z2)


if __name__ == '__main__':
    _test()
    # _test_ising_partition()
