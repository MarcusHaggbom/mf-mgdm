import sys
import os
import numpy as np
import statsmodels.api as sm
import utils
import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
if sys.platform == 'linux':
    plt.rcParams['backend'] = 'Agg'
plt.style.use('tableau-colorblind10')
rc('font', family='serif')
rc('font', size=9)
rc('legend', fontsize=8)
if config.LATEX_INSTALLED:
    rc('text', usetex=True)


FIG_SIZE = (4, 2.5)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(utils.next_file_path(path), format='pdf', bbox_inches='tight', dpi=300)


def set_same_lims(axs):
    xlims = [ax.get_xlim() for ax in axs]
    xmin = min([xlim[0] for xlim in xlims])
    xmax = max([xlim[1] for xlim in xlims])
    ylims = [ax.get_ylim() for ax in axs]
    ymin = min([ylim[0] for ylim in ylims])
    ymax = max([ylim[1] for ylim in ylims])
    for ax in axs:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


def compare_with_true(true_data, gen_data, labels, out_dir):
    assert len(gen_data) == len(labels)
    fig_size = (4, 2.5)
    nlags = 10
    nlags2 = 20

    acfs = [np.stack([sm.tsa.stattools.acf(d, nlags=nlags, adjusted=True)[1:] for d in data])
            for data in (true_data[None, :], *gen_data)]
    acf_true = acfs[0][0]
    acfs = acfs[1:]

    acfs2 = [np.stack([sm.tsa.stattools.acf(d ** 2, nlags=nlags2, adjusted=True)[1:] for d in data])
             for data in (true_data[None, :], *gen_data)]
    acf2_true = acfs2[0][0]
    acfs2 = acfs2[1:]

    lags = np.arange(1, nlags + 1)
    lags2 = np.arange(1, nlags2 + 1)
    alpha = 0.5

    fig, axs = plt.subplots(3, len(gen_data), figsize=(8, 4), sharex='row', sharey='row')
    for i in range(len(gen_data)):
        axs[0, i].plot(lags, acf_true, 'k', label='True')
        _parts = axs[0, i].violinplot(acfs[i])
        for _pc in _parts['bodies']:
            _pc.set_edgecolor(f'C{i}')
            _pc.set_facecolor(f'C{i}')
            _pc.set_alpha(alpha)
        for _pc in ['cbars', 'cmins', 'cmaxes']:
            _parts[_pc].set_color(f'C{i}')
            _parts[_pc].set_linewidth(1)
        axs[0, i].set_xlabel('Lag')
        axs[0, i].set_xlim(lags[0]-0.35, lags[-1]+0.35)
        axs[0, i].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        # axs[0, i].set_title(labels[i])

        axs[1, i].plot(lags2, acf2_true, 'k', label='True')
        _parts = axs[1, i].violinplot(acfs2[i])
        for _pc in _parts['bodies']:
            _pc.set_edgecolor(f'C{i}')
            _pc.set_facecolor(f'C{i}')
            _pc.set_alpha(alpha)
        for _pc in ['cbars', 'cmins', 'cmaxes']:
            _parts[_pc].set_color(f'C{i}')
            _parts[_pc].set_linewidth(1)
        axs[1, i].set_xlabel('Lag')
        axs[1, i].set_xlim(lags2[0]-0.4, lags2[-1]+0.4)
        axs[1, i].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        _, bins, _ = axs[2, i].hist(true_data, bins=50, color='k', density=True, histtype='step', label='True')
        axs[2, i].hist(gen_data[i].reshape(-1), bins=bins, density=True, label=labels[i], color=f'C{i}', edgecolor=f'C{i}')
        axs[2, i].set_xlim(bins[0], bins[-1])
        x = np.linspace(bins[0], bins[-1], 10000)
        axs[2, i].plot(x, (2 * np.pi) ** -0.5 * np.exp(-0.5 * x ** 2), '--', color='gray', alpha=0.7)
        axs[2, i].set_xlabel(labels[i])
        axs[2, i].set_xlim([-5, 5])
    axs[0, 0].set_ylabel('ACF')
    axs[1, 0].set_ylabel('ACF Sq')
    axs[2, 0].set_ylabel('Marginal')
    save_fig(fig, os.path.join(out_dir, 'acf-hist-combined.pdf'))


def inner_loss_though_descent(inner_loss, out_dir, suffix='', color='k'):
    fig, ax = plt.subplots(1, figsize=FIG_SIZE)
    ax.loglog(np.arange(1, len(inner_loss)), inner_loss[1:], color)
    ax.set_xlabel('Gradient step')
    ax.set_ylabel('Loss')
    save_fig(fig, os.path.join(out_dir, f'inner-loss-though-descent{suffix}.pdf'))


def robustness(true_losses, sample_losses, noise_levels, out_dir, suffix='', color='C0'):
    fig, ax = plt.subplots(1, figsize=FIG_SIZE)
    # ax.set_xscale('log')
    ax.plot(np.log10(noise_levels), true_losses, 'k|-')
    parts = ax.violinplot(sample_losses.T, positions=np.log10(noise_levels))
    for pc in parts['bodies']:
        pc.set_edgecolor(color)
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for pc in ['cbars', 'cmins', 'cmaxes']:
        parts[pc].set_color(color)
        parts[pc].set_linewidth(1)
    ax.set_xlabel('Perturbation ($\\log_{10}$)')
    ax.set_ylabel('Loss')
    # ax.legend()
    save_fig(fig, os.path.join(out_dir, f'robustness{suffix}.pdf'))


def compare_mf_batch_size(batch_sizes, kl_losses_per_bs, out_dir):
    fig, ax = plt.subplots(1, figsize=(5, 2.5))
    for kl_losses, bs in zip(kl_losses_per_bs, batch_sizes):
        ax.plot(kl_losses[:, 0], label=f'{bs}')
    ax.set_xlabel('Gradient step')
    ax.set_ylabel('Rev KL div')
    ax.legend(title='MF batch size', loc='upper left', bbox_to_anchor=(1.05, 1))
    save_fig(fig, os.path.join(out_dir, 'mf-batch-size.pdf'))


def compare_mf_with_regular(kl_losses_list, labels, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.5), sharex='all')
    for i, (kl_losses, label) in enumerate(zip(kl_losses_list, labels)):
        ax[0].plot(kl_losses[:, 0], f'C{i}', label=label)
        ax[1].plot(-kl_losses[:, 1], f'C{i}', label=label)
        ax[1].plot(kl_losses[:, 2], f'C{i}--')
    ax[0].set_xlabel('Gradient step')
    ax[0].set_ylabel('Rev KL div')
    ax[1].set_xlabel('Gradient step')
    ax[0].legend()
    save_fig(fig, os.path.join(out_dir, 'kl-div.pdf'))


def kl_and_parts(kl_losses, out_dir, color='C1', suffix=''):
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.5), sharex='all')
    ax[0].plot(kl_losses[:, 0], f'{color}')
    ax[0].set_xlabel('Gradient step')
    ax[0].set_ylabel('Rev KL div')
    ax[1].plot(-kl_losses[:, 1], f'{color}', label='negative entropy')
    ax[1].plot(kl_losses[:, 2], f'{color}--', label='log-likelihood')
    ax[1].set_xlabel('Gradient step')
    ax[1].legend()
    save_fig(fig, os.path.join(out_dir, f'kl-div-and-parts{suffix}.pdf'))


def kl_and_parts_in_same(kl_losses, out_dir, color='C1', suffix=''):
    fig, ax0 = plt.subplots(1, figsize=(4, 2.5))
    # ax0.plot(kl_losses[:, 0], f'{color}')
    # ax0.set_xlabel('Gradient step')
    # ax0.set_ylabel('Rev KL div')
    ax0.plot(-kl_losses[:, 1], f'{color}', label='negative entropy (lhs)')
    ax0.plot(kl_losses[:, 2], f'{color}--', label='log-likelihood (lhs)')
    ax0.set_xlabel('Gradient step')
    ax1 = ax0.twinx()
    ax1.plot(kl_losses[:, 0], f'{color}:', label='rev KL div (rhs)')
    ax0.legend(loc='upper left')
    ax1.legend(loc='lower right')
    save_fig(fig, os.path.join(out_dir, f'kl-div-and-parts-in-same{suffix}.pdf'))


def energy_pushforward_2d(energies, labels, out_dir, suffix=''):
    class Gaussian:
        def __init__(self, mu=0., cov=1.):
            self.mu = mu
            self.cov = cov

        def __call__(self, x):
            dist = (x - self.mu)[..., None, :] @ np.linalg.solve(self.cov, (x - self.mu)[..., None])
            dist = dist.squeeze()
            return self.normalizer * np.exp(-0.5 * dist)

        @property
        def normalizer(self):
            return (2 * np.pi * np.linalg.det(self.cov)) ** (-0.5)

        @classmethod
        def from_data(cls, data):
            assert data.ndim == 2
            mean = data.mean(0)
            cov = np.cov(data.T)
            return cls(mean, cov)

    levels = 5

    approximations = [Gaussian.from_data(energy) for energy in energies]
    N_min = min([energy.shape[0] for energy in energies])
    x_min, x_max = np.quantile(np.concatenate([energy[:N_min, 0] for energy in energies]), [0.01, 0.99])
    x_min, x_max = x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min)
    y_min, y_max = np.quantile(np.concatenate([energy[:N_min, 1] for energy in energies]), [0.01, 0.99])
    y_min, y_max = y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)

    x, y = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    zs = np.array([approx(np.stack([x, y], axis=-1)) for approx in approximations])
    heights = [approx.normalizer for approx in approximations]

    fig, axs = plt.subplots(1, figsize=FIG_SIZE)
    for i, label in enumerate(labels):
        height = heights[i]
        # axs.contourf(x, y, zs[i], cmap=cmap, alpha=.5)
        axs.contour(x, y, zs[i], colors=f'C{i}', alpha=1., levels=np.arange(1, levels + 1) * height / (levels + 1))
        axs.plot([x_min - 1, x_min], [y_min - 1, y_min - 1], label=label)
        # if i == 1:
        #     axs.scatter(energies[i][:, 0], energies[i][:, 1], s=1, c=color, label=label)
    axs.set_xlabel(r'$\phi_1$')
    axs.set_ylabel(r'$\phi_2$')
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)
    axs.legend()
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'energy-pushforward-2d{suffix}.pdf'))
