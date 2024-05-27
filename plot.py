import sys
import os
import numpy as np
import statsmodels.api as sm
import utils
import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
if sys.platform == 'linux':
    plt.rcParams['backend'] = 'Agg'

plt.style.use('tableau-colorblind10')

plt.rcParams.update({
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

if config.LATEX_INSTALLED:
    plt.rcParams.update({
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'''\renewcommand{\rmdefault}{ptm}
                                   \usepackage{mathtools}
                                   \usepackage{amsfonts}''',
        'font.family': 'serif'})

FIG_SIZE = (3.8, 2.4)
FIG_SIZE_ONE_THIRD = (2., 1.44)  # should actually be (2., 1.5)
FIG_SIZE_TWO_THIRDS = (3.5, 1.5)  # should actually be (4., 1.5)


def save_fig(fig, path, **kwargs):
    fig.tight_layout()
    if 'wspace' in kwargs:
        fig.subplots_adjust(wspace=kwargs['wspace'])
    if 'hspace' in kwargs:
        fig.subplots_adjust(hspace=kwargs['hspace'])
    pad_inches = kwargs.pop('pad_inches', 0.01)
    fig.savefig(path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=pad_inches)
    fig.savefig(utils.next_file_path(path), format='pdf', bbox_inches='tight', dpi=300, pad_inches=pad_inches)


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


def calc_acf(data, nlags, power=1.):
    dim = data.ndim
    if dim == 1:
        data = data[None, :]
    acfs = np.stack([sm.tsa.stattools.acf(d ** power, nlags=nlags, adjusted=True)[1:] for d in data])
    if dim == 1:
        acfs = acfs.squeeze(0)
    return acfs


def compare_with_true(true_data, true_data_validation, gen_data, labels, out_dir):
    assert len(gen_data) == len(labels)
    fig_size = (5.7, 3.5)
    nlags = 10
    nlags2 = 20

    acf_true = calc_acf(true_data, nlags)
    acfs_true_val = calc_acf(true_data_validation, nlags)
    acfs = [calc_acf(data, nlags) for data in gen_data]

    acf2_true = calc_acf(true_data, nlags2, power=2.)
    acfs2_true_val = calc_acf(true_data_validation, nlags2, power=2.)
    acfs2 = [calc_acf(data, nlags2, power=2.) for data in gen_data]

    lags = np.arange(1, nlags + 1)
    lags2 = np.arange(1, nlags2 + 1)
    alpha = 0.5

    fig, axs = plt.subplots(3, len(gen_data), figsize=fig_size, sharex='row', sharey='row')
    for i in range(len(gen_data)):
        axs[0, i].plot(lags, acf_true, 'k', label='True', linewidth=1)
        for acf_val in acfs_true_val:
            axs[0, i].plot(lags, acf_val, 'k', linewidth=0.5, alpha=0.5)
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

        axs[1, i].plot(lags2, acf2_true, 'k', label='True', linewidth=1)
        for acf2_val in acfs2_true_val:
            axs[1, i].plot(lags2, acf2_val, 'k', linewidth=0.5, alpha=0.75)
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
        axs[2, i].hist(np.stack([true_data, *true_data_validation], axis=0).reshape(-1), bins=bins, color='k', density=True, histtype='step', alpha=0.5)
        axs[2, i].hist(gen_data[i].reshape(-1), bins=bins, density=True, label=labels[i], color=f'C{i}', edgecolor=f'C{i}', linewidth=1)
        axs[2, i].set_xlim(bins[0], bins[-1])
        x = np.linspace(bins[0], bins[-1], 10000)
        axs[2, i].plot(x, (2 * np.pi) ** -0.5 * np.exp(-0.5 * x ** 2), '--', color='gray', alpha=0.7, linewidth=1)
        axs[2, i].set_xlabel(labels[i])
        axs[2, i].set_xlim([-5, 5])
    axs[0, 0].set_ylabel('ACF')
    axs[1, 0].set_ylabel('ACF Sq')
    axs[2, 0].set_ylabel('Marginal')
    save_fig(fig, os.path.join(out_dir, 'acf-hist-combined.pdf'), wspace=0.1)


def only_entropies(entropies, labels, out_dir):
    fig, ax = plt.subplots(1, figsize=FIG_SIZE)
    for i, (entropy, label) in enumerate(zip(entropies, labels)):
        ax.plot(entropy, f'C{i}', label=label)
    ax.set_xlabel('Gradient step')
    ax.set_ylabel('Entropy')
    ax.legend()
    save_fig(fig, os.path.join(out_dir, 'entropies.pdf'))

    fig, ax0 = plt.subplots(1, figsize=FIG_SIZE)
    ax1 = ax0.twinx()
    for i, (ax, entropy, label) in enumerate(zip((ax0, ax1), entropies, labels)):
        ax.semilogx(entropy, f'C{i}', label=label)
        ax.set_ylabel(f'Entropy {label}')
    ax0.set_xlabel('Gradient step')
    save_fig(fig, os.path.join(out_dir, 'entropies-sep-axis.pdf'))


def compare_timeseries_with_true(true_data_validation, generated_data_list, rng, labels, out_dir):
    nice_labels = {'SP500': r'S\&P 500'}
    slice_len = 128
    fig, axs = plt.subplots(2, 2, figsize=(5.7, 1.5), sharey='all')
    axs = axs.flatten()
    data_list = [rng.choice(true_data_validation)] + [rng.choice(gen_data) for gen_data in generated_data_list]
    for ax, data, label, color in zip(axs, data_list, labels, ['k', 'C0', 'C1', 'C2']):
        i = rng.integers(data.shape[0] - slice_len)
        ax.plot(data[i:i+slice_len], color, linewidth=1)
        ax.set_xlabel(nice_labels[label] if label in nice_labels else label)
        ax.set_xlim(0, slice_len - 1)
        ax.set_xticks([])
    ylim = max(np.abs(axs[0].get_ylim()))
    axs[0].set_ylim(-ylim, ylim)
    axs[0].set_yticks([])
    save_fig(fig, os.path.join(out_dir, 'timeseries-comparison.pdf'))


def inner_loss_though_descent(inner_loss, out_dir, suffix='', color='k'):
    fig, ax = plt.subplots(1, figsize=FIG_SIZE)
    ax.loglog(np.arange(1, len(inner_loss)), inner_loss[1:], color)
    ax.set_xlabel('Gradient step')
    ax.set_ylabel('Loss')
    ax.set_xlim(1, len(inner_loss) - 1)
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
    fig, ax = plt.subplots(1, figsize=(5.8, 1.5))
    for i, (kl_losses, bs) in enumerate(zip(kl_losses_per_bs, batch_sizes)):
        label = rf'$N = {bs}$' if i == 0 else rf'${bs}$'  # r'$\phantom{N = }' + str(bs) + r'$'
        ax.plot(kl_losses[:, 0], label=label)
        ax.set_xlim(0, kl_losses.shape[0] - 1)
    ax.set_xlabel(r'Gradient step $t$')
    ax.set_ylabel(r'$\mathcal{D}_{\text{KL}}(q_t \! \parallel \! p)$')
    # ax.legend(title=r'MF batch size $N$', loc='upper left', bbox_to_anchor=(1.05, 1), labelspacing=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), labelspacing=0.2, alignment='right', markerfirst=False)
    save_fig(fig, os.path.join(out_dir, 'mf-batch-size.pdf'))


def compare_mf_with_regular(kl_losses_list, labels, out_dir):
    fig, ax = plt.subplots(2, 1, figsize=(1.55, 2), sharex='all')
    for i, (kl_losses, label) in enumerate(zip(kl_losses_list, labels)):
        ax[0].plot(kl_losses[:, 0], f'C{i}', label=label, linewidth=1)
        ax[1].plot(-kl_losses[:, 1], f'C{i}', label=label, linewidth=1)
        ax[1].plot(kl_losses[:, 2], f'C{i}--', linewidth=1)
    ax[1].set_xlim(0, kl_losses_list[0].shape[0] - 1)
    # ax[0].set_yticks([])
    ax[1].set_yticks([])
    # ax[0].set_ylabel('Rev KL div')
    ax[1].set_xlabel(r'Gradient step $t$')
    # ax[0].legend()
    save_fig(fig, os.path.join(out_dir, 'kl-div.pdf'))
    ax[0].legend(loc='upper center', labelspacing=0.2, fontsize=6)
    save_fig(fig, os.path.join(out_dir, 'kl-div-w-legend.pdf'))


def kl_and_parts(kl_losses, out_dir, color='C1', suffix=''):
    fig, ax = plt.subplots(1, 2, figsize=FIG_SIZE_ONE_THIRD, sharex='all')
    ax[0].plot(kl_losses[:, 0], f'{color}')
    ax[0].set_xlabel('Gradient step')
    ax[0].set_ylabel('Rev KL div')
    ax[1].plot(-kl_losses[:, 1], f'{color}', label='negative entropy')
    ax[1].plot(kl_losses[:, 2], f'{color}--', label='log-likelihood')
    ax[1].set_xlabel('Gradient step')
    ax[1].legend()
    save_fig(fig, os.path.join(out_dir, f'kl-div-and-parts{suffix}.pdf'))


def kl_and_parts_in_same(kl_losses, out_dir, color='C1', suffix='', figsize=FIG_SIZE_TWO_THIRDS, adjust=0):
    fig, ax0 = plt.subplots(1, figsize=figsize)
    # ax0.plot(kl_losses[:, 0], f'{color}')
    # ax0.set_xlabel('Gradient step')
    # ax0.set_ylabel('Rev KL div')
    ax0.plot(-kl_losses[:, 1], f'{color}', label=r'$-H(q_t)$')
    ax0.plot(kl_losses[:, 2], f'{color}--', label=r'$\mathbb{E}_{q_t}[\log p]$')
    ax0.set_xlabel(r'Gradient step $t$')
    ax0.set_xlim(0, kl_losses.shape[0] - 1)
    ax0.set_ylabel(r'$-H$ and $\mathbb{E}[\log p]$')
    ax1 = ax0.twinx()
    ax1.plot(kl_losses[:, 0], f'{color}:', label=r'$\mathcal{D}_{\text{KL}}(q_t \! \parallel \! p)$')
    ax1.set_ylabel(r'$\mathcal{D}_{\text{KL}}$', labelpad=adjust)
    ax0.legend(loc='upper left', labelspacing=0.2, handletextpad=0.4)
    ax1.legend(loc='lower right', labelspacing=0.2, handletextpad=0.4)
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    save_fig(fig, os.path.join(out_dir, f'kl-div-and-parts-in-same{suffix}.pdf'), pad_inches=0.05)


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
    x_min, x_max = max(x_min - 0.2 * (x_max - x_min), -0.07), min(x_max + 0.2 * (x_max - x_min), 0.17)
    y_min, y_max = np.quantile(np.concatenate([energy[:N_min, 1] for energy in energies]), [0.01, 0.99])
    y_min, y_max = y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)

    x, y = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    zs = np.array([approx(np.stack([x, y], axis=-1)) for approx in approximations])
    heights = [approx.normalizer for approx in approximations]

    fig, axs = plt.subplots(1, figsize=FIG_SIZE_ONE_THIRD)
    for i, label in enumerate(labels):
        height = heights[i]
        # axs.contourf(x, y, zs[i], cmap=cmap, alpha=.5)
        axs.contour(x, y, zs[i], colors=f'C{i}', alpha=1., levels=np.arange(1, levels + 1) * height / (levels + 1))
        axs.plot([x_min - 1, x_min], [y_min - 1, y_min - 1], label=label)
        # if i == 1:
        #     axs.scatter(energies[i][:, 0], energies[i][:, 1], s=1, c=color, label=label)
    axs.set_xlabel(r'$\phi_1$', math_fontfamily='cm')
    axs.set_ylabel(r'$\phi_2$')
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)
    axs.legend(loc='upper left', handlelength=.75, labelspacing=0.2, handletextpad=0.4)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'energy-pushforward-2d{suffix}.pdf'))
