import sys
import os
import numpy as np
import statsmodels.api as sm
import utils
import tqdm
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


def only_entropies(entropies, labels, out_dir, dim, final_fn_applications=0):
    steps = np.arange(len(entropies[0]))
    fig_size = (3.5, 2.2)
    fig0, ax0 = plt.subplots(1, figsize=fig_size)
    if final_fn_applications:
        fig1, ax1 = plt.subplots(1, figsize=fig_size)
        fig2, ax2 = plt.subplots(1, figsize=fig_size)
    for i, (entropy, label) in enumerate(zip(entropies, labels)):
        ax0.plot(steps, entropy / dim, f'C{i}', label=label)
        ax0.set_xlim(*steps[[0, -1]])
        if final_fn_applications:
            ax1.plot(steps[:-final_fn_applications], entropy[:-final_fn_applications] / dim, f'C{i}', label=label)
            ax1.set_xlim(*steps[:-final_fn_applications][[0, -1]])
            ax2.plot(steps[-(final_fn_applications + 2):], entropy[-(final_fn_applications + 2):] / dim, '-o', color=f'C{i}', label=label)
            ax2.set_xlim(*steps[-(final_fn_applications + 2):][[0, -1]])
            ax2.set_xticks(steps[-(final_fn_applications + 2):])
    for ax in [ax0] + ([ax1, ax2] if final_fn_applications else []):
        ax.set_xlabel('Gradient step $t$')
        ax.set_ylabel('Entropy (nats/dim)')
        ax.legend()
    save_fig(fig0, os.path.join(out_dir, 'entropies.pdf'))
    if final_fn_applications:
        save_fig(fig1, os.path.join(out_dir, 'entropies-ex-last.pdf'))
        save_fig(fig2, os.path.join(out_dir, 'entropies-only-last.pdf'))


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


def inner_loss_though_descent(inner_loss, out_dir, labels=None, suffix='', color=None):
    plot_kwargs = {}
    if labels is not None:
        plot_kwargs['label'] = labels
    if color is not None:
        plot_kwargs['color'] = color

    fig, ax = plt.subplots(1, figsize=FIG_SIZE)
    ax.loglog(np.arange(len(inner_loss)) + 1, inner_loss, **plot_kwargs)
    ax.set_xlabel('Gradient step')
    ax.set_ylabel('Loss')
    ax.set_xlim(1, len(inner_loss))
    if labels is not None:
        ax.legend()
    save_fig(fig, os.path.join(out_dir, f'loss-though-descent-loglog{suffix}.pdf'))

    fig, ax = plt.subplots(1, figsize=FIG_SIZE)
    ax.semilogy(np.arange(len(inner_loss)), inner_loss, **plot_kwargs)
    ax.set_xlabel('Gradient step')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, len(inner_loss) - 1)
    if labels is not None:
        ax.legend()
    save_fig(fig, os.path.join(out_dir, f'loss-though-descent-logy{suffix}.pdf'))


def inner_losses_though_descent(inner_loss, out_dir, labels=None, suffix='', color=None):
    plot_kwargs = {}
    if color is not None:
        plot_kwargs['color'] = color

    fig, ax0 = plt.subplots(1, figsize=(3.8, 2.4))
    ax1 = ax0.twinx() if inner_loss.shape[1] > 1 else None

    ax0.loglog(np.arange(inner_loss.shape[0]) + 1, inner_loss[:, 0], label=labels[0] if labels else 'Loss', **plot_kwargs)
    ax0.set_xlabel('Gradient step')
    ax0.set_ylabel('Loss')
    ax0.set_xlim(1, inner_loss.shape[0])

    if ax1:
        ax1.semilogy(np.arange(inner_loss.shape[0]) + 1, inner_loss[:, 1], '--', color='C1', label=labels[1] if labels else 'Second Loss')
        ax1.set_ylabel(labels[1] if labels else 'Second Loss')

    if labels:
        ax0.legend(loc='upper left')
        if ax1:
            ax1.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'loss-though-descent-loglog{suffix}.pdf'))

    fig, ax0 = plt.subplots(1, figsize=(3.8, 2.4))
    ax1 = ax0.twinx() if inner_loss.shape[1] > 1 else None

    ax0.semilogy(np.arange(inner_loss.shape[0]), inner_loss[:, 0], label=labels[0] if labels else 'Loss', **plot_kwargs)
    ax0.set_xlabel('Gradient step')
    ax0.set_ylabel('Loss')
    ax0.set_xlim(0, inner_loss.shape[0] - 1)

    if ax1:
        ax1.semilogy(np.arange(inner_loss.shape[0]), inner_loss[:, 1], '--', color='C1', label=labels[1] if labels else 'Second Loss')
        ax1.set_ylabel(labels[1] if labels else 'Second Loss')

    if labels:
        ax0.legend(loc='upper left')
        if ax1:
            ax1.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'loss-though-descent-logy{suffix}.pdf'))


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
    fig, ax = plt.subplots(1, figsize=(3.5, 1.5))
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
    fig, ax = plt.subplots(2, 1, figsize=(1.8, 2), sharex='all')
    for i, (kl_losses, label) in enumerate(zip(kl_losses_list, labels)):
        ax[0].plot(kl_losses[:, 0], f'C{i}', label=label, linewidth=1)
        ax[1].plot(-kl_losses[:, 1], f'C{i}', label=label, linewidth=1)
        ax[1].plot(kl_losses[:, 2], f'C{i}--', linewidth=1)
    ax[1].set_xlim(0, kl_losses_list[0].shape[0] - 1)
    # ax[0].set_yticks([])
    # ax[1].set_yticks([])
    # ax[0].set_ylabel('Rev KL div')
    ax[1].set_xlabel(r'Gradient step $t$')
    # ax[0].legend()
    save_fig(fig, os.path.join(out_dir, 'kl-div.pdf'))
    ax[0].legend(loc='upper center', labelspacing=0.2, fontsize=6)
    save_fig(fig, os.path.join(out_dir, 'kl-div-w-legend.pdf'))


def compare_mf_with_regular_entr(kl_losses_list, labels, out_dir, dim, final_fn_applications):
    reg_entr = kl_losses_list[0][:, 1] / dim
    mf_entr = kl_losses_list[1][:, 1] / dim
    n_points = len(reg_entr)
    fig_size = (3.45, 1.5)

    fig, ax = plt.subplots(1, figsize=fig_size, sharex='all')
    ax.plot(reg_entr[:-final_fn_applications], label=labels[0], linewidth=1)
    ax.plot(mf_entr[:-final_fn_applications], label=labels[1], linewidth=1)
    ax.set_xlim(0, n_points - (1 + final_fn_applications))
    ax.set_ylabel('Entropy (nats/dim)')
    ax.set_xlabel(r'Gradient step $t$')
    ax.legend()
    save_fig(fig, os.path.join(out_dir, 'entropy-ex-last.pdf'))

    fig, ax = plt.subplots(1, figsize=fig_size, sharex='all')
    t_window = final_fn_applications + 2
    t = np.arange(n_points - t_window, n_points)
    ax.plot(t, reg_entr[-t_window:], '-o', label=labels[0], linewidth=1)
    ax.plot(t, mf_entr[-t_window:], '-o', label=labels[1], linewidth=1)
    ax.set_xlim(n_points - t_window, n_points-1)
    ax.set_xticks(t)
    ax.set_xlabel(r'Gradient step $t$')
    ax.set_ylabel('Entropy (nats/dim)')
    ax.legend(loc='lower left', labelspacing=0.2, fontsize=6)
    save_fig(fig, os.path.join(out_dir, 'entropy-last.pdf'))
    ax1 = ax.twinx()
    ax1.plot(t, mf_entr[-t_window:] - reg_entr[-t_window:], '--o', color='C2', label='Difference', linewidth=1)
    ax1.set_ylabel('Difference')
    ax1.legend(loc='upper right', labelspacing=0.2, fontsize=6)
    save_fig(fig, os.path.join(out_dir, 'entropy-last-incl-diff.pdf'))


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
    ax1.yaxis.set_major_locator(MultipleLocator(.002))
    save_fig(fig, os.path.join(out_dir, f'kl-div-and-parts-in-same{suffix}.pdf'), pad_inches=0.05)


def energy_pushforward_2d(energies, labels, out_dir, suffix='', colors=None, figsize=FIG_SIZE_ONE_THIRD,
                          legend_inside=True, lims=None, mcmc_path=None, epsilons=()):
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

    fig, axs = plt.subplots(1, figsize=figsize)
    for i, label in enumerate(labels):
        height = heights[i]
        # axs.contourf(x, y, zs[i], cmap=cmap, alpha=.5)
        cont_kwargs = {'colors': [colors[i] if colors is not None else f'C{i}']}
        plot_kwargs = {} if colors is None else {'color': colors[i]}

        axs.contour(x, y, zs[i], alpha=1., levels=np.arange(1, levels + 1) * height / (levels + 1), **cont_kwargs)
        axs.plot([x_min - 1, x_min], [y_min - 1, y_min - 1], label=label, **plot_kwargs)

        # if i == 1:
        #     axs.scatter(energies[i][:, 0], energies[i][:, 1], s=1, c=color, label=label)

    if mcmc_path is not None:
        axs.plot(mcmc_path[:, 0], mcmc_path[:, 1], 'k', alpha=0.8, linewidth=0.5, zorder=9)
        axs.scatter(mcmc_path[:, 0], mcmc_path[:, 1], s=12, alpha=0.7, c=range(mcmc_path.shape[0]), label='MCMC path', zorder=10, edgecolors='none')

    target_energy = energies[0].mean(0)
    phi = np.linspace(0, 2 * np.pi, 1000)
    for epsilon in epsilons:
        plt.plot(target_energy[0] + epsilon * np.cos(phi), target_energy[1] + epsilon * np.sin(phi), 'k--')

    axs.set_xlabel(r'$\phi_1$', math_fontfamily='cm')
    axs.set_ylabel(r'$\phi_2$')
    if lims is not None:
        x_min, x_max, y_min, y_max = lims
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)
    legend_kwargs = {} if legend_inside else {'bbox_to_anchor': (1.05, 1.05)}
    axs.legend(loc='upper left', handlelength=.75, labelspacing=0.2, handletextpad=0.4, **legend_kwargs).set_zorder(15)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'energy-pushforward-2d{suffix}.pdf'))


def energy_pushforward_2d_scat(energies, labels, out_dir, suffix='', colors=None, figsize=FIG_SIZE_ONE_THIRD,
                          legend_inside=True, lims=None, mcmc_path=None, epsilons=()):
    fig, axs = plt.subplots(1, figsize=figsize)
    N_points = 1024
    for i, label in enumerate(labels):
        color = colors[i] if colors is not None else f'C{i}'
        n_points = len(energies[i])
        jump = max(n_points // N_points, 1)
        axs.scatter(energies[i][::jump, 0], energies[i][::jump, 1], s=5, color=color, label=label, alpha=0.1, edgecolors='none')
        energy_mean = energies[i].mean(0)
        # axs.plot(energy_mean[0], energy_mean[1], 'o', color=color, label=label, markersize='5', alpha=1, edgecolors='none')
        axs.plot(energy_mean[0], energy_mean[1], 'x', color=color, markersize='5', alpha=1)

    target_energy = energies[0].mean(0)
    phi = np.linspace(0, 2 * np.pi, 1000)
    for epsilon in epsilons:
        plt.plot(target_energy[0] + epsilon * np.cos(phi), target_energy[1] + epsilon * np.sin(phi), 'k--')

    axs.set_xlabel(r'$\phi_1$', math_fontfamily='cm')
    axs.set_ylabel(r'$\phi_2$')
    legend_kwargs = {} if legend_inside else {'bbox_to_anchor': (1.05, 1.05)}
    axs.legend(loc='upper left', handlelength=.75, labelspacing=0.2, handletextpad=0.4, **legend_kwargs).set_zorder(15)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'energy-pushforward-2d-scat{suffix}.pdf'))
    fig.savefig(os.path.join(out_dir, f'energy-pushforward-2d-scat{suffix}.png'), format='png', bbox_inches='tight', dpi=300)


def mcmc_path(energies, out_dir, suffix=''):
    fig, axs = plt.subplots(1, figsize=(3., 2.))
    for i in range(len(energies)):
        axs.plot(energies[i][:, 0], energies[i][:, 1], 'k', alpha=0.5, linewidth=0.5)
        axs.scatter(energies[i][:, 0], energies[i][:, 1], s=50, c=range(energies[i].shape[0]))
    axs.set_xlabel(r'$\phi_1$', math_fontfamily='cm')
    axs.set_ylabel(r'$\phi_2$')
    axs.set_xlim(-0.05, 0.15)
    axs.set_ylim(1., 1.5)
    # axs.legend(loc='upper left', handlelength=.75, labelspacing=0.2, handletextpad=0.4)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'mcmc-energy-path-2d{suffix}.pdf'))


def mcmc_entropies(entropies, llhs, temps, out_dir, suffix=''):
    fig, axs = plt.subplots(2, sharex='all')
    entropies = np.array([entropy.mean(0) for entropy in entropies])
    llhs = np.array([llh.mean(0) for llh in llhs])
    axs[0].semilogx(temps, -entropies, label='-Entropy')
    axs[0].semilogx(temps, llhs, label='LLH')
    axs[0].set_ylabel('Entropy and LLH')
    axs[1].semilogx(temps, -entropies - llhs, label='Rev KL')
    axs[1].set_xlabel('Temperature')
    axs[1].set_ylabel('Rev KL')
    axs[0].legend()
    # axs[1].legend()
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'mcmc-entropies{suffix}.pdf'))


def _imsave(path, img, cmap='gray', vmin=None, vmax=None):
    plt.imsave(f'{path}.png', img, cmap=cmap, vmin=vmin, vmax=vmax, format='png')
    plt.imsave(f'{path}.pdf', img, cmap=cmap, vmin=vmin, vmax=vmax, format='pdf', dpi=300)
    plt.close()


def imshow(samples, true_samples, out_dir, vmin, vmax, latent_samples=None, prefix='ising', suffix='', cmap='gray', also_rounded=True):
    os.makedirs(out_dir, exist_ok=True)
    max_N = 16
    for i in range(min(len(true_samples), max_N)):
        _imsave(os.path.join(out_dir, f'{prefix}-true_sample{i}'), true_samples[i], cmap=cmap, vmin=vmin, vmax=vmax)
    if latent_samples is not None:
        for i in range(min(len(latent_samples), max_N)):
            _imsave(os.path.join(out_dir, f'{prefix}-latent_sample{i}'), latent_samples[i], cmap=cmap, vmin=vmin, vmax=vmax)
    if prefix == 'ising':
        fig, ax = plt.subplots(1)
        ax.hist(samples.reshape(-1), 50, density=True)
        fig.savefig(os.path.join(out_dir, f'{prefix}-histogram-{suffix}.png'), format='png', dpi=300)
    for i in range(min(len(samples), max_N)):
        sample = samples[i]
        _imsave(os.path.join(out_dir, f'{prefix}-fake_sample{suffix}{i:03}'), sample, cmap=cmap, vmin=vmin, vmax=vmax)
        if also_rounded:
            rounded = np.ones_like(sample)
            rounded[sample < 0] = -1
            _imsave(os.path.join(out_dir, f'{prefix}2d{suffix}{i:03}-rounded'), rounded, cmap=cmap)
            if latent_samples is not None:
                base_rounded = np.ones_like(sample)
                base_rounded[latent_samples[i] < 0] = -1
                _imsave(os.path.join(out_dir, f'{prefix}2d{suffix}{i:03}-rounded-latent'), base_rounded, cmap=cmap)
                # for eps in 0.1, 0.01, 0.001:
                #     mask = np.ones_like(sample)
                #     mask[abs(sample-1) < eps] = 0
                #     mask[(sample+1) < eps] = 0
                #     plt.imsave(os.path.join(out_dir, f'fouls{eps}{suffix}{i:03}.png'), mask, cmap='gray', format='png')


def imshow_diff(samples, out_dir, prefix, suffix='', cmap='gray'):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1)
    ax.hist(samples.reshape(-1), 100, density=True)
    save_fig(fig, os.path.join(out_dir, f'{prefix}-mfdiff-histogram-{suffix}.pdf'))

    vmin = 0.
    vmax = np.quantile(samples, 0.99)

    max_N = 16
    for i, sample in enumerate(samples):
        if i >= max_N:
            break
        _imsave(os.path.join(out_dir, f'{prefix}-mfdiff-{suffix}-vmax{vmax:.4f}-{i:03}'), sample, cmap=cmap, vmin=vmin, vmax=vmax)


def plot_descent_comparison(generated_reg: np.ndarray, generated_mf: np.ndarray, out_dir: str, cmap: str = 'gray'):
    print('saving figures')
    fig, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(12, 4))
    data = [generated_reg, generated_mf, generated_mf - generated_reg]
    vmin0 = min(data[0].min(), data[1].min())
    vmax0 = max(data[0].max(), data[1].max())
    vmin1 = data[2].min()
    vmax1 = data[2].max()

    axs[0].set_title('Regular')
    axs[1].set_title('MF')
    axs[2].set_title('Difference')

    ims = [ax.imshow(d[0], cmap=cmap) for ax, d in zip(axs, data)]
    cbars = [fig.colorbar(im, ax=ax) for im, ax in zip(ims, axs)]

    ims[0].set_clim(vmin=vmin0, vmax=vmax0)
    ims[1].set_clim(vmin=vmin0, vmax=vmax0)
    ims[2].set_clim(vmin=vmin1, vmax=vmax1)

    total_steps = generated_reg.shape[0]
    if total_steps >= 100:
        snapshots = np.concatenate((np.arange(100), np.arange(100, total_steps, 25)))
    else:
        snapshots = np.arange(total_steps)
    for k in tqdm.tqdm(snapshots):
        for im, d in zip(ims, data):
            im.set_data(d[k])
        fig.savefig(os.path.join(out_dir, f'iter-{k:04}.png'), bbox_inches='tight', dpi=300)

    plt.close(fig)
