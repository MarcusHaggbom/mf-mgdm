import os.path
import config
import numpy as np
import torch
import utils
import energies
import models
import generators
from generators import Generator, GeneratorMF, GeneratorConstrained, RevKLLoss, EntropyCalculator
import plot
import pandas as pd
import time
import scipy.io

PCA_DIST_SEED = (1, config.MAIN_SEED)
LATENT_SEED = (2, config.MAIN_SEED)
TARGET_ENERGY_SEED = (3, config.MAIN_SEED)
GARCH_SEED = (4, config.MAIN_SEED)
PERT_SEED = (5, config.MAIN_SEED)
TS_SLICE_SEED = (6, config.MAIN_SEED)
LANGEVIN_SEED = (7, config.MAIN_SEED)
HUTCHINSON_SEED = (8, config.MAIN_SEED)


def ar1_experiment():
    logT = 10
    N_gen = 1024
    N_gen_kl = 128
    N_true = 10000
    gpu_bs_logdet = 8
    grad_steps = 200
    grad_step_size = 10.
    true_model = models.ARMA(TARGET_ENERGY_SEED, arcoefs=(0.1,))
    energy_fn = energies.ACF(logT)

    OUT_DIR = os.path.join('output', repr(true_model), repr(energy_fn))
    os.makedirs(OUT_DIR, exist_ok=True)
    T = 2 ** logT
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    latent_model = models.Gaussian(LATENT_SEED)
    latent_samples = latent_model.generate_sample(T, N_gen_kl).to(device)
    true_samples = true_model.generate_sample(T, N_true)
    true_energies = energy_fn(true_samples)
    target_energy = true_energies.mean(0)

    kl_loss_fn = RevKLLoss(true_model, latent_model, None, None, device)

    generator = Generator(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_logdet=gpu_bs_logdet,
                          project=true_model.non_negative).to(device)
    generator_mf = GeneratorMF(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_logdet=gpu_bs_logdet,
                               project=true_model.non_negative).to(device)

    kl_losses_list = []
    for gen in [generator, generator_mf]:
        filename = f'kl--{repr(gen)}--{repr(latent_model)}--{N_gen_kl}--{T}.npy'
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            kl_losses = np.load(filepath)
        else:
            kl_losses = kl_loss_fn.through_descent_batched(gen, N_gen_kl, latent_samples.clone(), True).numpy()
            np.save(filepath, kl_losses)
        kl_losses_list.append(kl_losses)

    plot.kl_and_parts_in_same(kl_losses_list[0] / T, OUT_DIR, suffix='-regular', figsize=(3.4, 1.7), adjust=3)
    plot.kl_and_parts_in_same(kl_losses_list[1] / T, OUT_DIR, suffix='-mf', figsize=(3.45, 1.7), adjust=3)

    kl_min = kl_losses_list[0][:, 0].min()
    kl_min_mf = kl_losses_list[1][:, 0].min()
    kl_argmin = kl_losses_list[0][:, 0].argmin()
    kl_argmin_mf = kl_losses_list[1][:, 0].argmin()
    print(f'KL min   : {kl_min:.2f} (argmin: {kl_argmin})')
    print(f'KL min MF: {kl_min_mf:.2f} (argmin: {kl_argmin_mf})')

    latent_model = models.Gaussian(LATENT_SEED)
    latent_samples = latent_model.generate_sample(T, N_gen).to(device)
    latent_energies = energy_fn(latent_samples).cpu().numpy()

    mgdm_min_kl = Generator(energy_fn, target_energy, kl_argmin, grad_step_size, gpu_bs_logdet=gpu_bs_logdet,
                            project=true_model.non_negative).to(device)
    overfit_iterates = 100
    mgdm_overfitted = Generator(energy_fn, target_energy, overfit_iterates, grad_step_size, gpu_bs_logdet=gpu_bs_logdet,
                                project=true_model.non_negative).to(device)
    mf_mgdm_min_kl = GeneratorMF(energy_fn, target_energy, kl_argmin_mf, grad_step_size, gpu_bs_logdet=gpu_bs_logdet,
                                 project=true_model.non_negative).to(device)

    label_list = [[r'$\Phi_\#p$', r'$\Phi_\#q_{' + str(kl_argmin) + '}$', r'$\Phi_\#q_0$'],
                  [r'$\Phi_\#p$', r'$\Phi_\#q_{' + str(overfit_iterates) + '}$', r'$\Phi_\#q_0$'],
                  [r'$\Phi_\#p$', r'$\Phi_\#\overline{q}_{' + str(kl_argmin_mf) + '}$', r'$\Phi_\#\overline{q}_0$']]
    for gen, labels, suffix in zip([mgdm_min_kl, mgdm_overfitted, mf_mgdm_min_kl],
                                   label_list,
                                   ['-regular-min-kl', '-regular-overfit', '-mf-min-kl']):
        filename = f'energies--{repr(gen)}--{repr(latent_model)}--{N_gen}--{T}.npy'
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            model_energies = np.load(filepath)
        else:
            samples = gen(latent_samples.clone().to(device), include_log_det=False)
            model_energies = energy_fn(samples).cpu().numpy()
            np.save(filepath, model_energies)
        plot.energy_pushforward_2d([true_energies.numpy(), model_energies, latent_energies],
                                   labels,
                                   OUT_DIR, suffix=suffix)


def batch_size_experiment():
    logT = 10
    N_gen_kl = 128
    N_true = 10000
    gpu_bs_logdet = 8
    grad_steps = 200
    grad_step_size = 10.
    true_model = models.ARMA(TARGET_ENERGY_SEED, arcoefs=(0.1,))
    energy_fn = energies.ACF(logT)

    OUT_DIR = os.path.join('output', repr(true_model), f'comp-mf-bs-{N_gen_kl}')
    os.makedirs(OUT_DIR, exist_ok=True)
    T = 2 ** logT
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    latent_model = models.Gaussian(LATENT_SEED)
    latent_samples = latent_model.generate_sample(T, N_gen_kl).to(device)
    target_energy = energy_fn(true_model.generate_sample(T, N_true)).mean(0)

    kl_loss_fn = RevKLLoss(true_model, latent_model, None, None, device)

    generator = Generator(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_logdet=gpu_bs_logdet,
                          project=true_model.non_negative).to(device)
    generator_mf = GeneratorMF(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_logdet=gpu_bs_logdet,
                               project=true_model.non_negative).to(device)
    kl_losses_per_bs = []
    bs_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for mf_bs in bs_list:
        if mf_bs == 1:
            gen = generator
        else:
            gen = generator_mf
        filename = f'{repr(generator)}--{repr(latent_model)}--bs{mf_bs}.npy'
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            kl_losses = np.load(filepath)
        else:
            kl_losses = kl_loss_fn.through_descent_batched(gen, mf_bs if mf_bs != 1 else N_gen_kl,
                                                           latent_samples.clone(), True)
            np.save(filepath, kl_losses)
        kl_losses_per_bs.append(kl_losses)

    plot.compare_mf_batch_size(bs_list, kl_losses_per_bs, OUT_DIR)


def synthetic_data_experiment(true_model: models.Model, logT: int, N_gen: int, energy_fn: energies.Energy,
                              gen_kwargs: dict):
    T = 2 ** logT
    N_true = 10000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    energy_fn = energy_fn.cpu()
    latent_model = models.get_maximum_entropy_model(true_model, LATENT_SEED)
    target_energy = energy_fn(true_model.generate_sample(T, N_true)).mean(0)
    OUT_DIR = os.path.join('output', repr(true_model), repr(energy_fn))
    os.makedirs(OUT_DIR, exist_ok=True)

    generator = Generator(energy_fn, target_energy, **gen_kwargs, project=true_model.non_negative).to(device)
    generator_mf = GeneratorMF(energy_fn, target_energy, **gen_kwargs, project=true_model.non_negative).to(device)
    latent_samples = latent_model.generate_sample(T, N_gen).to(device)
    kl_loss_fn = RevKLLoss(true_model, latent_model, None, None, device)
    kl_losses_list = []
    for gen in [generator, generator_mf]:
        filename = f'kl--{repr(gen)}--{repr(latent_model)}--{N_gen}--{T}.npy'
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            kl_losses = np.load(filepath)
        else:
            kl_losses = kl_loss_fn.through_descent_batched(gen, N_gen, latent_samples.clone(), True).numpy()
            np.save(filepath, kl_losses)
        kl_losses_list.append(kl_losses)
    plot.compare_mf_with_regular([losses / T for losses in kl_losses_list], ('MGDM', 'MF--MGDM'), OUT_DIR)
    return kl_losses_list


def synthetic_data_experiments_main():
    def grad_steps(m: models.Model, ef: energies.Energy):
        if m.__class__ == models.ARMA and ef.__class__ == energies.ScatMean:
            return 1000
        if m.__class__ == models.CIR0 and ef.__class__ == energies.ScatSpectra:
            return 5000
        if m.__class__ == models.CIR0 and ef.__class__ != energies.ScatMean:
            return 500
        if m.__class__ == models.CIR1 and ef.__class__ == energies.ACF:
            return 2000
        if m.__class__ == models.CIR1 and ef.__class__ == energies.ScatCovPCA:
            return 1000
        return 250

    logT = 10
    N_gen = 128
    df_kl = pd.DataFrame()
    df_argmin = pd.DataFrame()
    true_models = [models.ARMA(TARGET_ENERGY_SEED, arcoefs=(0.1,)),
                   models.ARMA(TARGET_ENERGY_SEED, arcoefs=(0.2, -0.1)),
                   models.ARMA(TARGET_ENERGY_SEED, arcoefs=(-0.1, 0.2, 0.1)),
                   models.CIR0(TARGET_ENERGY_SEED),
                   models.CIR1(TARGET_ENERGY_SEED),
                   ]
    energy_fns = [energies.ACF(logT),
                  energies.ScatMean(logT, j1=4, j2=6),
                  energies.ScatCovPCA(logT, j1=4, j2=6, phase_shifts=2, pca_model=models.Gaussian(PCA_DIST_SEED)),
                  'scatspectra',
                  ]
    gen_kwargs = [{'grad_step_size': 10., 'gpu_bs_logdet': 64},
                  {'grad_step_size': 5., 'gpu_bs_logdet': 16},
                  {'grad_step_size': 5., 'gpu_bs_logdet': 11},
                  {'grad_step_size': 5., 'gpu_bs_logdet': 3},
                  ]
    assert len(energy_fns) == len(gen_kwargs)

    for true_model in true_models:
        for energy_fn, gen_kwarg in zip(energy_fns, gen_kwargs):
            if energy_fn == 'scatspectra':
                energy_fn = energies.ScatSpectra(logT, true_model.clone(), include_phase=False)
            print(repr(true_model), repr(energy_fn))
            gen_kwarg['grad_steps'] = grad_steps(true_model, energy_fn)
            kl_losses_list = synthetic_data_experiment(true_model.clone(), logT, N_gen, energy_fn, gen_kwarg)
            min_kls = [kl_losses[:, 0].min() for kl_losses in kl_losses_list]
            argmins_kls = [kl_losses[:, 0].argmin() for kl_losses in kl_losses_list]
            df_kl.loc[repr(true_model), repr(energy_fn)] = min_kls[0]
            df_kl.loc[repr(true_model), repr(energy_fn) + '-mf'] = min_kls[1]
            df_argmin.loc[repr(true_model), repr(energy_fn)] = argmins_kls[0]
            df_argmin.loc[repr(true_model), repr(energy_fn) + '-mf'] = argmins_kls[1]
    os.makedirs('output', exist_ok=True)
    df_kl.to_csv(os.path.join('output', 'kl-comp-synth-min.csv'))
    df_argmin.to_csv(os.path.join('output', 'kl-comp-synth-argmin.csv'))
    print('\n\nKL losses\n')
    print(df_kl.to_latex(float_format='%.2f'))
    print('\nargmins\n')
    print(df_argmin.to_latex())


class Rounding:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        return torch.sign(x) * (torch.abs(x) ** self.p)


def ising_experiment_main():
    # high res sampling
    for rounding_factor in (0.05,):
        ising_experiment(False, 8, 7, 16, 128, 16, 2.4, 10000, generators.LRConst(250.), generators.LRConst(50.),
                         rounding_factor, 128)
        ising_experiment(False, 8, 7, 16, 128, 16, 2.4, 10000, generators.LRConst(250.), generators.LRConst(50.),
                         rounding_factor, 128, efn='scattering')

    # low res entropy comparison
    for rounding_factor in (0.05,):
        ising_experiment(True, 5, 5, 128, 128, 16, 4, 150, generators.LRConst(250.), generators.LRConst(2.),
                         rounding_factor, 1024)
        ising_experiment(True, 5, 5, 128, 128, 16, 4, 150, generators.LRConst(250.), generators.LRConst(2.),
                         rounding_factor, 1024, efn='scattering')
        # ising_experiment(True, 5, 5, 128, 128, 16, 4, 150, generators.LRConst(250.), generators.LRConst(2.),
        #                  rounding_factor, 1024, final_fn_applications=5)


def ising_experiment(compute_entropy: bool, logT: int, J: int, N_gen: int, N_gen_mf: int, gpu_bs_logdet: int,
                     temperature: float,
                     grad_steps: int, lr: generators.LR, constraint_lr: generators.LR, rounding_factor: float | None,
                     N_true: int | None = None, gpu_bs_gen: int | None = None, efn: str = 'wavelet',
                     final_fn_applications: int = 1):
    if N_true is None:
        N_true = 2 ** (15 - logT)
    if gpu_bs_gen is None:
        gpu_bs_gen = max(N_gen, N_gen_mf)
    T = 2 ** logT

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ising_model = models.Ising(TARGET_ENERGY_SEED, temperature, 1024)
    relaxed_ising_model = models.IsingRelaxed(TARGET_ENERGY_SEED, temperature, 1024, alpha=1.25)
    if efn == 'wavelet':
        energy_fn = energies.WaveletL1(logT, j1=J, j2=J, dim=2, Q=8).to(device)
    elif efn == 'scattering':
        energy_fn = energies.ScatMean2d(logT, j1=J, j2=J, dim=2, Q=2).to(device)
    else:
        raise ValueError(f'Unrecognized energy function {efn}')
    constraint_fn = energies.Combined([energies.LpNorm(2, dim=2), energies.LpNorm(1, dim=2)]).to(device)
    target_constr = torch.ones(2, device=device)

    true_samples = ising_model.generate_sample(T, N_true)
    target_energy = torch.stack([energy_fn(s.to(device)) for s in true_samples], dim=0).mean(0)
    rounding_fn = None if rounding_factor is None else Rounding(rounding_factor)
    OUT_DIR = os.path.join('output', f'{repr(relaxed_ising_model)}--{N_true}', repr(energy_fn), str(rounding_factor))
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR: {OUT_DIR}')

    generator_list = []
    labels = []
    N_gen_list = []
    for is_mf in [False, True]:
        generator_list.append(GeneratorConstrained(energy_fn, target_energy, grad_steps, lr, gpu_bs_gen=gpu_bs_gen,
                                                   gpu_bs_logdet=gpu_bs_logdet, is_mf=is_mf,
                                                   constr_efn=constraint_fn,
                                                   constr_target=target_constr,
                                                   constr_grad_step_size=constraint_lr,
                                                   final_fn=rounding_fn,
                                                   final_fn_applications=final_fn_applications).to(device))
        labels.append('MF-MGDM' if is_mf else 'MGDM')
        N_gen_list.append(N_gen_mf if is_mf else N_gen)

    print('True expected losses:')
    for gen in generator_list:
        print(f'\t{gen.loss(true_samples.to(device)).mean(0).item()}')

    latent_model = models.Gaussian2D(LATENT_SEED)
    latent_samples = latent_model.generate_sample(T, max(N_gen, N_gen_mf)).to(device)

    if compute_entropy:
        kl_loss_fn = RevKLLoss(relaxed_ising_model, latent_model, None, None, device)
        kl_losses_list = []
        for i, gen in enumerate(generator_list):
            N_gen_i = N_gen_list[i]
            filename = f'kl--{repr(gen)}--{N_gen_i}--{T}.npy'
            filepath = os.path.join(OUT_DIR, filename)
            if os.path.exists(filepath):
                kl_losses = np.load(filepath)
            else:
                kl_losses = kl_loss_fn.through_descent_batched(gen, N_gen_i, latent_samples[:N_gen_i].clone(),
                                                               True).numpy()
                np.save(filepath, kl_losses)
            kl_losses_list.append(kl_losses)
        plot.compare_mf_with_regular_entr(kl_losses_list, labels, OUT_DIR, dim=T ** 2,
                                          final_fn_applications=final_fn_applications)

    samples = []
    for i, gen in enumerate(generator_list):
        N_gen_i = N_gen_list[i]
        filename = f'{repr(gen)}--{N_gen_i}.npz'
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            npzfile = np.load(filepath)
            sample = npzfile['sample']
            error = npzfile['error']
        else:
            print(f'No previously generated samples for {filename}, generating new...')
            t0 = time.time()
            sample, error = gen(latent_samples[:N_gen_i].clone().to(device), include_log_det=False, include_errors=True)
            print(f'... done; generated in {time.time() - t0:.2f} s')
            sample = sample.cpu().numpy()
            np.savez(filepath, sample=sample, error=error)
        samples.append(sample)
        print(f'final losses: {error[-1]}')
        out_of_bounds = np.concatenate([abs(sample[sample >= 0] - 1), abs(sample[sample < 0] + 1)])
        print(f'{repr(gen)} out of bounds: {out_of_bounds.mean()} {np.quantile(out_of_bounds, (0.5, 0.9, 0.99))}')

        plot.inner_losses_though_descent(error, OUT_DIR, labels=['Energy', 'Constraint'], suffix=f'-{labels[i]}')
        print(f'{repr(gen)} loss: {gen.loss(torch.from_numpy(sample).to(device)).mean().item()}')
        plot.imshow(sample, true_samples.numpy(), latent_samples.cpu().numpy(),
                    os.path.join(OUT_DIR, f'figs-{labels[i]}'), suffix=f'-{labels[i]}')

    return


def compare_generation_process(sample_indices: tuple, samples_reg: dict[int, np.ndarray],
                               samples_mf: dict[int, np.ndarray], out_dir: str):
    out_dir = os.path.join(out_dir, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    for idx in sample_indices:
        out_dir_i = os.path.join(out_dir, str(idx))
        if os.path.exists(out_dir_i):
            out_dir_i = utils.next_file_path(out_dir_i)
        os.makedirs(out_dir_i, exist_ok=False)
        plot.plot_descent_comparison(samples_reg[idx], samples_mf[idx], out_dir_i)


def load_images(dataset: str, img_nbr: int | None, img_size: int):
    if img_size % 2 != 0:
        raise ValueError(f'Image size {img_size} is not divisible by 2')

    match dataset:
        case 'bubbles':
            imgs = scipy.io.loadmat(os.path.join('data', 'imgs', 'demo_brDuD111_N256.mat'))['imgs']
        case 'mrw':
            imgs = scipy.io.loadmat(os.path.join('data', 'imgs', 'demo_mrw2dd_train_N256.mat'))['imgs']
        case 'turbulence':
            imgs = scipy.io.loadmat(os.path.join('data', 'imgs', 'ns_randn4_train_N256.mat'))['imgs']
        case _:
            raise ValueError(f'Unrecognized dataset {dataset}')
    imgs = torch.tensor(imgs, dtype=torch.float32)
    imgs = imgs.permute(2, 0, 1)
    if img_nbr is not None:
        imgs = imgs[img_nbr:img_nbr + 1]
    B, N, M = imgs.shape
    if N != M:
        raise NotImplementedError('Non-square images not supported')
    if img_size > N:
        raise ValueError(f'Image size {img_size} is not smaller than the image size {N}')
    k = N // img_size
    patches = imgs.unfold(1, img_size, img_size).unfold(2, img_size, img_size)
    patches = patches.reshape(B * k * k, img_size, img_size)
    return patches


class FractionalIntegrate:
    def __init__(self, size: int, H: float):
        kx = torch.arange(-size / 2, size / 2)
        fx, fy = torch.meshgrid(kx, kx, indexing='xy')
        fmod = (fx ** 2 + fy ** 2) ** .5
        fmod[fmod == 0] = torch.min(fmod[fmod != 0])
        self.ffilter = torch.fft.ifftshift(1 / (fmod ** (1 + H)))

    def __call__(self, x: torch.Tensor):
        assert self.ffilter.shape == x.shape[-2:]
        return torch.fft.ifft2(self.ffilter * torch.fft.fft2(x)).real

    def to(self, device):
        self.ffilter = self.ffilter.to(device)
        return self

    def cpu(self):
        self.ffilter = self.ffilter.cpu()
        return self


def synthetic_2d_experiment(dataset, img_nbr=None, N_gen=128, img_size=128, img_training_size=None, grad_steps=1000,
                            grad_step_size: float | generators.LR = 0.01,
                            gpu_bs_gen=1, gpu_bs_logdet=1, gpu_bs_loss=None, approx_entropy=False,
                            generate_samples=True, compute_entropy=False, visual_comparison=False,
                            tracked_indices=(0,)):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    imgs = load_images(dataset, None, img_size)
    signal_std = imgs.std(dim=(-2, -1)).mean().item()
    if img_training_size is None:
        img_training_size = img_size
    if img_training_size != img_size:
        raise NotImplementedError(
            'Need to instantiate another energy_fn and make sure internal, calibrated parameters (e.g. DivInitStd) are equivalent')
    training_imgs = load_images(dataset, img_nbr, img_training_size)

    J = 5
    L = 8
    M, N = imgs.shape[-2:]
    assert M == N
    energy_fn = energies.PhaseHarmonicCovD(M, N, J, L, delta_j=1, delta_l=L, delta_k=0, nb_chunks=1, chunk_id=0,
                                           stdnorm=1, kmax=None).to(device)
    target_energy = energy_fn(training_imgs.to(device)).mean(0)
    print(target_energy.shape)

    OUT_DIR = os.path.join('output', '2d-synth',
                           f'{dataset}-{img_size}x{img_size}-{img_training_size}-{"all" if img_nbr is None else img_nbr}',
                           repr(energy_fn))
    os.makedirs(OUT_DIR, exist_ok=True)

    latent_model = models.Gaussian2D(LATENT_SEED, std=signal_std)
    latent_samples = latent_model.generate_sample(M, N_gen).to(device)

    assert all(0 <= idx < N_gen for idx in tracked_indices)

    final_fn = FractionalIntegrate(img_size, 0.2) if dataset == 'mrw' else None
    if final_fn is not None:
        imgs = final_fn(imgs)
        final_fn = final_fn.to(device)

    gen_reg = GeneratorConstrained(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_gen, gpu_bs_logdet,
                                   gpu_bs_loss,
                                   project=False, is_mf=False,
                                   approx_logdet=approx_entropy, hutchinson_seed=HUTCHINSON_SEED, hutchinson_mc_count=4,
                                   final_fn=final_fn).to(device)
    gen_mf = GeneratorConstrained(energy_fn, target_energy, grad_steps, grad_step_size, gpu_bs_gen, gpu_bs_logdet,
                                  gpu_bs_loss,
                                  project=False, is_mf=True,
                                  approx_logdet=approx_entropy, hutchinson_seed=HUTCHINSON_SEED, hutchinson_mc_count=4,
                                  final_fn=final_fn).to(device)

    generators_and_labels = [
        (gen_reg, 'MGDM'),
        (gen_mf, 'MF-MGDM'),
    ]

    tracked_samples_dict = {}
    if generate_samples or visual_comparison:
        ims_to_show = []
        for gen, label in generators_and_labels:
            filename = f's--{N_gen}--{N}--{repr(gen)}.npz'
            filepath = os.path.join(OUT_DIR, filename)
            regenerate = False

            indices_to_track = set(tracked_indices)

            if os.path.exists(filepath):
                npzfile = np.load(filepath)
                sample = npzfile['sample']
                error = npzfile['error']
                cached_tracked_indices = set(
                    int(k.split('_')[-1]) for k in npzfile.keys() if k.startswith('tracked_sample_'))
                if set(tracked_indices) - cached_tracked_indices:
                    regenerate = True
                    indices_to_track |= cached_tracked_indices
                else:
                    tracked_samples = {i: npzfile[f'tracked_sample_{i}'] for i in tracked_indices}
            else:
                regenerate = True

            indices_to_track = tuple(indices_to_track)

            if regenerate:
                print(f'No previously generated samples for {filename} or missing tracked paths, generating new...')
                t0 = time.time()
                sample, error, tracked_samples = gen(latent_samples.clone().to(device), include_log_det=False,
                                                     include_errors=True, track_indices=indices_to_track)
                print(f'... done; generated in {time.time() - t0:.2f} s')
                sample = sample.cpu().numpy()
                tracked_samples = tracked_samples.cpu().numpy()
                tracked_samples = {i: tracked_sample for i, tracked_sample in zip(indices_to_track, tracked_samples)}
                np.savez(filepath, sample=sample, error=error,
                         **{f'tracked_sample_{i}': s for i, s in tracked_samples.items()})

            tracked_samples_dict[label] = tracked_samples
            ims_to_show.append(sample)
            plot.inner_loss_though_descent(error, OUT_DIR, labels=f'Inner loss {label}', suffix=f'-{label}')

        imgs = imgs.numpy()

        q = 0.005
        vmin = min(np.quantile(im, q) for im in ims_to_show + [imgs])
        vmax = max(np.quantile(im, 1 - q) for im in ims_to_show + [imgs])

        for (gen, label), samples in zip(generators_and_labels, ims_to_show):
            plot.imshow(sample, imgs, OUT_DIR, vmin, vmax, prefix=dataset, suffix=f'-{label}', also_rounded=False)

        plot.imshow_diff(np.abs(ims_to_show[1] - ims_to_show[0]) / signal_std, OUT_DIR, prefix=dataset)

    if visual_comparison:
        compare_generation_process(tracked_indices, tracked_samples_dict['MGDM'], tracked_samples_dict['MF-MGDM'],
                                   OUT_DIR)

    if compute_entropy:
        entropy_calculator = EntropyCalculator(latent_model)
        entropy_list = []
        label_list = []
        for gen, label in generators_and_labels:
            filename = f'kl--{N_gen}--{N}--{repr(gen)}.npy'
            filepath = os.path.join(OUT_DIR, filename)
            if os.path.exists(filepath):
                kl_losses = np.load(filepath)
            else:
                kl_losses = entropy_calculator.through_descent(gen, N_gen, latent_samples[:N_gen].clone()).numpy()
                np.save(filepath, kl_losses)
            entropy_list.append(kl_losses)
            label_list.append(label)
        plot.only_entropies(entropy_list, label_list, OUT_DIR, dim=M * N,
                            final_fn_applications=0 if final_fn is None else 1)


def synthetic_2d_experiment_main():
    # Bubbles ##########################################################################################################
    synthetic_2d_experiment('bubbles', None, 128, 32, grad_steps=500,
                            grad_step_size=generators.LRNSteps([.000005, .00001, .00005, .0001, .0005, .001, .005, .01],
                                                               [.1, .2, .3, .4, .5, .6, .7]),
                            gpu_bs_gen=128, gpu_bs_logdet=1, approx_entropy=False,
                            generate_samples=True, compute_entropy=True, visual_comparison=False,
                            tracked_indices=(0, 1, 2))
    synthetic_2d_experiment('bubbles', 0, 128, 256, grad_steps=2000,
                            grad_step_size=generators.LRNSteps([0.01, 0.1, 0.8], [0.15, 0.3]),
                            gpu_bs_gen=1, gpu_bs_logdet=1, gpu_bs_loss=32, approx_entropy=False,
                            generate_samples=True, compute_entropy=False, visual_comparison=False,
                            tracked_indices=(0, 1, 2))


def evaluate_robustness(loss_fn, true_sample, gen_samples, gen_labels, out_dir):
    device = true_sample.device
    rng = np.random.default_rng(PERT_SEED)
    M = 100
    gen_samples = [torch.from_numpy(gen_sample).to(true_sample.dtype).to(device) for gen_sample in gen_samples]
    assert all(true_sample.shape[-1] == gen_sample.shape[-1] for gen_sample in gen_samples)
    T = true_sample.shape[-1]
    noise_levels = 10. ** np.arange(-4, 0)
    perturbation = torch.from_numpy(rng.normal(0, 1, (M, T))).to(true_sample.dtype).to(device)
    for i, (gen_sample, gen_label) in enumerate(zip(gen_samples, gen_labels)):
        true_losses = []
        sample_losses = []
        unperturbed_loss = loss_fn(gen_sample).cpu().numpy()
        for noise_level in noise_levels:
            true_losses.append(loss_fn(true_sample + noise_level * perturbation).mean(0).item() / noise_level)
            perturbed_loss = np.array([loss_fn(x + noise_level * perturbation).mean(0).item() for x in gen_sample])
            sample_losses.append((perturbed_loss - unperturbed_loss) / noise_level)
        sample_losses = np.array(sample_losses)
        plot.robustness(true_losses, sample_losses, noise_levels, out_dir, suffix=f'-{gen_label}', color=f'C{i}')


def real_data_experiment(dataset: str, logT_true: int, logT_gen: int, energy_cls, energy_kwargs: dict,
                         N_gen: int, gen_kwargs: dict, check_robustness=False, init_garch=False, init_ar1garch=False,
                         init_student_t=False, n_true_samples: int = 1, compute_entropy=False):
    T_gen = 2 ** logT_gen
    T_true = 2 ** logT_true
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    true_data_all = utils.load_real_data(dataset)
    assert true_data_all.shape[0] >= T_true * n_true_samples, f'Not enough data for {n_true_samples}'
    true_data_samples = true_data_all[-T_true * n_true_samples:].reshape(n_true_samples, T_true)
    true_data = torch.from_numpy(true_data_samples[0]).to(torch.float32)
    true_data_validation = true_data_samples[1:]
    true_data = (true_data - true_data.mean()) / true_data.std()
    true_data_validation = (true_data_validation - true_data_validation.mean(-1, keepdim=True)) / true_data_validation.std(-1, keepdims=True)

    if energy_cls == energies.ScatCovPCA:
        if not energy_kwargs['pca_model'].startswith('GARCH'):
            raise NotImplementedError(energy_kwargs['pca_model'])  # TODO
        energy_kwargs['pca_model'] = models.GARCH.from_str(PCA_DIST_SEED, energy_kwargs['pca_model'])
        energy_fn = energy_cls(logT_gen, **energy_kwargs)
        target_energy = energy_cls(logT_true, **energy_kwargs, projector=energy_fn.pc_projector)(true_data)
    elif energy_cls == energies.ScatSpectra:
        _energy_fn = energy_cls(logT_true, true_data, **energy_kwargs)
        target_energy = _energy_fn(true_data)
        energy_fn = energy_cls(logT_gen, None, **energy_kwargs, sigma2=_energy_fn.sigma2)
        del _energy_fn
    else:
        target_energy = energy_cls(logT_true, **energy_kwargs)(true_data)
        energy_fn = energy_cls(logT_gen, **energy_kwargs)

    OUT_DIR = os.path.join('output', dataset, f'{T_true}-{T_gen}-{n_true_samples}--{repr(energy_fn)}')
    os.makedirs(OUT_DIR, exist_ok=True)

    if init_ar1garch:
        latent_model = models.AR1GARCH.from_true_data(LATENT_SEED, true_data.numpy())
    elif init_garch:
        latent_model = models.GARCH.from_true_data(LATENT_SEED, true_data.numpy())
    elif init_student_t:
        latent_model = models.StudentT(LATENT_SEED)
    else:
        latent_model = models.Gaussian(LATENT_SEED)
    latent_samples = latent_model.generate_sample(T_gen, N_gen).to(device)

    gen_kwargs_ex_expand = gen_kwargs.copy()
    if 'grad_step_size_expand' in gen_kwargs:
        del gen_kwargs_ex_expand['grad_step_size_expand']
    base_generator = Generator(energy_fn, target_energy, **gen_kwargs_ex_expand).to(device)
    mf_generator = GeneratorMF(energy_fn, target_energy, **gen_kwargs_ex_expand).to(device)
    generators = [base_generator, mf_generator]
    labels = ['MGDM', 'MF--MGDM']

    samples = []
    for i, gen in enumerate(generators):
        filename = f'{repr(gen)}--{repr(latent_model)}--{N_gen}.npz'
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            npzfile = np.load(filepath)
            sample = npzfile['sample']
            error = npzfile['error']
        else:
            print(f'No previously generated samples for {filename}, generating new...')
            t0 = time.time()
            sample, error = gen(latent_samples.clone().to(device), include_log_det=False, include_errors=True)
            print(f'... done; generated in {time.time() - t0:.2f} s')
            sample = sample.cpu().numpy()
            np.savez(filepath, sample=sample, error=error)
        samples.append(sample)
        plot.inner_loss_though_descent(error, OUT_DIR, suffix=f'-{labels[i]}', color=f'C{i}')
        print(f'{repr(gen)} loss: {gen.loss(torch.from_numpy(sample).to(device)).mean().item()}')

    if compute_entropy:
        entropies = []
        entropy_calculator = EntropyCalculator(latent_model)
        for i, gen in enumerate(generators):
            filename = f'{repr(gen)}--{repr(latent_model)}--{N_gen}-entropy.npy'
            filepath = os.path.join(OUT_DIR, filename)
            if os.path.exists(filepath):
                entropy = np.load(filepath)
            else:
                print(f'No previously computed entropy for {filename}, computing new...')
                t0 = time.time()
                entropy = entropy_calculator.through_descent(gen, N_gen, latent_samples.clone()).numpy()
                print(f'... done; computed in {time.time() - t0:.2f} s')
                np.save(filepath, entropy)
            print(f'{repr(gen)} entropy: {entropy[-1]} (init {entropy[0]})')
            entropies.append(entropy)
        plot.only_entropies(entropies, labels, OUT_DIR, T_gen)

    if check_robustness:
        evaluate_robustness(base_generator.loss, true_data.to(device), samples, labels, OUT_DIR)
    true_data = true_data.cpu().numpy()

    # GARCH approximation
    garch_model = models.AR1GARCH.from_true_data(GARCH_SEED, true_data)
    samples.append(garch_model.generate_sample(T_gen, N_gen))
    labels += ['GARCH']

    plot.compare_with_true(true_data, true_data_validation, samples, labels, OUT_DIR)

    rng_slice = np.random.default_rng(TS_SLICE_SEED)
    plot.compare_timeseries_with_true(true_data_validation, samples, rng_slice, [dataset, *labels], OUT_DIR)


def real_data_experiments_main():
    real_data_experiment('SP500', 10, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 20000, 'grad_step_size': 50., 'gpu_bs_gen': 1024}, n_true_samples=4,
                         compute_entropy=True)
    real_data_experiment('USD5Y', 10, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 2500, 'grad_step_size': 50., 'gpu_bs_gen': 1024}, n_true_samples=4)
    real_data_experiment('USD10Y', 10, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 2500, 'grad_step_size': 50., 'gpu_bs_gen': 1024}, n_true_samples=4)
    real_data_experiment('EUR5Y', 9, 9, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 500, 'grad_step_size': 50., 'gpu_bs_gen': 1024}, n_true_samples=4)
    real_data_experiment('EUR10Y', 9, 9, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 500, 'grad_step_size': 50., 'gpu_bs_gen': 1024}, n_true_samples=4)

    real_data_experiment('SP500', 10, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 500, 'grad_step_size': 1.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, n_true_samples=4)
    real_data_experiment('USD5Y', 10, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 250, 'grad_step_size': 1.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, n_true_samples=4)
    real_data_experiment('USD10Y', 10, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 2000, 'grad_step_size': 10.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, n_true_samples=4)
    real_data_experiment('EUR5Y', 9, 9, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 1000, 'grad_step_size': 10.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, n_true_samples=4)
    real_data_experiment('EUR10Y', 9, 9, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 1000, 'grad_step_size': 10.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, n_true_samples=4)

    real_data_experiment('SP500', 10, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 10000, 'grad_step_size': 1.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, n_true_samples=4)
    real_data_experiment('SP500', 10, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 25000, 'grad_step_size': 0.1,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, n_true_samples=4, init_garch=True)


def main():
    ar1_experiment()
    synthetic_data_experiments_main()
    batch_size_experiment()
    ising_experiment_main()
    synthetic_2d_experiment_main()


if __name__ == '__main__':
    main()
