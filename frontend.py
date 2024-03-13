import os.path
import config
import numpy as np
import torch
import utils
import energies
import models
from generators import Generator, GeneratorMF, RevKLLoss
import plot
import pandas as pd

PCA_DIST_SEED = (1, config.MAIN_SEED)
LATENT_SEED = (2, config.MAIN_SEED)
TARGET_ENERGY_SEED = (3, config.MAIN_SEED)
GARCH_SEED = (4, config.MAIN_SEED)
PERT_SEED = (5, config.MAIN_SEED)


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

    plot.kl_and_parts(kl_losses_list[0], OUT_DIR, suffix='-regular')
    plot.kl_and_parts_in_same(kl_losses_list[1], OUT_DIR, suffix='-mf')

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
    for gen, label, suffix in zip([mgdm_min_kl, mgdm_overfitted, mf_mgdm_min_kl],
                                  ['MGDM', 'MGDM', 'MF-MGDM'],
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
                                   ['True', label, 'Latent'],
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
    plot.compare_mf_with_regular(kl_losses_list, ('MGDM', 'MF-MGDM'), OUT_DIR)
    return kl_losses_list


def synthetic_data_experiments_main():
    def grad_steps(m: models.Model, ef: energies.Energy):
        if m.__class__ == models.ARMA and ef.__class__ == energies.ScatMean:
            return 1000
        if m.__class__ == models.CIR0 and ef.__class__ == energies.ScatSpectra:
            return 1000
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
                   models.CIR1(TARGET_ENERGY_SEED)]
    energy_fns = [energies.ACF(logT),
                  energies.ScatMean(logT, j1=4, j2=6),
                  energies.ScatCovPCA(logT, j1=4, j2=6, phase_shifts=2, pca_model=models.Gaussian(PCA_DIST_SEED)),
                  'scatspectra']
    gen_kwargs = [{'grad_step_size': 10., 'gpu_bs_logdet': 64},
                  {'grad_step_size': 5., 'gpu_bs_logdet': 16},
                  {'grad_step_size': 5., 'gpu_bs_logdet': 11},
                  {'grad_step_size': 5., 'gpu_bs_logdet': 3}]
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
    df_kl.to_csv(os.path.join('output', 'kl-comp-synth-min.csv'))
    df_argmin.to_csv(os.path.join('output', 'kl-comp-synth-argmin.csv'))
    print('\n\nKL losses\n')
    print(df_kl.to_latex(float_format='%.2f'))
    print('\nargmins\n')
    print(df_argmin.to_latex())


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
                         N_gen: int, gen_kwargs: dict, check_robustness=False, init_garch=False, init_ar1garch=False):
    T_gen = 2 ** logT_gen
    T_true = 2 ** logT_true
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    true_data = torch.from_numpy(utils.load_real_data(dataset)[-T_true:]).to(torch.float32)

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

    OUT_DIR = os.path.join('output', dataset, f'{T_true}--{repr(energy_fn)}')
    os.makedirs(OUT_DIR, exist_ok=True)

    if init_ar1garch:
        latent_model = models.AR1GARCH.from_true_data(LATENT_SEED, true_data.numpy())
    elif init_garch:
        latent_model = models.GARCH.from_true_data(LATENT_SEED, true_data.numpy())
    else:
        latent_model = models.Gaussian(LATENT_SEED)
    latent_samples = latent_model.generate_sample(T_gen, N_gen).to(device)

    gen_kwargs_ex_expand = gen_kwargs.copy()
    if 'grad_step_size_expand' in gen_kwargs:
        del gen_kwargs_ex_expand['grad_step_size_expand']
    base_generator = Generator(energy_fn, target_energy, **gen_kwargs_ex_expand).to(device)
    mf_generator = GeneratorMF(energy_fn, target_energy, **gen_kwargs_ex_expand).to(device)
    generators = [base_generator, mf_generator]
    labels = ['MGDM', 'MF-MGDM']
    if 'grad_step_size_expand' in gen_kwargs:
        generators.append(GeneratorMF(energy_fn, target_energy, **gen_kwargs).to(device))
        labels.append('eMF-MGDM')

    samples = []
    for i, gen in enumerate(generators):
        filename = f'{repr(gen)}--{repr(latent_model)}--{N_gen}.npz'
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            npzfile = np.load(filepath)
            sample = npzfile['sample']
            error = npzfile['error']
        else:
            print(f'No previously generated samples for {filename}, generating new')
            sample, error = gen(latent_samples.clone().to(device), include_log_det=False, include_errors=True)
            sample = sample.cpu().numpy()
            np.savez(filepath, sample=sample, error=error)
        samples.append(sample)
        plot.inner_loss_though_descent(error, OUT_DIR, suffix=f'-{labels[i]}', color=f'C{i}')
        print(f'{repr(gen)} loss: {gen.loss(torch.from_numpy(sample).to(device)).mean().item()}')

    if check_robustness:
        evaluate_robustness(base_generator.loss, true_data.to(device), samples, labels, OUT_DIR)
    true_data = true_data.cpu().numpy()

    # GARCH approximation
    garch_model = models.AR1GARCH.from_true_data(GARCH_SEED, true_data)
    samples.append(garch_model.generate_sample(T_gen, N_gen))
    labels += ['GARCH']

    plot.compare_with_true(true_data, samples, labels, OUT_DIR)


def real_data_experiments_main():
    real_data_experiment('SP500', 12, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 20000, 'grad_step_size': 50., 'gpu_bs_gen': 1024})
    real_data_experiment('USD5Y', 12, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 2500, 'grad_step_size': 50., 'gpu_bs_gen': 1024})
    real_data_experiment('USD10Y', 12, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 2500, 'grad_step_size': 50., 'gpu_bs_gen': 1024})
    real_data_experiment('EUR5Y', 11, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 2500, 'grad_step_size': 50., 'gpu_bs_gen': 1024})
    real_data_experiment('EUR10Y', 11, 10, energies.ACF2,
                         {'lags': 20},
                         1024,
                         {'grad_steps': 5000, 'grad_step_size': 50., 'gpu_bs_gen': 1024})

    real_data_experiment('SP500', 12, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 500, 'grad_step_size': 1.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11})
    real_data_experiment('USD5Y', 12, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 250, 'grad_step_size': 1.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11})
    real_data_experiment('USD10Y', 12, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 2000, 'grad_step_size': 10.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11})
    real_data_experiment('EUR5Y', 11, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 1000, 'grad_step_size': 10.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11})
    real_data_experiment('EUR10Y', 11, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 1000, 'grad_step_size': 10.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11})

    real_data_experiment('SP500', 12, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 10000, 'grad_step_size': 1.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11})
    real_data_experiment('SP500', 12, 10, energies.ScatCovPCA,
                         {'j1': 4, 'j2': 6, 'phase_shifts': 2, 'pca_model': 'GARCH(0.03,0.1,0.87)'},
                         1024,
                         {'grad_steps': 2500, 'grad_step_size': 1.,
                          'gpu_bs_gen': 1024, 'gpu_bs_logdet': 11}, init_garch=True)


def main():
    ar1_experiment()
    real_data_experiments_main()
    synthetic_data_experiments_main()
    batch_size_experiment()


if __name__ == '__main__':
    main()
