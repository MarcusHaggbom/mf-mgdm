# Mean-Field Microcanonical Gradient Descent

Code to replicate results from the paper "Mean-Field Microcanonical Gradient Descent" by Marcus Häggbom, Morten 
Karlsmark and Joakim Andén (2025), available at https://arxiv.org/abs/2403.08362.

Run frontend.py to generate plots.

A global seed can be changed in config.py. 

Requires GPU.
Parameters gpu_bs_gen and especially gpu_ps_logdet will need to be tuned depending on GPU size.
Optimal gpu_bs depends on energy function and cannot be tweaked centrally in config, unfortunately.
We have used a 40GB or 80GB A100, depending on load.
Generation only (without computing entropy) requires much less memory, but can still be high for high-dimensional signals and complex energy functions.

For Bubbles experiments with real data, download `demo_brDuD111_N256.mat` from https://cloud.irit.fr/s/PIVBO4JJBT73rcp and save in `data/imgs/`.


Copyright 2025 Marcus Häggbom
