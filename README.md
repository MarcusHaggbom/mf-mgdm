# Mean-Field Microcanonical Gradient Descent

Code to replicate results from the paper "Mean-Field Microcanonical Gradient Descent" by Marcus Häggbom, Morten 
Karlsmark and Joakim Andén (2024), available at https://arxiv.org/abs/2403.08362.

Run frontend.py to generate plots.

A global seed can be changed in config.py. 

Requires GPU.
Parameters gpu_bs_gen and especially gpu_ps_logdet will need to be tuned depending on GPU size.
Optimal gpu_bs depends on energy function and cannot be tweaked centrally in config, unfortunately.
We have used a 40GB A100, except for computing entropy for the approximations on S&P 500 where 80GB was needed.
However, generating only (without computing entropy) requires much less memory.

For experiments with real data, download the following data:
* https://finance.yahoo.com/quote/%5EGSPC/history save in data/sp500.csv
* https://www.riksbank.se/en-gb/statistics/interest-rates-and-exchange-rates/search-interest-rates-and-exchange-rates/?s=g100-EMGVB10Y&s=g100-USGVB10Y&s=g7-SEGVB10YC&s=g7-SEGVB2YC&s=g7-SEGVB5YC&s=g7-SEGVB7YC&s=g97-EUDP3MEUR&s=g97-EUDP3MUSD&s=g98-EUDP6MEUR&s=g98-EUDP6MUSD&s=g99-EMGVB5Y&s=g99-USGVB5Y&a=D&from=2000-01-03&to=2024-04-30&fs=3#result-section save in data/rates.csv

Data up until and including 2024-04-30 was used in the paper.


Copyright 2024 Marcus Häggbom
