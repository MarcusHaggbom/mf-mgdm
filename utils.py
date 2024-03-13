import os
import numpy as np
import torch
import pandas as pd
from types import ModuleType


def is_torch_else_numpy(x) -> bool:
    if torch.is_tensor(x):
        return True
    if isinstance(x, (np.ndarray, np.number)):
        return False
    raise NotImplementedError('Must be either torch.tensor or numpy.ndarray')


def lib(tensor_or_array: np.ndarray | torch.Tensor) -> ModuleType:
    return torch if is_torch_else_numpy(tensor_or_array) else np


def next_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        numb = 1
        while True:
            new_path = '{0} ({2}){1}'.format(*os.path.splitext(file_path) + (numb,))
            if os.path.exists(new_path):
                numb += 1
            else:
                return new_path
    return file_path


def load_real_data(s: str) -> np.ndarray:
    if s == 'SP500':
        df = pd.read_csv(os.path.join('data', 'sp500.csv'), sep=';', thousands=',', index_col=0, parse_dates=True)
        df = df.sort_index(ascending=True)
        log_rets = np.diff(np.log(df['Adj Close**'].values))
        return (log_rets - log_rets.mean()) / log_rets.std()

    rates_map = {'EUR3M': 'EU 3 Months',
                 'EUR6M': 'EU 6 Months',
                 'EUR5Y': 'EU 5 Year',
                 'EUR10Y': 'EU 10 Year',
                 'SEK10Y': 'SE GVB 10 Year',
                 'SEK2Y': 'SE GVB 2 Year',
                 'SEK5Y': 'SE GVB 5 Year',
                 'SEK7Y': 'SE GVB 7 Year',
                 'USD10Y': 'US 10 Year',
                 'USD3M': 'US 3 Months',
                 'USD5Y': 'US 5 Year',
                 'USD6M': 'US 6 Months'}
    if s in rates_map:
        df = pd.read_csv(os.path.join('data', 'rates.csv'),
                         sep=';', index_col=None, parse_dates=['Date'], dayfirst=True)
        df = df[df['Series'] == rates_map[s]]
        df.index = df['Date']
        df = df.sort_index(ascending=True)
        rets = np.diff(df['Value'].dropna().values)
        return (rets - rets.mean()) / rets.std()

    raise ValueError(s)
