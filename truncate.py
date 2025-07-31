import pandas as pd
from statsmodels.tsa.stattools import ccf
from scipy.signal import correlate
from scipy.signal import hilbert
import numpy as np


def cross_corr(x, y, max_lag):
    x = x - x.mean()
    y = y - y.mean()
    full = correlate(y, x, mode='full')
    lags = np.arange(-len(x)+1, len(x))
    mask = np.abs(lags) <= max_lag
    Corr = full[mask] / np.sqrt(np.sum(x**2)*np.sum(y**2))
    return Corr


def delay_correlation_truncate(dynamics, maxlag, threshold =0.8):
    to_drop = set()
    for i in range(dynamics.shape[0]):
        if i in to_drop:
            i = i+1
        for j in range(i+1, dynamics.shape[0]):
            corr = cross_corr(dynamics[i], dynamics[j], maxlag)
            if np.max(np.abs(corr)) > threshold and i not in to_drop:
                to_drop.add(j)
                
    df_dynamics = pd.DataFrame(dynamics)
    df_new_dynamics = df_dynamics.drop(df_dynamics.index[list(to_drop)])
    print(to_drop)
    return df_new_dynamics
            
    


def truncate_plv(dynamics, modes, threshold=0.8):
    to_drop = set()
    for i in range(dynamics.shape[0]):
        if i in to_drop:
            i = i+1
        for j in range(i + 1, dynamics.shape[0]):
            difference_dynamics = np.abs(hilbert(dynamics[i].real) - hilbert(dynamics[j].real))
            plv_dym = np.abs(np.mean(np.exp(1j * difference_dynamics)))
            if (plv_dym > threshold) and i not in to_drop:
                to_drop.add(j)

    df_dynamics = pd.DataFrame(dynamics)
    df_new_dynamics = df_dynamics.drop(df_dynamics.index[list(to_drop)])
    df_modes = pd.DataFrame(modes)
    df_new_modes = df_modes.drop(df_modes.index[list(to_drop)])
    print(to_drop)
    return df_new_dynamics, df_new_modes


def conjugate_truncate(dynamics, modes, lambdas):
    r = len(lambdas)
    pairs = {}
    for i in range(r):
        
        for j in range(i, r):
            eigen_match = np.allclose(lambdas[i], np.conj(lambdas[j]))
            dynamics_match = np.allclose(dynamics[i], np.conj(dynamics[j]))
            mode_match = np.allclose(modes[:, i], np.conj(modes[:, j]))
            if (
                eigen_match
                and dynamics_match
                and mode_match
                and i != j
                and j not in pairs
                and j not in pairs.values()
            ):
                pairs[i] = j

        if i not in pairs and i not in pairs.values():
            pairs[i] = i

    k = len(pairs)
    new_lambdas = np.zeros(k, dtype=complex)
    new_dynamics = np.zeros((k, dynamics.shape[1]), dtype=complex)
    new_modes = np.zeros((modes.shape[0], k), dtype=complex)

    j = 0
    for i in pairs:
        new_lambdas[j] = lambdas[i]
        new_dynamics[j] = dynamics[i]
        new_modes[:, j] = modes[:, i]
        j += 1
    return new_dynamics



def correlation_truncate(all_dynamics, threshold = 0.8):
    df = pd.DataFrame(all_dynamics)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)

def autocorrelation_truncate(all_dynamics, maxlag = 1, threshold = 0.9):
    df = pd.DataFrame(all_dynamics)
    keep_cols = []
    for col in df.columns:
        series = df[col].dropna().values
        if len(series) <= maxlag:
            continue
        x1 = series[:-maxlag]
        x2 = series[maxlag:]
        corr = np.corrcoef(x1, x2)[0, 1]
        if abs(corr) >= threshold:
            keep_cols.append(col)
    return df[keep_cols]