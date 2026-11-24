"""Temporal autocorrelation and spatial correlations with missing heights."""
import numpy as np


def autocorrelation(values, max_lag: int | None = None) -> np.ndarray:
    """FFT autocorrelation, normalized by the lag-zero sum of squares."""
    x = np.asarray(values,float)
    if x.ndim != 1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('Require a finite one-dimensional trajectory')
    x = x-x.mean()
    if not np.any(x):
        raise ValueError('Constant trajectory has undefined correlation')
    n = len(x)
    lag = n-1 if max_lag is None else max_lag
    if not 0 <= lag < n:
        raise ValueError('Invalid maximum lag')
    fft = np.fft.rfft(x,n=2*n)
    ac = np.fft.irfft(fft*fft.conj(),n=2*n)[:lag+1]
    return ac/ac[0]


def integrated_time(rho) -> float:
    """Truncate at first nonpositive adjacent-lag pair; time in sample intervals."""
    rho = np.asarray(rho,float)
    if rho.ndim != 1 or not len(rho) or not np.isfinite(rho).all() or not np.isclose(rho[0],1):
        raise ValueError('Require normalized autocorrelation')
    total = 0.
    for i in range(0,len(rho)-1,2):
        pair = rho[i]+rho[i+1]
        if pair <= 0:
            break
        total += pair
    return max(.5,total-.5)


def height_correlation(profiles, max_lag: int) -> list[dict]:
    """Pool valid ray pairs across profiles; center using pooled valid heights."""
    x = np.asarray(profiles,float)
    if x.ndim != 2 or not 0 <= max_lag < x.shape[1] or np.isinf(x).any() or not np.isfinite(x).any():
        raise ValueError('Require rectangular height profiles and valid lag')
    x = x-np.nanmean(x)
    variance = np.nanmean(x*x)
    rows = []
    for lag in range(max_lag+1):
        a,b = x[:,:x.shape[1]-lag],x[:,lag:]
        valid = np.isfinite(a)&np.isfinite(b)
        count = int(valid.sum())
        cov = float((a[valid]*b[valid]).mean()) if count else None
        rows.append(dict(lag=lag,pairs=count,covariance=cov,correlation=cov/variance if count and variance>0 else None))
    return rows
