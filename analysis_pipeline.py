"""
UToE 2.1 — Preregistered Logistic–Scalar Adversarial Test
Confirmatory analysis pipeline

OSF Preregistration DOI: 10.17605/OSF.IO/HV7S5
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from sklearn.metrics import r2_score

# -----------------------------
# Core model definitions
# -----------------------------

def logistic_scalar(t, r, a, phi_max):
    """
    Logistic–scalar growth model.
    a = lambda * gamma (effective growth driver)
    """
    return phi_max / (1 + np.exp(-r * a * (t - np.mean(t))))

def linear_model(t, m, b):
    return m * t + b

# -----------------------------
# Φ(t) construction
# -----------------------------

def construct_phi(R):
    """
    Deterministic construction of scalar integration Φ(t)
    from empirical observable R(t).

    Preregistered constraint:
    - Min–max normalization to [0, 1]
    """
    R = np.asarray(R)
    phi = (R - np.min(R)) / (np.max(R) - np.min(R))
    return phi

# -----------------------------
# Phase randomization control
# -----------------------------

def phase_randomize(x, seed=0):
    """
    Phase-randomized surrogate preserving amplitude spectrum
    and destroying temporal structure.
    """
    rng = np.random.default_rng(seed)
    X = fft(x)
    phases = np.exp(1j * rng.uniform(0, 2*np.pi, len(X)))
    X_rand = np.abs(X) * phases
    return np.real(ifft(X_rand))

# -----------------------------
# Model fitting and metrics
# -----------------------------

def fit_and_score(model, t, y, p0):
    popt, _ = curve_fit(model, t, y, p0=p0, maxfev=10000)
    y_hat = model(t, *popt)

    rss = np.sum((y - y_hat)**2)
    n = len(y)
    k = len(popt)

    aic = n * np.log(rss / n) + 2 * k
    adj_r2 = 1 - (1 - r2_score(y, y_hat)) * (n - 1) / (n - k - 1)

    return aic, adj_r2, popt

# -----------------------------
# Confirmatory pipeline
# -----------------------------

def run_confirmatory_test(R):
    """
    Executes preregistered adversarial comparison:
    M1 (Logistic–Scalar) vs M2 (Linear)
    """

    phi = construct_phi(R)
    t = np.arange(len(phi))

    # Fit models
    aic_log, r2_log, _ = fit_and_score(
        logistic_scalar, t, phi, p0=[0.1, 1.0, 1.0]
    )

    aic_lin, r2_lin, _ = fit_and_score(
        linear_model, t, phi, p0=[0.0, 0.5]
    )

    delta_aic = aic_log - aic_lin
    delta_r2 = r2_log - r2_lin

    return delta_aic, delta_r2

# -----------------------------
# Phase-randomized robustness test
# -----------------------------

def run_phase_control(R):
    R_rand = phase_randomize(R)
    return run_confirmatory_test(R_rand)

# -----------------------------
# Entry point (example stub)
# -----------------------------

if __name__ == "__main__":
    # Placeholder example signal
    # Real datasets are loaded externally via MNE/OpenNeuro
    np.random.seed(0)
    R_example = np.cumsum(np.random.randn(500)) + np.linspace(0, 10, 500)

    daic, dr2 = run_confirmatory_test(R_example)
    daic_rand, dr2_rand = run_phase_control(R_example)

    print("Confirmatory results:")
    print("ΔAIC =", daic)
    print("ΔAdjR² =", dr2)

    print("\nPhase-randomized control:")
    print("ΔAIC =", daic_rand)
    print("ΔAdjR² =", dr2_rand)
