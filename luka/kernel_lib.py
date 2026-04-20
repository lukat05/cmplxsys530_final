from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import numpy as np


Array = np.ndarray


def _as_1d(a: Array, name: str) -> Array:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    return a


def kernel_matrix(
    hosts: Dict[str, Array],
    kernel_fn: Callable[[Array, Array, Dict[str, Any]], Array],
    theta: Dict[str, Any],
    t: float = 0.0,
    mask_self: bool = True,
) -> Array:
    """
    Return full NxN matrix Lambda where Lambda[i,j] = K_theta(x_i, x_j, t).

    Parameters
    ----------
    hosts : dict
        Must contain 'x' (1D, length N). May contain 'g' (1D int group labels).
    kernel_fn : callable
        Vectorized kernel: kernel_fn(x_i, x_j, theta) -> NxN array.
    theta : dict
        Kernel parameters.
    t : float
        Time (unused for K0-K3, but included for signature compatibility).
    mask_self : bool
        If True, set diagonal to 0 (no self-infection).
    """
    x = _as_1d(hosts["x"], "hosts['x']")
    # broadcast to NxN
    xi = x[:, None]
    xj = x[None, :]

    lam = kernel_fn(xi, xj, theta, hosts=hosts, t=t)
    lam = np.asarray(lam, dtype=float)

    if lam.shape != (x.size, x.size):
        raise ValueError(f"kernel_fn returned shape {lam.shape}, expected {(x.size, x.size)}")

    if mask_self:
        np.fill_diagonal(lam, 0.0)

    # guard against tiny negatives from numerical issues
    if np.any(lam < -1e-12):
        raise ValueError("Kernel produced negative intensities.")
    lam = np.clip(lam, 0.0, None)
    return lam


# ---------- Kernels ----------

def K0_homogeneous(xi: Array, xj: Array, theta: Dict[str, Any], *, hosts=None, t: float = 0.0) -> Array:
    """
    Homogeneous mixing: K(i,j) = beta / N
    theta: {'beta': float}
    """
    beta = float(theta["beta"])
    if hosts is None or "x" not in hosts:
        raise ValueError("hosts with key 'x' required to infer N.")
    N = int(np.asarray(hosts["x"]).size)
    return (beta / N) * np.ones(np.broadcast(xi, xj).shape, dtype=float)


def K1_block_mixing(xi: Array, xj: Array, theta: Dict[str, Any], *, hosts=None, t: float = 0.0) -> Array:
    """
    Block/group mixing: K(i,j) = B[g_i, g_j]
    theta: {'B': (G,G) array}
    hosts must include 'g' integer labels in {0..G-1} (recommended).
    """
    if hosts is None or "g" not in hosts:
        raise ValueError("hosts['g'] required for block mixing kernel K1.")
    g = _as_1d(hosts["g"], "hosts['g']")
    B = np.asarray(theta["B"], dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"theta['B'] must be square (G,G), got {B.shape}")

    gi = g[:, None]
    gj = g[None, :]

    if gi.min() < 0 or gj.min() < 0 or gi.max() >= B.shape[0] or gj.max() >= B.shape[0]:
        raise ValueError("Group labels out of range for B.")

    return B[gi, gj]


def K2_scalar_similarity(xi: Array, xj: Array, theta: Dict[str, Any], *, hosts=None, t: float = 0.0) -> Array:
    """
    Scalar similarity: K(i,j) = beta * exp(-alpha * |x_i - x_j|)
    theta: {'beta': float, 'alpha': float}
    """
    beta = float(theta["beta"])
    alpha = float(theta["alpha"])
    return beta * np.exp(-alpha * np.abs(xi - xj))


def K3_assortative_power(xi: Array, xj: Array, theta: Dict[str, Any], *, hosts=None, t: float = 0.0) -> Array:
    """
    Generalization of K2: K(i,j) = beta * f(|x_i-x_j|; alpha, p)
    We use: f(d) = exp(-alpha * d^p)
    - p > 1: more assortative (sharp decay with distance)
    - 0 < p < 1: more disassortative / heavy-tailed decay

    theta: {'beta': float, 'alpha': float, 'p': float}
    """
    beta = float(theta["beta"])
    alpha = float(theta["alpha"])
    p = float(theta["p"])
    if p <= 0:
        raise ValueError("p must be > 0 for K3.")
    d = np.abs(xi - xj)
    return beta * np.exp(-alpha * np.power(d, p))


# Convenience registry
KERNELS = {
    "K0": K0_homogeneous,
    "K1": K1_block_mixing,
    "K2": K2_scalar_similarity,
    "K3": K3_assortative_power,
}
