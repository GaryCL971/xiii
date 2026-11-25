"""Daily circular block resampling and compounded high-water-mark risk."""
from __future__ import annotations

import math

import numpy as np

N_PATHS = 2000
BLOCK_DAYS = 20
SEED = 13
BREACH_DD = -0.10


def block_bootstrap(
    rendements: np.ndarray,
    n_chemins: int = N_PATHS,
    taille_bloc: int = BLOCK_DAYS,
    graine: int | np.random.Generator = SEED,
) -> np.ndarray:
    """Draw circular blocks, retaining the finite daily sample length."""
    r = np.asarray(rendements, dtype=float)
    if r.ndim != 1:
        raise ValueError("Daily returns must be a one-dimensional series.")
    if n_chemins < 1 or taille_bloc < 1:
        raise ValueError("The number of paths and block size must be positive.")
    r = r[np.isfinite(r)]
    n = r.size
    if n < taille_bloc * 2:
        raise ValueError(
            f"Série trop courte ({n}) pour des blocs de {taille_bloc}. "
            "Réduire la taille de bloc fausserait la préservation de l'autocorrélation."
        )

    rng = np.random.default_rng(graine)
    starts = rng.integers(0, n, size=(n_chemins, math.ceil(n / taille_bloc)))
    indices = (starts[:, :, None] + np.arange(taille_bloc)) % n
    return r[indices.reshape(n_chemins, -1)[:, :n]]


def courbe_equite(rendements: np.ndarray) -> np.ndarray:
    """Compound along the last axis with initial equity 1 and absorbing ruin."""
    r = np.asarray(rendements, dtype=float)
    return np.cumprod(np.maximum(0.0, 1.0 + np.nan_to_num(r, nan=0.0)), axis=-1)


def path_drawdowns(equity: np.ndarray) -> np.ndarray:
    """Drawdown at each step, including the starting balance in the peak."""
    peaks = np.maximum(1.0, np.maximum.accumulate(equity, axis=-1))
    return equity / peaks - 1.0


def breach_paths(equity: np.ndarray, threshold: float = BREACH_DD) -> np.ndarray:
    """Test in equity units so an exact touch is not lost to subtractive rounding."""
    if threshold >= 0:
        raise ValueError("seuil_dd doit être négatif.")
    peaks = np.maximum(1.0, np.maximum.accumulate(equity, axis=-1))
    return (equity <= peaks * (1.0 + threshold)).any(axis=-1)


def probabilite_breach_chemin(chemins: np.ndarray, seuil_dd: float = BREACH_DD) -> float:
    """Fraction of paths touching the high-water-mark barrier at any step."""
    return float(breach_paths(courbe_equite(chemins), seuil_dd).mean())
