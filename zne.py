"""Zero-noise extrapolation of the logical error rate (or Pauli expectation)."""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .config import ZNEArgs
from .results import FitResult, ZNEResult


class FitError(ValueError):
    pass


# --- weighted least squares (numpy only) ----------------------------------
def _wls(x, y, sigma, deg):
    A = np.vander(x, deg + 1, increasing=True)
    w = 1.0 / sigma
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, y * w, rcond=None)
    cov = np.linalg.pinv(Aw.T @ Aw)
    return coef, cov


# --- models: each returns (predict(t), params, cov) -----------------------
def _fit_powerlaw(lam, y, s):
    if np.any(y <= 0):
        raise FitError("power law needs positive values")
    (a, k), cov = _wls(np.log(lam), np.log(y), s / y, 1)
    pred = lambda t: np.exp(a + k * np.log(t))
    return pred, {"log_A": a, "A": float(np.exp(a)), "exponent": k}, cov


def _fit_exponential(lam, y, s):
    if np.any(y <= 0):
        raise FitError("exponential needs positive values")
    (a, b), _ = _wls(lam, np.log(y), s / y, 1)
    f = lambda t, A, B: A * np.exp(B * np.asarray(t, dtype=float))
    try:
        popt, pcov = curve_fit(f, lam, y, p0=[np.exp(a), b], sigma=s,
                               absolute_sigma=True, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        raise FitError(f"exponential fit failed: {exc}")
    return (lambda t: f(t, *popt)), {"A": popt[0], "b": popt[1]}, pcov


def _fit_polynomial(lam, y, s, order):
    coef, cov = _wls(lam, y, s, order)
    pred = lambda t: np.polyval(coef[::-1], t)
    return pred, {f"c{i}": c for i, c in enumerate(coef)}, cov


def _fitter(model: str, order: int) -> Tuple[Callable, int]:
    if model == "powerlaw":
        return _fit_powerlaw, 2
    if model == "exponential":
        return _fit_exponential, 2
    if model == "polynomial":
        return partial(_fit_polynomial, order=order), order + 1
    raise ValueError(f"unknown model {model!r} (use powerlaw, exponential, polynomial)")


# --- data prep ------------------------------------------------------------
def _prepare(lams, failures, shots, quantity):
    p = failures / shots
    sp = np.sqrt(p * (1 - p) / shots)
    mask = (failures > 0) & (failures < shots)  # zero-variance points can't be weighted
    if quantity == "logical_error_rate":
        return p, sp, mask
    if quantity == "expectation":
        return 1 - 2 * p, 2 * sp, mask
    raise ValueError("quantity must be 'logical_error_rate' or 'expectation'")


def _bootstrap(fit, npar, lams, failures, shots, target, z: ZNEArgs, rng):
    p_hat = failures / shots
    vals = []
    for _ in range(z.n_bootstrap):
        fb = rng.binomial(shots, p_hat)
        y, s, m = _prepare(lams, fb, shots, z.quantity)
        if m.sum() <= npar:
            continue
        try:
            pred, _, _ = fit(lams[m], y[m], s[m])
            v = float(pred(target))
        except (FitError, np.linalg.LinAlgError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals)


def extrapolate(lams, failures, shots, z: ZNEArgs,
                max_noise_rate: Optional[float] = None) -> ZNEResult:
    """Fit each requested model and extrapolate to z.target.

    lams, failures, shots are per-lambda arrays. max_noise_rate is the largest
    scaled physical rate that was simulated (for the threshold guard).
    """
    lams = np.asarray(lams, dtype=float)
    failures = np.asarray(failures, dtype=np.int64)
    shots = np.asarray(shots, dtype=np.int64)
    rng = np.random.default_rng(z.bootstrap_seed)
    warnings: List[str] = []
    fits: List[FitResult] = []
    failed: Dict[str, str] = {}

    y_all, s_all, mask = _prepare(lams, failures, shots, z.quantity)
    if (~mask).any():
        dropped = lams[~mask].tolist()
        warnings.append(f"dropped lambdas {dropped}: zero or all-shots failures "
                        "(no variance estimate). Use more shots or larger lambda.")
    if max_noise_rate is not None and max_noise_rate > z.threshold_guard:
        warnings.append(f"largest simulated physical rate {max_noise_rate:.3g} exceeds "
                        f"threshold_guard {z.threshold_guard:g}: the fit may be biased "
                        "(above/near threshold p_L no longer follows the low-noise law).")
    lo, hi = lams[mask].min() if mask.any() else lams.min(), lams[mask].max() if mask.any() else lams.max()
    if z.target < lo and hi > lo and (lo - z.target) / (hi - lo) > 2:
        warnings.append(f"long extrapolation: target {z.target:g} is {(lo - z.target) / (hi - lo):.1f}x "
                        "the sampled lambda range below it.")
    if z.target > hi:
        warnings.append("target lies above the sampled lambda range (this is not extrapolation to lower noise).")

    for model in z.models:
        try:
            if model == "powerlaw":
                if z.quantity != "logical_error_rate":
                    raise FitError("power law applies to the logical error rate, not the expectation")
                if z.target <= 0:
                    raise FitError("power law cannot be evaluated at target <= 0 "
                                   "(p_L -> 0 trivially); use a target > 0")
            fit, npar = _fitter(model, z.poly_order if z.poly_order is not None else 2)
            n = int(mask.sum())
            if n <= npar:
                raise FitError(f"need more than {npar} usable points, have {n}")
            pred, params, cov = fit(lams[mask], y_all[mask], s_all[mask])
            value = float(pred(z.target))
            resid = (y_all[mask] - pred(lams[mask])) / s_all[mask]
            chi2 = float(np.sum(resid ** 2))
            dof = n - npar
            red = chi2 / dof if dof > 0 else None
            boot = _bootstrap(fit, npar, lams, failures, shots, z.target, z, rng)
            fw: List[str] = []
            lo_ci = hi_ci = bstd = None
            if len(boot) >= max(10, z.n_bootstrap // 2):
                a = (1 - z.confidence) / 2 * 100
                lo_ci, hi_ci = (float(v) for v in np.percentile(boot, [a, 100 - a]))
                bstd = float(boot.std())
            else:
                fw.append(f"only {len(boot)}/{z.n_bootstrap} bootstrap fits succeeded; no CI reported")
            if red is not None and red > 3:
                fw.append(f"poor fit: reduced chi2 = {red:.1f}")
            p_eq = value if z.quantity == "logical_error_rate" else (1 - value) / 2
            fits.append(FitResult(
                model=model, quantity=z.quantity, target=z.target, value=value,
                ci_low=lo_ci, ci_high=hi_ci, boot_std=bstd, p_L_equivalent=float(p_eq),
                params={k: float(v) for k, v in params.items()},
                param_cov=np.asarray(cov).tolist(), chi2=chi2, dof=dof,
                reduced_chi2=red, n_points_used=n, n_boot_valid=int(len(boot)), warnings=fw))
        except (FitError, np.linalg.LinAlgError) as exc:
            failed[model] = str(exc)

    if not fits:
        raise FitError(f"no model could be fitted: {failed}")
    primary = fits[0]
    spread = None
    if len(fits) > 1:
        v = [f.value for f in fits]
        spread = float(max(v) - min(v))
    for m, why in failed.items():
        warnings.append(f"model {m!r} skipped: {why}")
    return ZNEResult(primary_model=primary.model, value=primary.value,
                     ci_low=primary.ci_low, ci_high=primary.ci_high, fits=fits,
                     model_spread=spread, failed_models=failed, warnings=warnings)
