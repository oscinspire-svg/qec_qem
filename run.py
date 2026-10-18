"""run(config): simulate each noise scale, then extrapolate."""
from __future__ import annotations

import dataclasses
import importlib.metadata as md
import math
import platform
import secrets
from typing import Dict, Optional

from . import __version__
from .config import BudgetError, Config
from .results import QubitReport, RunResult
from .simulate import simulate_point
from .zne import FitError, extrapolate


def _versions() -> Dict[str, Optional[str]]:
    out = {"qec_qem": __version__, "python": platform.python_version()}
    for pkg in ("numpy", "scipy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def run(cfg: Config) -> RunResult:
    cfg.validate()
    base_seed = cfg.sampler.seed if cfg.sampler.seed is not None else secrets.randbits(32)
    report: Dict[str, QubitReport] = {}

    def check_budget(stats):
        if cfg.qubit_budget is None:
            return
        per = stats["num_qubits"]
        if per > cfg.qubit_budget:
            raise BudgetError(f"circuit needs {per} physical qubits but the budget is "
                              f"{cfg.qubit_budget}; lower the distance or raise the budget")
        n = cfg.qubit_budget // per
        report["r"] = QubitReport(budget=cfg.qubit_budget, physical_qubits_per_patch=per,
                                  patches_that_fit=n, qubits_in_use=n * per,
                                  unused_qubits=cfg.qubit_budget - n * per)

    points = []
    for i, lam in enumerate(cfg.lambdas):   # seed + i: independent samples per noise scale
        points.append(simulate_point(cfg, lam, base_seed + i,
                                     pre_check=check_budget if i == 0 else None))

    z = cfg.zne
    if z.poly_order is None:
        z = dataclasses.replace(z, poly_order=max(1, math.ceil(cfg.circuit.distance / 2)))
    zres, zerr = None, None
    try:
        zres = extrapolate([p.lam for p in points], [p.failures for p in points],
                           [p.shots for p in points], z,
                           max_noise_rate=max(p.max_noise_rate for p in points))
    except FitError as exc:   # keep the simulated points even if the fit is impossible
        zerr = str(exc)
    return RunResult(config=cfg.to_dict(), versions=_versions(), seed=base_seed,
                     qubit_report=report.get("r"), points=points, zne=zres, zne_error=zerr)
