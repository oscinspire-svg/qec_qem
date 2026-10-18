"""Result dataclasses. Everything serializes to JSON via RunResult.to_json()."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PointResult:
    """One simulated noise scale (lambda)."""
    lam: float
    noise: Optional[Dict[str, float]]   # the four scaled noise rates used
    max_noise_rate: Optional[float]
    rounds: Optional[int]
    seed: int                           # seed used for this point
    shots: int
    failures: int
    p_L: float
    stderr: float
    stats: Dict[str, int]               # num_qubits, num_detectors, num_observables, ...
    circuit_text: str
    dem_text: str
    decoder: str
    decoder_info: Dict[str, Any] = field(default_factory=dict)  # effective distance, solver counts
    raw: Optional[Dict[str, Any]] = None  # det_events, obs_flips, predictions


@dataclass
class FitResult:
    model: str
    quantity: str
    target: float
    value: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    boot_std: Optional[float]
    p_L_equivalent: Optional[float]     # same as value when quantity is the error rate
    params: Dict[str, float]
    param_cov: Optional[List[List[float]]]
    chi2: float
    dof: int
    reduced_chi2: Optional[float]
    n_points_used: int
    n_boot_valid: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class ZNEResult:
    primary_model: str
    value: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    fits: List[FitResult]
    model_spread: Optional[float]       # max - min across models (systematic error)
    failed_models: Dict[str, str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class QubitReport:
    budget: int
    physical_qubits_per_patch: int      # num_qubits of the generated circuit
    patches_that_fit: int               # budget // per patch
    qubits_in_use: int
    unused_qubits: int


@dataclass
class RunResult:
    config: Dict[str, Any]
    versions: Dict[str, Optional[str]]
    seed: int
    qubit_report: Optional[QubitReport]
    points: List[PointResult]
    zne: Optional[ZNEResult]            # None if the extrapolation could not be fitted
    zne_error: Optional[str] = None     # reason, when zne is None

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as fh:
                fh.write(text)
        return text


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)
