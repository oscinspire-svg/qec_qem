"""Configuration: one section per stage of the pipeline."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

# The four noise rates of the circuit; ZNE scales these by lambda.
NOISE_FIELDS = (
    "after_clifford_depolarization",
    "before_round_data_depolarization",
    "before_measure_flip_probability",
    "after_reset_flip_probability",
)


class BudgetError(ValueError):
    """The circuit needs more physical qubits than the user's budget."""


@dataclass
class CircuitArgs:
    """The rotated surface-code memory circuit."""
    code_task: str = "rotated_memory_z"   # or "rotated_memory_x" ("surface_code:" prefix allowed)
    distance: int = 3                     # odd, >= 3
    rounds: Optional[int] = None          # syndrome rounds; None -> same as distance
    after_clifford_depolarization: float = 1e-3     # DEPOLARIZE1/2 after H and CX gates
    before_round_data_depolarization: float = 1e-3  # DEPOLARIZE1 on data qubits each round
    before_measure_flip_probability: float = 1e-3   # measurement flip error
    after_reset_flip_probability: float = 1e-3      # reset flip error

    def resolved_rounds(self) -> int:
        return self.distance if self.rounds is None else self.rounds


@dataclass
class SamplerArgs:
    seed: Optional[int] = None   # None -> random; the seed actually used is recorded
    shots: int = 100_000         # shots per noise scale
    batch_size: int = 50_000     # shots simulated at once (limits memory)


@dataclass
class DecoderArgs:
    decoder: str = "matching"    # built-in minimum-weight matching decoder
    # Custom decoder: callable(graph) -> object with decode_batch(det_events) -> (shots, 1) array.
    decoder_factory: Optional[Callable[[Any], Any]] = None
    max_exact_component: int = 60      # largest event cluster solved exactly
    max_states: int = 200_000          # DP-state cap per cluster before greedy fallback


@dataclass
class ZNEArgs:
    """Extrapolation settings."""
    target: float = 1.0   # noise scale to extrapolate to (1.0 = the user's base rates)
    models: Tuple[str, ...] = ("powerlaw",)   # first is primary; others give a model spread
    poly_order: Optional[int] = None  # polynomial degree; None -> ceil(distance / 2)
    quantity: str = "logical_error_rate"      # or "expectation" (= 1 - 2 p_L)
    n_bootstrap: int = 500
    confidence: float = 0.95
    threshold_guard: float = 0.007   # warn if any scaled physical rate exceeds this
    bootstrap_seed: Optional[int] = None


def default_scale(args: CircuitArgs, lam: float) -> Dict[str, float]:
    """Default noise scaling: multiply all four rates by lambda."""
    return {name: getattr(args, name) * lam for name in NOISE_FIELDS}


@dataclass
class Config:
    circuit: CircuitArgs = field(default_factory=CircuitArgs)
    sampler: SamplerArgs = field(default_factory=SamplerArgs)
    decoder: DecoderArgs = field(default_factory=DecoderArgs)
    zne: ZNEArgs = field(default_factory=ZNEArgs)

    lambdas: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)  # noise scales to simulate
    qubit_budget: Optional[int] = None
    store_raw: bool = False   # keep detection events / observable flips / predictions
    noise_scale: Optional[Callable[[CircuitArgs, float], Dict[str, float]]] = None

    def validate(self) -> None:
        from .circuit import parse_task
        if not self.lambdas:
            raise ValueError("lambdas must not be empty")
        if any(l <= 0 for l in self.lambdas):
            raise ValueError("all lambdas must be > 0")
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("lambdas must be distinct")
        c, s, d = self.circuit, self.sampler, self.decoder
        parse_task(c.code_task)
        if not isinstance(c.distance, int) or c.distance < 3 or c.distance % 2 == 0:
            raise ValueError("circuit.distance must be an odd integer >= 3")
        if c.rounds is not None and (not isinstance(c.rounds, int) or c.rounds < 1):
            raise ValueError("circuit.rounds must be an integer >= 1")
        for name in NOISE_FIELDS:
            v = getattr(c, name)
            if not 0 <= v <= 1:
                raise ValueError(f"circuit.{name} must be in [0, 1]")
        if s.shots <= 0:
            raise ValueError("sampler.shots must be > 0")
        if s.batch_size <= 0:
            raise ValueError("sampler.batch_size must be > 0")
        if d.decoder_factory is None and d.decoder != "matching":
            raise ValueError("decoder must be 'matching' (or pass decoder_factory)")
        if d.max_exact_component <= 0 or d.max_states <= 0:
            raise ValueError("decoder limits must be > 0")
        if self.qubit_budget is not None and self.qubit_budget <= 0:
            raise ValueError("qubit_budget must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly copy; callables become repr strings."""
        def section(obj):
            out = {}
            for f in dataclasses.fields(obj):
                v = getattr(obj, f.name)
                out[f.name] = repr(v) if callable(v) else v
            return out
        out = {n: section(getattr(self, n)) for n in ("circuit", "sampler", "decoder", "zne")}
        out.update(lambdas=list(self.lambdas), qubit_budget=self.qubit_budget,
                   store_raw=self.store_raw,
                   noise_scale=None if self.noise_scale is None else repr(self.noise_scale))
        return out
