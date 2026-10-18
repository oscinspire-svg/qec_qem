"""Rotated surface-code memory circuit, generated in pure Python.

Layout (d odd, d >= 3): data qubits sit on a d x d grid (index b*d + a); parity
checks ("plaquettes") sit at the corners between them, on a (d+1) x (d+1) grid of
centres (i, j). Interior plaquettes alternate X/Z in a checkerboard; weight-2
boundary plaquettes are X-type on the top/bottom edges and Z-type on the
left/right edges. That gives d*d data + (d*d - 1) check qubits = 2d^2 - 1.

Each round: data depolarizing noise, H on X-checks, four layers of CX gates
(X-checks control the data, data control the Z-checks), H again, then
measure + reset every check qubit. The gate order uses the "Z" and "N" shapes
so that hook errors do not shorten the effective code distance.

Noise channels (the same four as Stim's generated circuits):
  data     DEPOLARIZE1 on data qubits at the start of every round
  clifford DEPOLARIZE1 after H gates, DEPOLARIZE2 after each CX
  measure  basis-flip error right before each measurement
  reset    basis-flip error right after each reset
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

CHANNELS = ("clifford", "data", "measure", "reset")

# (da, db) offset from plaquette centre (i, j) to the data qubit touched in each CX layer.
X_ORDER = ((-1, -1), (0, -1), (-1, 0), (0, 0))   # last two data are a horizontal pair
Z_ORDER = ((-1, -1), (-1, 0), (0, -1), (0, 0))   # last two data are a vertical pair


@dataclass
class Plaquette:
    index: int
    i: int
    j: int
    kind: str                  # "X" or "Z"
    layers: Tuple[int, ...]    # data qubit touched in each of the 4 CX layers (-1 = none)


@dataclass
class Op:
    name: str                  # H CX R M X_ERROR Z_ERROR DEPOLARIZE1 DEPOLARIZE2
    qubits: np.ndarray         # shape (k,) or, for CX / DEPOLARIZE2, shape (k, 2)
    channel: str = ""          # noise channel (noise ops only)
    basis: str = "z"           # R / M basis (affects what a measurement records)

    @property
    def is_noise(self) -> bool:
        return self.name in ("X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2")


@dataclass
class Program:
    distance: int
    rounds: int
    basis: str
    n_qubits: int
    n_data: int
    plaquettes: List[Plaquette]
    ops: List[Op]
    n_measurements: int
    detectors: List[List[int]]     # each detector = XOR of these measurement indices
    observable: List[int]          # logical observable = XOR of these measurement indices

    def stats(self) -> dict:
        return {"num_qubits": self.n_qubits,
                "num_detectors": len(self.detectors),
                "num_observables": 1,
                "num_measurements": self.n_measurements,
                "num_ticks": sum(1 for o in self.ops if not o.is_noise)}

    def to_text(self, rates=None) -> str:
        """Human-readable, Stim-like listing of the circuit."""
        names = {"X_ERROR": "X_ERROR", "Z_ERROR": "Z_ERROR",
                 "DEPOLARIZE1": "DEPOLARIZE1", "DEPOLARIZE2": "DEPOLARIZE2"}
        lines = []
        for op in self.ops:
            flat = " ".join(str(int(q)) for q in np.asarray(op.qubits).reshape(-1))
            if op.is_noise:
                p = "" if rates is None else f"({rates[op.channel]:.6g})"
                lines.append(f"{names[op.name]}{p} {flat}")
            elif op.name in ("R", "M") and op.basis == "x":
                lines.append(f"{op.name}X {flat}")
            else:
                lines.append(f"{op.name} {flat}")
        for k, det in enumerate(self.detectors):
            lines.append(f"DETECTOR D{k} = " + " ^ ".join(f"rec[{m}]" for m in det))
        lines.append("OBSERVABLE L0 = " + " ^ ".join(f"rec[{m}]" for m in self.observable))
        return "\n".join(lines)


def parse_task(code_task: str) -> str:
    """Return the memory basis ('z' or 'x') for a code task name."""
    name = code_task.split(":")[-1].strip()
    if name == "rotated_memory_z":
        return "z"
    if name == "rotated_memory_x":
        return "x"
    raise ValueError(f"unsupported code_task {code_task!r}; "
                     "use 'rotated_memory_z' or 'rotated_memory_x' "
                     "(optionally prefixed with 'surface_code:')")


def make_plaquettes(d: int) -> List[Plaquette]:
    out: List[Plaquette] = []
    for j in range(d + 1):
        for i in range(d + 1):
            if i in (0, d) and j in (0, d):
                continue                                  # corners: weight 1, not used
            kind = "X" if (i + j) % 2 == 0 else "Z"
            interior = 0 < i < d and 0 < j < d
            if not interior:
                if j in (0, d) and kind != "X":
                    continue                              # top/bottom boundary: X only
                if i in (0, d) and kind != "Z":
                    continue                              # left/right boundary: Z only
            order = X_ORDER if kind == "X" else Z_ORDER
            layers = []
            for da, db in order:
                a, b = i + da, j + db
                layers.append(b * d + a if 0 <= a < d and 0 <= b < d else -1)
            out.append(Plaquette(len(out), i, j, kind, tuple(layers)))
    return out


@functools.lru_cache(maxsize=32)
def build_program(distance: int, rounds: int, basis: str) -> Program:
    d = distance
    if not isinstance(d, int) or d < 3 or d % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be an integer >= 1")
    if basis not in ("x", "z"):
        raise ValueError("basis must be 'x' or 'z'")

    plaqs = make_plaquettes(d)
    n_data, n_anc = d * d, len(plaqs)
    nq = n_data + n_anc
    data = np.arange(n_data)
    anc = np.arange(n_data, nq)
    xanc = np.array([n_data + p.index for p in plaqs if p.kind == "X"])
    flip = "X_ERROR" if basis == "z" else "Z_ERROR"       # error that flips the data basis

    ops: List[Op] = []
    ops.append(Op("R", anc, basis="z"))
    ops.append(Op("R", data, basis=basis))
    ops.append(Op("X_ERROR", anc, "reset"))
    ops.append(Op(flip, data, "reset"))

    for _ in range(rounds):
        ops.append(Op("DEPOLARIZE1", data, "data"))
        ops.append(Op("H", xanc))
        ops.append(Op("DEPOLARIZE1", xanc, "clifford"))
        for layer in range(4):
            pairs = []
            for p in plaqs:
                dq = p.layers[layer]
                if dq < 0:
                    continue
                a = n_data + p.index
                pairs.append((a, dq) if p.kind == "X" else (dq, a))
            pairs = np.array(pairs, dtype=np.int64)
            ops.append(Op("CX", pairs))
            ops.append(Op("DEPOLARIZE2", pairs, "clifford"))
        ops.append(Op("H", xanc))
        ops.append(Op("DEPOLARIZE1", xanc, "clifford"))
        ops.append(Op("X_ERROR", anc, "measure"))
        ops.append(Op("M", anc, basis="z"))
        ops.append(Op("R", anc, basis="z"))
        ops.append(Op("X_ERROR", anc, "reset"))

    ops.append(Op(flip, data, "measure"))
    ops.append(Op("M", data, basis=basis))

    # measurement index layout: round r check k -> r*n_anc + k ; final data q -> rounds*n_anc + q
    det_kind = "Z" if basis == "z" else "X"
    detectors: List[List[int]] = []
    for r in range(rounds):
        for p in plaqs:
            if p.kind != det_kind:
                continue
            idx = [r * n_anc + p.index]
            if r > 0:
                idx.append((r - 1) * n_anc + p.index)
            detectors.append(idx)
    for p in plaqs:
        if p.kind != det_kind:
            continue
        idx = [rounds * n_anc + q for q in p.layers if q >= 0]
        idx.append((rounds - 1) * n_anc + p.index)
        detectors.append(idx)

    if basis == "z":      # logical Z = Z on the bottom row of data qubits
        obs = [rounds * n_anc + (0 * d + a) for a in range(d)]
    else:                 # logical X = X on the left column of data qubits
        obs = [rounds * n_anc + (b * d + 0) for b in range(d)]

    return Program(d, rounds, basis, nq, n_data, plaqs, ops,
                   rounds * n_anc + n_data, detectors, obs)
