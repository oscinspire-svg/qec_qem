"""Pauli-frame simulation (numpy, vectorised over shots).

Stabiliser circuits propagate Pauli errors deterministically, so a shot is
simulated by tracking only the error ("frame") on each qubit. A detector fires
when the XOR of the measurement flips it covers is 1.

Two modes share one engine:
  * sampling    - random faults, many shots, returns detection events.
  * enumeration - one shot per possible single fault, used to learn which
                  detectors each fault flips (the error model for decoding).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .circuit import CHANNELS, Program


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def _execute(program: Program, shots: int, noise_fn) -> np.ndarray:
    nq = program.n_qubits
    X = np.zeros((nq, shots), dtype=bool)
    Z = np.zeros((nq, shots), dtype=bool)
    rec = np.zeros((program.n_measurements, shots), dtype=bool)
    m = 0
    for k, op in enumerate(program.ops):
        n = op.name
        if n == "H":
            q = op.qubits
            tmp = X[q].copy()
            X[q] = Z[q]
            Z[q] = tmp
        elif n == "CX":
            c, t = op.qubits[:, 0], op.qubits[:, 1]
            X[t] ^= X[c]
            Z[c] ^= Z[t]
        elif n == "R":
            X[op.qubits] = False
            Z[op.qubits] = False
        elif n == "M":
            q = op.qubits
            rec[m:m + len(q)] = (X if op.basis == "z" else Z)[q]
            m += len(q)
        else:
            noise_fn(k, op, X, Z)
    return rec


def _detectors_and_obs(program: Program, rec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    shots = rec.shape[1]
    det = np.zeros((len(program.detectors), shots), dtype=bool)
    for i, idx in enumerate(program.detectors):
        det[i] = np.bitwise_xor.reduce(rec[idx], axis=0)
    obs = np.bitwise_xor.reduce(rec[program.observable], axis=0)
    return det, obs


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def _events(rng: np.random.Generator, trials: int, p: float) -> np.ndarray:
    """Indices (in 0..trials-1) of independent Bernoulli(p) successes."""
    if p <= 0 or trials == 0:
        return np.empty(0, dtype=np.int64)
    if p >= 1:
        return np.arange(trials, dtype=np.int64)
    if p > 0.05:
        return np.flatnonzero(rng.random(trials) < p)
    n = int(rng.binomial(trials, p))
    if n == 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(trials, size=n, replace=False).astype(np.int64)


def _make_sampling_noise(rates: Dict[str, float], rng: np.random.Generator, shots: int):
    def fn(k, op, X, Z):
        p = rates[op.channel]
        if p <= 0:
            return
        qs = op.qubits
        if op.name in ("X_ERROR", "Z_ERROR", "DEPOLARIZE1"):
            idx = _events(rng, len(qs) * shots, p)
            if idx.size == 0:
                return
            q, s = qs[idx // shots], idx % shots
            if op.name == "X_ERROR":
                X[q, s] ^= True
            elif op.name == "Z_ERROR":
                Z[q, s] ^= True
            else:
                r = rng.integers(0, 3, size=idx.size)     # 0=X 1=Y 2=Z
                X[q, s] ^= (r < 2)
                Z[q, s] ^= (r > 0)
        else:  # DEPOLARIZE2: 15 equally likely non-identity two-qubit Paulis
            idx = _events(rng, len(qs) * shots, p)
            if idx.size == 0:
                return
            pair, s = idx // shots, idx % shots
            r = rng.integers(1, 16, size=idx.size)
            p1, p2 = r >> 2, r & 3                        # bit0 = X part, bit1 = Z part
            q1, q2 = qs[pair, 0], qs[pair, 1]
            X[q1, s] ^= (p1 & 1).astype(bool)
            Z[q1, s] ^= ((p1 >> 1) & 1).astype(bool)
            X[q2, s] ^= (p2 & 1).astype(bool)
            Z[q2, s] ^= ((p2 >> 1) & 1).astype(bool)
    return fn


def sample_batch(program: Program, rates: Dict[str, float], shots: int,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (detection_events[shots, n_det] bool, observable_flips[shots, 1] bool)."""
    rec = _execute(program, shots, _make_sampling_noise(rates, rng, shots))
    det, obs = _detectors_and_obs(program, rec)
    return np.ascontiguousarray(det.T), obs.reshape(-1, 1)


# --------------------------------------------------------------------------- #
# fault enumeration (the error model)
# --------------------------------------------------------------------------- #
@dataclass
class FaultTable:
    n_faults: int
    channel: np.ndarray            # channel index (into CHANNELS) per fault
    frac: np.ndarray               # fraction of the channel probability (1, 1/3, 1/15)
    pieces: List[List[Tuple[int, int, int]]]   # graph-like pieces (u, v, obs); v=-1 is boundary
    undetectable_logical: np.ndarray           # bool per fault: flips logical, no detector


def _enumerate(program: Program):
    channel_of, frac_of, factor = [], [], []
    comps: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    base = 0
    chan_index = {c: i for i, c in enumerate(CHANNELS)}
    for k, op in enumerate(program.ops):
        if not op.is_noise:
            continue
        qs = op.qubits
        ci = chan_index[op.channel]
        if op.name in ("X_ERROR", "Z_ERROR"):
            m = len(qs)
            rows = base + np.arange(m)
            x = np.full(m, op.name == "X_ERROR")
            comps[k] = [(rows, qs, x, ~x)]
            channel_of += [ci] * m
            frac_of += [1.0] * m
            factor += [(-1, -1)] * m
            base += m
        elif op.name == "DEPOLARIZE1":
            m = len(qs)
            lst = []
            for t, (xb, zb) in enumerate(((1, 0), (1, 1), (0, 1))):
                rows = base + 3 * np.arange(m) + t
                lst.append((rows, qs, np.full(m, bool(xb)), np.full(m, bool(zb))))
            comps[k] = lst
            channel_of += [ci] * (3 * m)
            frac_of += [1 / 3] * (3 * m)
            factor += [(-1, -1)] * (3 * m)
            base += 3 * m
        else:  # DEPOLARIZE2
            m = len(qs)
            lst = []
            fac = np.full((m * 15, 2), -1, dtype=np.int64)
            for r in range(1, 16):
                p1, p2 = r >> 2, r & 3
                rows = base + 15 * np.arange(m) + (r - 1)
                if p1:
                    lst.append((rows, qs[:, 0], np.full(m, bool(p1 & 1)), np.full(m, bool(p1 & 2))))
                if p2:
                    lst.append((rows, qs[:, 1], np.full(m, bool(p2 & 1)), np.full(m, bool(p2 & 2))))
                if p1 and p2:       # remember the two single-qubit factors of this fault
                    fac[15 * np.arange(m) + (r - 1), 0] = base + 15 * np.arange(m) + (p1 * 4 - 1)
                    fac[15 * np.arange(m) + (r - 1), 1] = base + 15 * np.arange(m) + (p2 - 1)
            comps[k] = lst
            channel_of += [ci] * (15 * m)
            frac_of += [1 / 15] * (15 * m)
            factor += [tuple(f) for f in fac]
            base += 15 * m
    return base, np.array(channel_of), np.array(frac_of), np.array(factor), comps


def _decompose(dets: Tuple[int, ...], obs: int, known: Dict[Tuple[int, ...], set]):
    """Split a >2-detector fault into known graph-like pieces (XOR obs must match)."""
    dets = list(dets)

    def rec(rest, need):
        if not rest:
            return [] if need == 0 else None
        first = rest[0]
        for partner in [None] + rest[1:]:
            part = (first,) if partner is None else (first, partner)
            if part not in known:
                continue
            rem = [r for r in rest if r not in part]
            for o in known[part]:
                sub = rec(rem, need ^ o)
                if sub is not None:
                    return [(part, o)] + sub
        return None

    return rec(dets, obs)


def enumerate_faults(program: Program) -> FaultTable:
    n_faults, channel, frac, factor, comps = _enumerate(program)

    def noise_fn(k, op, X, Z):
        for rows, qs, xb, zb in comps.get(k, []):
            X[qs, rows] ^= xb
            Z[qs, rows] ^= zb

    rec = _execute(program, n_faults, noise_fn)
    det, obs = _detectors_and_obs(program, rec)
    det_t = np.ascontiguousarray(det.T)
    dets = [tuple(int(i) for i in np.flatnonzero(det_t[f])) for f in range(n_faults)]
    obs = obs.astype(int)

    known: Dict[Tuple[int, ...], set] = {}
    for f in range(n_faults):
        if 0 < len(dets[f]) <= 2:
            known.setdefault(dets[f], set()).add(int(obs[f]))

    def piece(D, o):
        return (D[0], D[1] if len(D) == 2 else -1, int(o))

    pieces: List[List[Tuple[int, int, int]]] = []
    undetectable = np.zeros(n_faults, dtype=bool)
    for f in range(n_faults):
        D, o = dets[f], int(obs[f])
        if len(D) == 0:
            undetectable[f] = bool(o)
            pieces.append([])
        elif len(D) <= 2:
            pieces.append([piece(D, o)])
        else:
            parts = None
            f1, f2 = int(factor[f][0]), int(factor[f][1])
            if f1 >= 0 and len(dets[f1]) <= 2 and len(dets[f2]) <= 2:
                parts = [(dets[f1], int(obs[f1])), (dets[f2], int(obs[f2]))]
                parts = [(d_, o_) for d_, o_ in parts if len(d_) > 0]
            if parts is None:
                parts = _decompose(D, o, known)
            if parts is None:
                raise ValueError("a fault flips more than two detectors and could not be "
                                 "decomposed into graph-like pieces")
            pieces.append([piece(d_, o_) for d_, o_ in parts])
    return FaultTable(n_faults, channel, frac, pieces, undetectable)
