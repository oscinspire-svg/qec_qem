"""Build the circuit and error model, sample, decode, count failures (pure Python + numpy)."""
from __future__ import annotations

import functools
from typing import Callable, Dict, Optional

import numpy as np

from .circuit import build_program, parse_task
from .config import Config, default_scale
from .decoder import MatchingDecoder
from .frame import enumerate_faults, sample_batch
from .graph import DecodingGraph
from .results import PointResult


@functools.lru_cache(maxsize=16)
def _fault_table(distance: int, rounds: int, basis: str):
    """Which detectors each possible fault flips. Independent of the noise rates."""
    return enumerate_faults(build_program(distance, rounds, basis))


def rates_from_noise(noise: Dict[str, float]) -> Dict[str, float]:
    return {"clifford": noise["after_clifford_depolarization"],
            "data": noise["before_round_data_depolarization"],
            "measure": noise["before_measure_flip_probability"],
            "reset": noise["after_reset_flip_probability"]}


def simulate_point(cfg: Config, lam: float, seed: int,
                   pre_check: Optional[Callable[[Dict[str, int]], None]] = None) -> PointResult:
    a = cfg.circuit
    basis, rounds = parse_task(a.code_task), a.resolved_rounds()
    noise = (cfg.noise_scale or default_scale)(a, lam)
    rates = rates_from_noise(noise)
    for k, v in rates.items():
        if not 0 <= v <= 1:
            raise ValueError(f"scaled noise rate for {k!r} is {v:g}, outside [0, 1]")

    program = build_program(a.distance, rounds, basis)
    stats = program.stats()
    if pre_check is not None:
        pre_check(stats)

    table = _fault_table(a.distance, rounds, basis)
    graph = DecodingGraph.from_faults(table, len(program.detectors), rates)
    if cfg.decoder.decoder_factory is not None:
        decoder, dname = cfg.decoder.decoder_factory(graph), "custom"
    else:
        decoder = MatchingDecoder(graph, cfg.decoder.max_exact_component, cfg.decoder.max_states)
        dname = "matching"

    rng = np.random.default_rng(seed)
    shots, batch = cfg.sampler.shots, cfg.sampler.batch_size
    failures, done = 0, 0
    raw = {"det_events": [], "obs_flips": [], "predictions": []}
    while done < shots:
        n = min(batch, shots - done)
        det, obs = sample_batch(program, rates, n, rng)
        pred = np.asarray(decoder.decode_batch(det))
        failures += int(np.count_nonzero(np.any(pred != obs, axis=1)))
        done += n
        if cfg.store_raw:
            raw["det_events"].append(det)
            raw["obs_flips"].append(obs)
            raw["predictions"].append(pred)

    p = failures / shots
    info = {"effective_distance": graph.effective_distance(),
            "n_graph_edges": len(graph.edges)}
    if isinstance(decoder, MatchingDecoder):
        info.update(unique_syndromes_decoded=decoder.n_unique,
                    approximate_clusters=decoder.n_approximate)
    return PointResult(
        lam=float(lam), noise=dict(noise), max_noise_rate=max(noise.values()),
        rounds=rounds, seed=seed, shots=shots, failures=failures, p_L=p,
        stderr=float(np.sqrt(p * (1 - p) / shots)), stats=stats,
        circuit_text=program.to_text(rates), dem_text=graph.to_text(), decoder=dname,
        decoder_info=info,
        raw={k: np.concatenate(v) for k, v in raw.items()} if cfg.store_raw else None)
