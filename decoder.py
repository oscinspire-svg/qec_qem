"""Minimum-weight perfect matching decoder (exact for small syndromes, pure Python)."""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np

from .graph import DecodingGraph


class _TooBig(Exception):
    pass


class MatchingDecoder:
    """Pairs up detection events (or sends them to the boundary) at minimum total weight,
    and predicts the logical-observable flip from the chosen paths.

    The problem is split into independent clusters: two events are only worth pairing
    if the path between them is cheaper than sending both to the boundary. Each cluster is
    solved exactly by dynamic programming over the events in time order, which only visits
    states reachable through nearby pairings. A cluster with more than ``max_exact`` events,
    or one needing more than ``max_states`` DP states, falls back to a greedy matching
    (counted in ``n_approximate`` and reported with a warning).
    """

    def __init__(self, graph: DecodingGraph, max_exact: int = 60, max_states: int = 200_000):
        self.graph = graph
        self.max_exact = int(max_exact)
        self.max_states = int(max_states)
        n = graph.n
        self.bd = graph.dist[:n, n]          # distance to boundary
        self.bp = graph.par[:n, n]           # observable parity to boundary
        self._cache: Dict[bytes, int] = {}
        self.n_approximate = 0
        self.n_unique = 0
        self._warned = False

    # ---------------------------------------------------------------- core
    def _clusters(self, ev: List[int]) -> List[List[int]]:
        g = self.graph
        parent = {e: e for e in ev}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for ai in range(len(ev)):
            for bi in range(ai + 1, len(ev)):
                i, j = ev[ai], ev[bi]
                if g.dist[i, j] < self.bd[i] + self.bd[j] - 1e-12:
                    parent[find(i)] = find(j)
        groups: Dict[int, List[int]] = {}
        for e in ev:
            groups.setdefault(find(e), []).append(e)
        return list(groups.values())

    def _solve_exact(self, nodes: List[int]) -> int:
        g = self.graph
        n = len(nodes)
        b = [float(self.bd[i]) for i in nodes]
        bp = [int(self.bp[i]) for i in nodes]
        pair: List[List[Tuple[int, float, int]]] = [[] for _ in range(n)]
        for a in range(n):
            for c in range(a + 1, n):
                w = float(g.dist[nodes[a], nodes[c]])
                if w < b[a] + b[c] - 1e-12:
                    pair[a].append((c, w, int(g.par[nodes[a], nodes[c]])))
        memo: Dict[int, Tuple[float, int]] = {0: (0.0, 0)}

        def f(mask: int) -> Tuple[float, int]:
            hit = memo.get(mask)
            if hit is not None:
                return hit
            low = (mask & -mask).bit_length() - 1
            rest = mask & ~(1 << low)
            c0, p0 = f(rest)
            best = (c0 + b[low], p0 ^ bp[low])
            for c, w, par in pair[low]:
                if rest >> c & 1:
                    cc, pp = f(rest & ~(1 << c))
                    if cc + w < best[0] - 1e-12:
                        best = (cc + w, pp ^ par)
            memo[mask] = best
            if len(memo) > self.max_states:
                raise _TooBig
            return best

        return f((1 << n) - 1)[1]

    def _solve_greedy(self, nodes: List[int]) -> int:
        g = self.graph
        cand = []
        for a in range(len(nodes)):
            for c in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[c]
                w = g.dist[i, j]
                if w < self.bd[i] + self.bd[j] - 1e-12:
                    cand.append((w - self.bd[i] - self.bd[j], i, j))
        cand.sort()
        used, parity = set(), 0
        for _s, i, j in cand:
            if i not in used and j not in used:
                used.update((i, j))
                parity ^= int(g.par[i, j])
        for i in nodes:
            if i not in used:
                parity ^= int(self.bp[i])
        return parity

    def decode_events(self, ev: List[int]) -> int:
        parity = 0
        for cl in self._clusters(ev):
            cl.sort()
            if len(cl) <= self.max_exact:
                try:
                    parity ^= self._solve_exact(cl)
                    continue
                except _TooBig:
                    pass
            self.n_approximate += 1
            parity ^= self._solve_greedy(cl)
            if not self._warned:
                self._warned = True
                warnings.warn(f"a cluster of {len(cl)} detection events was too large for "
                              "the exact solver (max_exact_component / max_states); greedy "
                              "matching was used for it (slightly less accurate).")
        return parity

    # --------------------------------------------------------------- batch
    def decode_batch(self, det_events: np.ndarray) -> np.ndarray:
        """det_events: bool array (shots, n_detectors). Returns uint8 (shots, 1)."""
        det_events = np.asarray(det_events, dtype=bool)
        shots, nd = det_events.shape
        pred = np.zeros((shots, 1), dtype=np.uint8)
        rows = np.flatnonzero(det_events.any(axis=1))
        if rows.size == 0:
            return pred
        packed = np.packbits(det_events[rows], axis=1)
        uniq, first, inv = np.unique(packed, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        out = np.zeros(len(uniq), dtype=np.uint8)
        for u in range(len(uniq)):
            key = uniq[u].tobytes()
            hit = self._cache.get(key)
            if hit is None:
                ev = np.flatnonzero(det_events[rows[first[u]]]).tolist()
                hit = self.decode_events(ev)
                self._cache[key] = hit
                self.n_unique += 1
            out[u] = hit
        pred[rows, 0] = out[inv]
        return pred
