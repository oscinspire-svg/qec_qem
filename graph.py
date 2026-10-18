"""Decoding graph: turns the fault table + noise rates into a weighted matching graph."""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .circuit import CHANNELS
from .frame import FaultTable


def _xor_prob(a: float, b: float) -> float:
    return a + b - 2 * a * b


class DecodingGraph:
    """Nodes 0..n-1 are detectors; node n is the boundary.

    edges: list of (u, v, p, obs); v == -1 means the boundary.
    dist[i, j] / par[i, j]: shortest-path weight and logical-observable parity between nodes.
    """

    def __init__(self, n_detectors: int, edges: List[Tuple[int, int, float, int]]):
        self.n = n_detectors
        self.edges = edges
        self._all_pairs()

    # ---------------------------------------------------------------- build
    @classmethod
    def from_faults(cls, table: FaultTable, n_detectors: int, rates: Dict[str, float]):
        rate = np.array([rates[c] for c in CHANNELS], dtype=float)
        pf = table.frac * rate[table.channel]
        if np.any(table.undetectable_logical & (pf > 0)):
            raise ValueError("the circuit has an undetectable single fault that flips the "
                             "logical observable (effective distance 1)")
        acc: Dict[Tuple[int, int, int], float] = {}
        for f in np.flatnonzero(pf > 0):
            p = float(pf[f])
            for key in table.pieces[f]:
                acc[key] = _xor_prob(acc.get(key, 0.0), p)
        # parallel edges with different observable effect: keep the likelier one
        best: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for (u, v, o), p in acc.items():
            cur = best.get((u, v))
            if cur is None or p > cur[0]:
                best[(u, v)] = (p, o)
        edges = [(u, v, p, o) for (u, v), (p, o) in sorted(best.items())]
        return cls(n_detectors, edges)

    @staticmethod
    def weight(p: float) -> float:
        p = min(max(p, 1e-15), 1 - 1e-15)
        return max(math.log((1 - p) / p), 0.0)

    def _all_pairs(self) -> None:
        n = self.n
        B = n
        adj: List[List[Tuple[int, float, int]]] = [[] for _ in range(n + 1)]
        for u, v, p, o in self.edges:
            v = B if v < 0 else v
            w = self.weight(p)
            adj[u].append((v, w, o))
            adj[v].append((u, w, o))
        inf = float("inf")
        dist = np.full((n + 1, n + 1), inf)
        par = np.zeros((n + 1, n + 1), dtype=np.uint8)
        for s in range(n + 1):
            d = [inf] * (n + 1)
            pr = [0] * (n + 1)
            d[s] = 0.0
            heap = [(0.0, s)]
            while heap:
                du, u = heapq.heappop(heap)
                if du > d[u]:
                    continue
                for v, w, o in adj[u]:
                    nd = du + w
                    if nd < d[v] - 1e-12:
                        d[v] = nd
                        pr[v] = pr[u] ^ o
                        heapq.heappush(heap, (nd, v))
            dist[s] = d
            par[s] = pr
        self.dist, self.par = dist, par

    # -------------------------------------------------------------- queries
    def effective_distance(self) -> int:
        """Fewest edges in a boundary-to-boundary path that flips the observable."""
        n = self.n
        B = n
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(n + 1)]
        for u, v, _p, o in self.edges:
            v = B if v < 0 else v
            adj[u].append((v, o))
            adj[v].append((u, o))
        seen = {(B, 0): 0}
        q = deque([(B, 0)])
        while q:
            u, par = q.popleft()
            for v, o in adj[u]:
                st = (v, par ^ o)
                if st not in seen:
                    seen[st] = seen[(u, par)] + 1
                    q.append(st)
        return seen.get((B, 1), -1)

    def to_text(self) -> str:
        """Stim-like listing: error(p) D.. L0"""
        lines = []
        for u, v, p, o in self.edges:
            tail = f"D{u}" + ("" if v < 0 else f" D{v}") + (" L0" if o else "")
            lines.append(f"error({p:.6g}) {tail}")
        return "\n".join(lines)
