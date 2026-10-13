"""
M-HySMap v0.7: Activity-weighted multicast hypergraph mapping for SNNs on mesh NoCs.

v0.7 release: scaling benchmark suite on top of incremental joint M-HySMap. The full method no longer
freezes the row-major placement during hypergraph refinement; it periodically refreshes
multicast-aware placement and then moves neurons under that current physical layout.

This is the minimum paper-core implementation:
  - directed SNN graph -> one multicast hyperedge per source neuron
  - profiled or synthetic activity weights r_u
  - objective based on distinct destination cores, multicast routed hops, and link congestion
  - local hyper-FM style refinement
  - local placement swap refinement

The implementation is intentionally self-contained and conservative. It is meant as a clean
research prototype, not yet a hyper-optimized production partitioner.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Sequence, Tuple, Optional
import math
import time
import heapq

import numpy as np
import scipy.sparse as sp

POPS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]


def potjans_mini_sizes(scale: float = 0.006) -> Dict[str, int]:
    """Small Potjans-inspired population sizes. The proportions match the internship prototype."""
    base = {
        "L23E": 0.125, "L23I": 0.031,
        "L4E":  0.170, "L4I":  0.043,
        "L5E":  0.085, "L5I":  0.021,
        "L6E":  0.200, "L6I":  0.050,
    }
    total = max(8, int(77000 * scale))
    return {name: max(1, int(total * frac)) for name, frac in base.items()}


def potjans_example_pij() -> np.ndarray:
    """Representative directed connection probabilities from the internship prototype."""
    p = np.array([
        [0.08, 0.12, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01],
        [0.06, 0.15, 0.02, 0.03, 0.02, 0.03, 0.01, 0.02],
        [0.03, 0.05, 0.10, 0.14, 0.02, 0.02, 0.02, 0.01],
        [0.02, 0.06, 0.05, 0.12, 0.02, 0.03, 0.01, 0.02],
        [0.02, 0.03, 0.02, 0.02, 0.09, 0.13, 0.03, 0.02],
        [0.02, 0.04, 0.02, 0.03, 0.06, 0.10, 0.02, 0.03],
        [0.02, 0.03, 0.02, 0.02, 0.03, 0.03, 0.07, 0.12],
        [0.01, 0.03, 0.01, 0.02, 0.02, 0.03, 0.05, 0.11],
    ], dtype=float)
    return np.clip(p, 0.0, 0.3)


def build_potjans_mini_directed(scale: float = 0.006, seed: int = 1) -> Tuple[sp.csr_matrix, List[Tuple[str, int, int]]]:
    """Return directed CSR adjacency and population ranges."""
    rng = np.random.RandomState(seed)
    sizes = potjans_mini_sizes(scale)
    pij = potjans_example_pij()

    starts: Dict[str, int] = {}
    cur = 0
    for name in POPS:
        starts[name] = cur
        cur += sizes[name]
    n = cur
    groups = [(name, starts[name], starts[name] + sizes[name]) for name in POPS]

    A = np.zeros((n, n), dtype=np.uint8)
    for i, src in enumerate(POPS):
        s1, e1 = starts[src], starts[src] + sizes[src]
        for j, dst in enumerate(POPS):
            p = pij[i, j]
            if p <= 0:
                continue
            s2, e2 = starts[dst], starts[dst] + sizes[dst]
            A[s1:e1, s2:e2] = (rng.rand(e1 - s1, e2 - s2) < p).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return sp.csr_matrix(A), groups


def make_population_activity(groups: List[Tuple[str, int, int]], seed: int = 1, eps: float = 0.05) -> np.ndarray:
    """
    Synthetic activity profile for smoke tests.
    Replace this with spike_count/T_profile once warmup simulation is wired in.
    """
    rng = np.random.RandomState(seed)
    n = max(e for _, _, e in groups)
    rates = np.zeros(n, dtype=float)

    # Hz-ish relative rates; deliberately heterogeneous so activity-aware mapping has signal.
    base_rate = {
        "L23E": 7.0, "L23I": 15.0,
        "L4E":  9.0, "L4I":  18.0,
        "L5E":  5.0, "L5I":  13.0,
        "L6E":  4.0, "L6I":  10.0,
    }
    for name, s, e in groups:
        mu = base_rate.get(name, 5.0)
        # lognormal heterogeneity around the population mean
        vals = rng.lognormal(mean=math.log(max(mu, eps)), sigma=0.35, size=e - s)
        rates[s:e] = np.maximum(vals, eps)
    return rates


# -----------------------------------------------------------------------------
# Activity profiling: turn the static SNN graph into profiled or synthetic activity weights r_u
# -----------------------------------------------------------------------------

def population_ranges(groups: List[Tuple[str, int, int]]) -> Dict[str, Tuple[int, int]]:
    """Return {population_name: (start, end)} from the Potjans-style group list."""
    return {name: (int(s), int(e)) for name, s, e in groups}


def population_id_vector(groups: List[Tuple[str, int, int]]) -> np.ndarray:
    """Map each neuron id to its population index in POPS."""
    n = max(e for _, _, e in groups)
    pop_id = np.full(n, -1, dtype=np.int32)
    name_to_id = {name: i for i, name in enumerate(POPS)}
    for name, s, e in groups:
        pop_id[s:e] = name_to_id[name]
    return pop_id


def poisson_spikes_per_neuron(neuron_ids: Iterable[int], T: float, rate_hz: float, rng: Optional[np.random.RandomState] = None) -> List[Tuple[float, int]]:
    """Generate homogeneous Poisson input spikes as (time, source_neuron)."""
    if rng is None:
        rng = np.random.RandomState(1)
    events: List[Tuple[float, int]] = []
    lam = float(rate_hz)
    for src in neuron_ids:
        t = 0.0
        while True:
            t += float(rng.exponential(1.0 / max(lam, 1e-12)))
            if t > T:
                break
            events.append((t, int(src)))
    events.sort(key=lambda x: x[0])
    return events


def build_profile_synapses(
    A_dir: sp.csr_matrix,
    groups: List[Tuple[str, int, int]],
    w_exc: float = 0.34,
    w_inh: float = -0.48,
    d_base: float = 0.001,
    d_jitter: float = 0.00025,
    seed: int = 0,
) -> List[List[Tuple[int, float, float]]]:
    """
    Build directed synapse lists for activity profiling.

    The source population decides the sign: E sources excite, I sources inhibit.
    This is deliberately lightweight: the goal is not biological validation, but obtaining
    a repeatable activity vector r_u for the mapping objective.
    """
    rng = np.random.RandomState(seed)
    A = A_dir.tocsr()
    pop_id = population_id_vector(groups)
    pop_names = np.array(POPS, dtype=object)
    outs: List[List[Tuple[int, float, float]]] = [[] for _ in range(A.shape[0])]
    for u in range(A.shape[0]):
        src_pop = str(pop_names[pop_id[u]])
        sign_w = float(w_inh if src_pop.endswith('I') else w_exc)
        for t in range(A.indptr[u], A.indptr[u + 1]):
            v = int(A.indices[t])
            if u == v:
                continue
            delay = float(d_base + rng.rand() * d_jitter)
            outs[u].append((v, sign_w, delay))
    return outs


@dataclass
class ProfileStats:
    T: float
    input_rate_hz: float
    external_events: int
    internal_spikes: int
    total_recorded_spikes: int
    active_neurons: int
    mean_rate_hz: float
    max_rate_hz: float
    pop_counts: Dict[str, int]

    def as_dict(self) -> Dict[str, float]:
        d = {
            "profile_T": float(self.T),
            "profile_input_rate_hz": float(self.input_rate_hz),
            "profile_external_events": float(self.external_events),
            "profile_internal_spikes": float(self.internal_spikes),
            "profile_total_recorded_spikes": float(self.total_recorded_spikes),
            "profile_active_neurons": float(self.active_neurons),
            "profile_mean_rate_hz": float(self.mean_rate_hz),
            "profile_max_rate_hz": float(self.max_rate_hz),
        }
        for k, v in self.pop_counts.items():
            d[f"profile_pop_{k}"] = float(v)
        return d


def profile_activity_event_driven(
    A_dir: sp.csr_matrix,
    groups: List[Tuple[str, int, int]],
    T: float = 0.250,
    input_rate_hz: float = 8.0,
    drive_pops: Tuple[str, ...] = ("L4E", "L23E"),
    seed: int = 0,
    eps_rate_hz: float = 0.05,
    V_th: float = 1.0,
    V_reset: float = 0.0,
    tau_ref: float = 0.002,
    w_exc: float = 0.34,
    w_inh: float = -0.48,
    d_base: float = 0.001,
    d_jitter: float = 0.00025,
    include_external_in_rate: bool = True,
    max_events: int = 250_000,
) -> Tuple[np.ndarray, ProfileStats]:
    """
    Estimate r_u by running a short event-driven warmup simulation.

    External Poisson spikes force selected input populations to fire. Their cascades are
    processed one at a time with a simple jump-and-fire threshold/reset/refractory model.
    The returned r_u is spike_count(u)/T with a tiny epsilon floor so temporarily silent
    structural sources are not erased from the hypergraph objective.
    """
    rng = np.random.RandomState(seed)
    n = A_dir.shape[0]
    outs = build_profile_synapses(A_dir, groups, w_exc=w_exc, w_inh=w_inh, d_base=d_base, d_jitter=d_jitter, seed=seed + 17)
    ranges = population_ranges(groups)

    drivers: List[int] = []
    for pop in drive_pops:
        if pop in ranges:
            s, e = ranges[pop]
            drivers.extend(range(s, e))
    if not drivers:
        drivers = list(range(n))

    external = poisson_spikes_per_neuron(drivers, T=T, rate_hz=input_rate_hz, rng=rng)

    V = np.zeros(n, dtype=float)
    ref_until = np.full(n, -1e9, dtype=float)
    counts = np.zeros(n, dtype=np.int64)
    internal_spikes = 0
    events_processed = 0

    for cause_id, (t_emit, src) in enumerate(external):
        if include_external_in_rate:
            counts[src] += 1
        q: List[Tuple[float, int, float, int]] = []
        for j, w, d in outs[src]:
            heapq.heappush(q, (t_emit + d, int(j), float(w), int(cause_id)))
        while q:
            events_processed += 1
            if events_processed > max_events:
                # Safety valve against pathological recurrent bursts.
                q.clear()
                break
            t, j, w, cause = heapq.heappop(q)
            if t > T or t < ref_until[j]:
                continue
            V[j] += w
            if V[j] >= V_th:
                counts[j] += 1
                internal_spikes += 1
                V[j] = V_reset
                ref_until[j] = t + tau_ref
                for k, w2, d2 in outs[j]:
                    heapq.heappush(q, (t + d2, int(k), float(w2), int(cause)))

    raw_rates = counts.astype(float) / max(float(T), 1e-12)
    rates = np.maximum(raw_rates, float(eps_rate_hz))

    pop_counts: Dict[str, int] = {}
    for name, s, e in groups:
        pop_counts[name] = int(counts[s:e].sum())

    stats = ProfileStats(
        T=float(T),
        input_rate_hz=float(input_rate_hz),
        external_events=int(len(external)),
        internal_spikes=int(internal_spikes),
        total_recorded_spikes=int(counts.sum()),
        active_neurons=int(np.count_nonzero(counts)),
        mean_rate_hz=float(raw_rates.mean()),
        max_rate_hz=float(raw_rates.max() if raw_rates.size else 0.0),
        pop_counts=pop_counts,
    )
    return rates, stats


def csr_out_neighbors(A: sp.csr_matrix) -> List[np.ndarray]:
    A = A.tocsr()
    return [A.indices[A.indptr[u]:A.indptr[u + 1]].astype(np.int32, copy=False) for u in range(A.shape[0])]


def csr_in_neighbors(A: sp.csr_matrix) -> List[np.ndarray]:
    return csr_out_neighbors(A.transpose().tocsr())


def grid_coords(rows: int, cols: int) -> List[Tuple[int, int]]:
    return [(x, y) for y in range(rows) for x in range(cols)]


def manhattan_path(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...]:
    """Deterministic XY path. Links are stored directed by travel direction."""
    x, y = a
    x2, y2 = b
    path = []
    step_x = 1 if x2 > x else -1
    while x != x2:
        nx = x + step_x
        path.append(((x, y), (nx, y)))
        x = nx
    step_y = 1 if y2 > y else -1
    while y != y2:
        ny = y + step_y
        path.append(((x, y), (x, ny)))
        y = ny
    return tuple(path)


def balanced_random_labels(n: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    labels = np.repeat(np.arange(k, dtype=np.int32), math.ceil(n / k))[:n]
    rng.shuffle(labels)
    return labels


def labels_to_parts(labels: np.ndarray) -> Dict[int, List[int]]:
    parts: Dict[int, List[int]] = defaultdict(list)
    for u, c in enumerate(labels):
        parts[int(c)].append(int(u))
    return dict(parts)


def edge_cut_cost(out_neighbors: List[np.ndarray], labels: np.ndarray, rates: Optional[np.ndarray] = None) -> float:
    """Directed edge-cut cost. If rates is given, edge u->v is weighted by source rate r_u."""
    total = 0.0
    if rates is None:
        for u, outs in enumerate(out_neighbors):
            if outs.size:
                total += float(np.count_nonzero(labels[outs] != labels[u]))
    else:
        for u, outs in enumerate(out_neighbors):
            if outs.size:
                total += float(rates[u]) * float(np.count_nonzero(labels[outs] != labels[u]))
    return total


@dataclass
class MeshObjectiveWeights:
    alpha_hops: float = 1.0
    beta_max_link: float = 0.10
    gamma_link_var: float = 0.01
    delta_size_imbalance: float = 0.0
    delta_rate_imbalance: float = 0.0


@dataclass
class MeshMetrics:
    edge_cut: float
    weighted_edge_cut: float
    remote_fanout: float
    multicast_hops: float
    max_link_load: float
    link_load_variance: float
    packet_count: float
    size_imbalance: float
    rate_imbalance: float
    objective: float

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def multicast_metrics(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: np.ndarray,
    perm: Optional[np.ndarray] = None,
    rows: int = 4,
    cols: int = 4,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
) -> MeshMetrics:
    """
    Evaluate activity-weighted multicast mapping.

    Each source neuron u contributes one multicast tree to its distinct remote destination cores.
    We approximate the tree as the union of XY shortest paths from source core to each destination core.
    """
    k = int(labels.max()) + 1
    if perm is None:
        perm = np.arange(k, dtype=np.int32)
    coords = grid_coords(rows, cols)
    link_load: Counter = Counter()
    remote_fanout = 0.0
    multicast_hops = 0.0
    packet_count = 0.0

    for u, outs in enumerate(out_neighbors):
        if outs.size == 0 or rates[u] <= 0:
            continue
        src_part = int(labels[u])
        # Distinct remote destination cores. This is the hypergraph move, not edge cut.
        dest_parts = set(int(c) for c in labels[outs])
        dest_parts.discard(src_part)
        if not dest_parts:
            continue
        r = float(rates[u])
        remote_fanout += r * len(dest_parts)
        packet_count += r * len(dest_parts)

        src_loc = int(perm[src_part])
        src_xy = coords[src_loc]
        tree_links = set()
        for dst_part in dest_parts:
            dst_loc = int(perm[dst_part])
            dst_xy = coords[dst_loc]
            for hop in manhattan_path(src_xy, dst_xy):
                tree_links.add(hop)
        multicast_hops += r * len(tree_links)
        for hop in tree_links:
            link_load[hop] += r

    loads = np.array(list(link_load.values()), dtype=float)
    max_link = float(loads.max()) if loads.size else 0.0
    var_link = float(loads.var()) if loads.size else 0.0

    counts = np.bincount(labels, minlength=k).astype(float)
    size_imb = float(counts.max() / max(counts.mean(), 1e-9))
    rate_per_core = np.bincount(labels, weights=rates, minlength=k).astype(float)
    rate_imb = float(rate_per_core.max() / max(rate_per_core.mean(), 1e-9))

    ecut = edge_cut_cost(out_neighbors, labels, None)
    wecut = edge_cut_cost(out_neighbors, labels, rates)
    obj = (
        weights.alpha_hops * multicast_hops
        + weights.beta_max_link * max_link
        + weights.gamma_link_var * var_link
        + weights.delta_size_imbalance * size_imb
        + weights.delta_rate_imbalance * rate_imb
    )
    return MeshMetrics(
        edge_cut=ecut,
        weighted_edge_cut=wecut,
        remote_fanout=remote_fanout,
        multicast_hops=multicast_hops,
        max_link_load=max_link,
        link_load_variance=var_link,
        packet_count=packet_count,
        size_imbalance=size_imb,
        rate_imbalance=rate_imb,
        objective=float(obj),
    )


def boundary_nodes(out_neighbors: List[np.ndarray], in_neighbors: List[np.ndarray], labels: np.ndarray) -> List[int]:
    b = []
    for u in range(len(labels)):
        c = labels[u]
        if out_neighbors[u].size and np.any(labels[out_neighbors[u]] != c):
            b.append(u)
            continue
        if in_neighbors[u].size and np.any(labels[in_neighbors[u]] != c):
            b.append(u)
    return b


def candidate_parts_for_node(
    u: int,
    out_neighbors: List[np.ndarray],
    in_neighbors: List[np.ndarray],
    labels: np.ndarray,
    k: int,
    max_extra: int = 2,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    cur = int(labels[u])
    cands = set(int(x) for x in labels[out_neighbors[u]]) | set(int(x) for x in labels[in_neighbors[u]])
    cands.discard(cur)
    if rng is not None and len(cands) < max_extra + 1:
        extras = list(range(k))
        rng.shuffle(extras)
        for x in extras:
            if x != cur:
                cands.add(int(x))
            if len(cands) >= max_extra + 1:
                break
    return sorted(cands)


def balance_limits(n: int, k: int, slack: float) -> Tuple[int, int]:
    target = n / float(k)
    lo = max(0, int(math.floor((1.0 - slack) * target)))
    hi = int(math.ceil((1.0 + slack) * target))
    return lo, hi


def can_move(labels: np.ndarray, u: int, dst: int, lo: int, hi: int) -> bool:
    src = int(labels[u])
    if src == dst:
        return False
    counts = np.bincount(labels, minlength=max(int(labels.max()) + 1, dst + 1))
    return counts[src] - 1 >= lo and counts[dst] + 1 <= hi


def refine_by_objective(
    out_neighbors: List[np.ndarray],
    in_neighbors: List[np.ndarray],
    labels0: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    perm: Optional[np.ndarray] = None,
    objective: str = "multicast",  # "edge", "weighted_edge", or "multicast"
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    passes: int = 5,
    slack: float = 0.12,
    seed: int = 0,
    time_limit_s: Optional[float] = None,
) -> Tuple[np.ndarray, MeshMetrics, Dict[str, float]]:
    """Simple k-way local-refinement engine. Full recomputation = slower but hard to get wrong."""
    rng = np.random.RandomState(seed)
    labels = labels0.astype(np.int32, copy=True)
    k = int(labels.max()) + 1
    n = len(labels)
    lo, hi = balance_limits(n, k, slack)
    if perm is None:
        perm = np.arange(k, dtype=np.int32)

    def score(lab: np.ndarray) -> float:
        if objective == "edge":
            return edge_cut_cost(out_neighbors, lab, None)
        if objective == "weighted_edge":
            return edge_cut_cost(out_neighbors, lab, rates)
        if objective == "multicast":
            return multicast_metrics(out_neighbors, lab, rates, perm=perm, rows=rows, cols=cols, weights=weights).objective
        raise ValueError(f"unknown objective={objective}")

    t0 = time.perf_counter()
    cur_score = score(labels)
    moves = 0
    evaluated = 0

    for _pass in range(passes):
        improved_this_pass = False
        nodes = boundary_nodes(out_neighbors, in_neighbors, labels)
        rng.shuffle(nodes)
        for u in nodes:
            if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
                break
            best_dst = None
            best_score = cur_score
            cands = candidate_parts_for_node(u, out_neighbors, in_neighbors, labels, k, rng=rng)
            for dst in cands:
                if not can_move(labels, u, dst, lo, hi):
                    continue
                old = int(labels[u])
                labels[u] = int(dst)
                s = score(labels)
                evaluated += 1
                labels[u] = old
                if s < best_score - 1e-9:
                    best_score = s
                    best_dst = int(dst)
            if best_dst is not None:
                labels[u] = best_dst
                cur_score = best_score
                moves += 1
                improved_this_pass = True
        if not improved_this_pass:
            break

    metrics = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights)
    info = {
        "moves": float(moves),
        "evaluated_moves": float(evaluated),
        "runtime_s": float(time.perf_counter() - t0),
        "objective_used": objective,
    }
    return labels, metrics, info


def refine_placement_by_swaps(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    max_passes: int = 4,
    time_limit_s: Optional[float] = None,
) -> Tuple[np.ndarray, MeshMetrics, Dict[str, float]]:
    """Local placement search: swap two core locations when multicast objective improves."""
    k = int(labels.max()) + 1
    perm = np.arange(k, dtype=np.int32)
    t0 = time.perf_counter()
    cur = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).objective
    swaps = 0
    evals = 0
    for _ in range(max_passes):
        improved = False
        for i in range(k):
            for j in range(i + 1, k):
                if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
                    break
                cand = perm.copy()
                cand[i], cand[j] = cand[j], cand[i]
                s = multicast_metrics(out_neighbors, labels, rates, perm=cand, rows=rows, cols=cols, weights=weights).objective
                evals += 1
                if s < cur - 1e-9:
                    perm = cand
                    cur = s
                    swaps += 1
                    improved = True
            if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
                break
        if not improved:
            break
    metrics = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights)
    return perm, metrics, {"placement_swaps": float(swaps), "placement_evals": float(evals), "placement_runtime_s": float(time.perf_counter() - t0)}


def run_mhysmap_core(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
) -> Dict[str, Dict[str, float]]:
    """Run a compact baseline suite for smoke testing and ablation."""
    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)

    rows_out: Dict[str, Dict[str, float]] = {}

    def add(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    add("random_rowmajor", labels_random, row_perm)

    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    add("edgecut_refined_rowmajor", labels_edge, row_perm, info_edge)

    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    add("activity_edge_refined_rowmajor", labels_wedge, row_perm, info_wedge)

    labels_hyper, _, info_hyper = refine_by_objective(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        objective="multicast", weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_rowmajor", labels_hyper, row_perm, info_hyper)

    perm_hyper, metrics_place, info_place = refine_placement_by_swaps(
        outn, labels_hyper, rates, rows, cols, weights=weights, max_passes=4
    )
    m = metrics_place.as_dict()
    m.update(info_hyper)
    m.update(info_place)
    rows_out["mhysmap_partition_placement"] = m

    return rows_out

# -----------------------------------------------------------------------------
# v0.3 placement certification / anytime search
# -----------------------------------------------------------------------------

@dataclass
class PlacementCert:
    status: str
    incumbent_objective: float
    lower_bound: float
    absolute_gap: float
    relative_gap: float
    permutations_evaluated: int
    runtime_s: float
    method: str

    def as_dict(self, prefix: str = "placement_cert_") -> Dict[str, float]:
        return {
            prefix + "status_code": float(1 if self.status == "optimal" else 0),
            prefix + "incumbent_objective": float(self.incumbent_objective),
            prefix + "lower_bound": float(self.lower_bound),
            prefix + "absolute_gap": float(self.absolute_gap),
            prefix + "relative_gap": float(self.relative_gap),
            prefix + "permutations_evaluated": float(self.permutations_evaluated),
            prefix + "runtime_s": float(self.runtime_s),
            prefix + "method_code": float(1 if self.method == "exact" else 2),
        }


def num_directed_mesh_links(rows: int, cols: int) -> int:
    """Number of directed adjacent links in a rectangular mesh."""
    undirected = rows * max(cols - 1, 0) + cols * max(rows - 1, 0)
    return 2 * undirected


def placement_lower_bound(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
) -> float:
    """
    Safe placement lower bound for the multicast objective.

    For each source neuron u, any connected route from its source core to |D_u| distinct
    remote destination cores uses at least |D_u| links. Therefore
      total multicast hops >= remote_fanout.
    Also max_link_load >= total_load / (# directed mesh links), using the same lower bound
    on total load. Link variance is nonnegative. Balance penalties are fixed by labels.

    This bound is intentionally conservative but valid for all placements.
    """
    row_metrics = multicast_metrics(
        out_neighbors, labels, rates, perm=np.arange(int(labels.max()) + 1, dtype=np.int32),
        rows=rows, cols=cols, weights=weights,
    )
    remote = float(row_metrics.remote_fanout)
    nlinks = max(num_directed_mesh_links(rows, cols), 1)
    lb = weights.alpha_hops * remote
    lb += weights.beta_max_link * (remote / nlinks)
    # gamma term is >= 0 and is deliberately omitted.
    lb += weights.delta_size_imbalance * row_metrics.size_imbalance
    lb += weights.delta_rate_imbalance * row_metrics.rate_imbalance
    return float(lb)


def exact_placement_bruteforce(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    time_limit_s: Optional[float] = None,
    fix_core0: bool = True,
) -> Tuple[np.ndarray, MeshMetrics, PlacementCert]:
    """
    Exact placement by enumerating mesh-location permutations.

    Intended for tiny certified experiments (k <= 8 or so). If fix_core0=True we break a
    harmless translation/label symmetry by keeping core 0 at location 0, which reduces work
    and still gives a useful certificate for that normalized placement space.
    """
    import itertools

    k = int(labels.max()) + 1
    if rows * cols < k:
        raise ValueError("mesh has fewer locations than active cores")
    t0 = time.perf_counter()
    lb = placement_lower_bound(out_neighbors, labels, rates, rows, cols, weights)

    best_perm: Optional[np.ndarray] = None
    best_metrics: Optional[MeshMetrics] = None
    best_obj = float("inf")
    evals = 0
    locations = list(range(rows * cols))

    if fix_core0:
        tails = itertools.permutations(locations[1:], k - 1)
        iterator = ([0, *tail] for tail in tails)
    else:
        iterator = itertools.permutations(locations, k)

    exhausted = True
    for perm_tuple in iterator:
        if time_limit_s is not None and (time.perf_counter() - t0) > time_limit_s:
            exhausted = False
            break
        perm = np.asarray(perm_tuple, dtype=np.int32)
        m = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights)
        evals += 1
        if m.objective < best_obj - 1e-12:
            best_obj = float(m.objective)
            best_perm = perm.copy()
            best_metrics = m

    if best_perm is None or best_metrics is None:
        # fallback should only happen with absurdly tiny time limits
        best_perm = np.arange(k, dtype=np.int32)
        best_metrics = multicast_metrics(out_neighbors, labels, rates, perm=best_perm, rows=rows, cols=cols, weights=weights)
        best_obj = float(best_metrics.objective)

    status = "optimal" if exhausted else "feasible"
    cert_lb = best_obj if exhausted else lb
    abs_gap = max(0.0, best_obj - cert_lb)
    rel_gap = abs_gap / max(abs(cert_lb), 1e-12)
    cert = PlacementCert(
        status=status,
        incumbent_objective=float(best_obj),
        lower_bound=float(cert_lb),
        absolute_gap=float(abs_gap),
        relative_gap=float(rel_gap),
        permutations_evaluated=int(evals),
        runtime_s=float(time.perf_counter() - t0),
        method="exact",
    )
    return best_perm, best_metrics, cert


def local_swap_improve_from_perm(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    perm0: np.ndarray,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    max_passes: int = 3,
    time_limit_s: Optional[float] = None,
    t0: Optional[float] = None,
) -> Tuple[np.ndarray, MeshMetrics, int]:
    """Greedy 2-swap placement improvement from a provided initial permutation."""
    if t0 is None:
        t0 = time.perf_counter()
    perm = perm0.astype(np.int32, copy=True)
    k = len(perm)
    cur_metrics = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights)
    cur = float(cur_metrics.objective)
    evals = 0
    for _ in range(max_passes):
        improved = False
        for i in range(k):
            for j in range(i + 1, k):
                if time_limit_s is not None and (time.perf_counter() - t0) > time_limit_s:
                    return perm, cur_metrics, evals
                cand = perm.copy()
                cand[i], cand[j] = cand[j], cand[i]
                m = multicast_metrics(out_neighbors, labels, rates, perm=cand, rows=rows, cols=cols, weights=weights)
                evals += 1
                if m.objective < cur - 1e-12:
                    perm = cand
                    cur_metrics = m
                    cur = float(m.objective)
                    improved = True
        if not improved:
            break
    return perm, cur_metrics, evals


def anytime_placement_search(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    time_limit_s: float = 2.0,
    restarts: int = 16,
    seed: int = 0,
    exact_threshold_k: int = 8,
) -> Tuple[np.ndarray, MeshMetrics, PlacementCert]:
    """
    Certified/anytime placement wrapper.

    - If k <= exact_threshold_k, enumerate placements and return an optimal certificate.
    - Otherwise, run multi-start local swap search and return a feasible incumbent plus a
      safe lower-bound certificate/gap.
    """
    k = int(labels.max()) + 1
    if k <= exact_threshold_k:
        return exact_placement_bruteforce(
            out_neighbors, labels, rates, rows, cols, weights=weights,
            time_limit_s=time_limit_s, fix_core0=True,
        )

    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    lb = placement_lower_bound(out_neighbors, labels, rates, rows, cols, weights)
    best_perm = np.arange(k, dtype=np.int32)
    best_metrics = multicast_metrics(out_neighbors, labels, rates, perm=best_perm, rows=rows, cols=cols, weights=weights)
    evals_total = 0

    # First improve row-major, then random restarts.
    init_perms = [best_perm.copy()]
    for _ in range(max(0, restarts - 1)):
        p = np.arange(k, dtype=np.int32)
        rng.shuffle(p)
        init_perms.append(p)

    for p0 in init_perms:
        if (time.perf_counter() - t0) > time_limit_s:
            break
        perm, metrics, evals = local_swap_improve_from_perm(
            out_neighbors, labels, rates, rows, cols, p0,
            weights=weights, max_passes=3, time_limit_s=time_limit_s, t0=t0,
        )
        evals_total += evals
        if metrics.objective < best_metrics.objective - 1e-12:
            best_perm, best_metrics = perm.copy(), metrics

    inc = float(best_metrics.objective)
    abs_gap = max(0.0, inc - lb)
    rel_gap = abs_gap / max(abs(lb), 1e-12)
    cert = PlacementCert(
        status="feasible",
        incumbent_objective=inc,
        lower_bound=float(lb),
        absolute_gap=float(abs_gap),
        relative_gap=float(rel_gap),
        permutations_evaluated=int(evals_total),
        runtime_s=float(time.perf_counter() - t0),
        method="anytime",
    )
    return best_perm, best_metrics, cert


def run_mhysmap_core_v03(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
) -> Dict[str, Dict[str, float]]:
    """v0.3 baseline suite: v0.2 core plus certified/anytime placement reporting."""
    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)

    rows_out: Dict[str, Dict[str, float]] = {}

    def add(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    add("random_rowmajor", labels_random, row_perm)

    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    add("edgecut_refined_rowmajor", labels_edge, row_perm, info_edge)

    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    add("activity_edge_refined_rowmajor", labels_wedge, row_perm, info_wedge)

    labels_hyper, _, info_hyper = refine_by_objective(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        objective="multicast", weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_rowmajor", labels_hyper, row_perm, info_hyper)

    # legacy local swap placement for apples-to-apples with v0.2
    perm_local, metrics_local, info_local = refine_placement_by_swaps(
        outn, labels_hyper, rates, rows, cols, weights=weights, max_passes=4,
        time_limit_s=placement_time_limit_s,
    )
    m_local = metrics_local.as_dict()
    m_local.update(info_hyper)
    m_local.update(info_local)
    rows_out["mhysmap_partition_local_placement"] = m_local

    # new v0.3 certified/anytime placement
    perm_cert, metrics_cert, cert = anytime_placement_search(
        outn, labels_hyper, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 33, exact_threshold_k=exact_threshold_k,
    )
    m_cert = metrics_cert.as_dict()
    m_cert.update(info_hyper)
    m_cert.update(cert.as_dict())
    rows_out["mhysmap_partition_certified_placement"] = m_cert

    return rows_out

# -----------------------------------------------------------------------------
# v0.4 stronger baselines: pairwise-QAP-style placement for graph baselines
# -----------------------------------------------------------------------------

def partition_flow_matrix(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: Optional[np.ndarray] = None,
    include_self: bool = False,
) -> np.ndarray:
    """
    Collapse neuron-level communication into a directed partition-to-partition flow matrix.

    If rates is None, each synapse u->v contributes 1. Otherwise it contributes r_u.
    This is the classical pairwise/QAP abstraction we compare against the multicast
    hypergraph objective.
    """
    k = int(labels.max()) + 1
    F = np.zeros((k, k), dtype=float)
    for u, outs in enumerate(out_neighbors):
        if outs.size == 0:
            continue
        src = int(labels[u])
        w = 1.0 if rates is None else float(rates[u])
        if w <= 0:
            continue
        for v in outs:
            dst = int(labels[int(v)])
            if include_self or dst != src:
                F[src, dst] += w
    return F


def mesh_distance_matrix(rows: int, cols: int) -> np.ndarray:
    """All-pairs Manhattan distance between mesh locations."""
    coords = grid_coords(rows, cols)
    m = len(coords)
    D = np.zeros((m, m), dtype=float)
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            D[i, j] = abs(x1 - x2) + abs(y1 - y2)
    return D


def pairwise_qap_objective(F: np.ndarray, perm: np.ndarray, rows: int, cols: int) -> float:
    """Classical directed pairwise placement objective: sum_ij F_ij * dist(pi_i, pi_j)."""
    D = mesh_distance_matrix(rows, cols)
    p = np.asarray(perm, dtype=np.int32)
    return float((F * D[p[:, None], p[None, :]]).sum())


@dataclass
class PairwisePlacementInfo:
    method: str
    status: str
    qap_objective: float
    runtime_s: float
    evaluations: int
    swaps: int

    def as_dict(self, prefix: str = "pairwise_place_") -> Dict[str, float]:
        return {
            prefix + "method_code": float(1 if self.method == "exact" else 2),
            prefix + "status_code": float(1 if self.status == "optimal" else 0),
            prefix + "qap_objective": float(self.qap_objective),
            prefix + "runtime_s": float(self.runtime_s),
            prefix + "evaluations": float(self.evaluations),
            prefix + "swaps": float(self.swaps),
        }


def pairwise_exact_placement(
    F: np.ndarray,
    rows: int,
    cols: int,
    time_limit_s: Optional[float] = None,
    fix_core0: bool = True,
) -> Tuple[np.ndarray, PairwisePlacementInfo]:
    """Exact pairwise-QAP placement for tiny k by enumerating mesh-location assignments."""
    import itertools

    k = int(F.shape[0])
    locations = list(range(rows * cols))
    if len(locations) < k:
        raise ValueError("mesh has fewer locations than active cores")
    t0 = time.perf_counter()
    best_obj = float("inf")
    best_perm = np.arange(k, dtype=np.int32)
    evals = 0
    exhausted = True

    if fix_core0:
        tails = itertools.permutations(locations[1:], k - 1)
        iterator = ([0, *tail] for tail in tails)
    else:
        iterator = itertools.permutations(locations, k)

    for perm_tuple in iterator:
        if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
            exhausted = False
            break
        perm = np.asarray(perm_tuple, dtype=np.int32)
        obj = pairwise_qap_objective(F, perm, rows, cols)
        evals += 1
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_perm = perm.copy()

    info = PairwisePlacementInfo(
        method="exact",
        status="optimal" if exhausted else "feasible",
        qap_objective=float(best_obj),
        runtime_s=float(time.perf_counter() - t0),
        evaluations=int(evals),
        swaps=0,
    )
    return best_perm, info


def pairwise_local_swap_from_perm(
    F: np.ndarray,
    rows: int,
    cols: int,
    perm0: np.ndarray,
    max_passes: int = 3,
    time_limit_s: Optional[float] = None,
    t0: Optional[float] = None,
) -> Tuple[np.ndarray, float, int, int]:
    """Greedy 2-swap local search under the pairwise QAP objective."""
    if t0 is None:
        t0 = time.perf_counter()
    perm = perm0.astype(np.int32, copy=True)
    k = len(perm)
    cur = pairwise_qap_objective(F, perm, rows, cols)
    evals = 0
    swaps = 0
    for _ in range(max_passes):
        improved = False
        for i in range(k):
            for j in range(i + 1, k):
                if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
                    return perm, float(cur), evals, swaps
                cand = perm.copy()
                cand[i], cand[j] = cand[j], cand[i]
                obj = pairwise_qap_objective(F, cand, rows, cols)
                evals += 1
                if obj < cur - 1e-12:
                    perm = cand
                    cur = obj
                    swaps += 1
                    improved = True
        if not improved:
            break
    return perm, float(cur), evals, swaps


def pairwise_placement_search(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: Optional[np.ndarray],
    rows: int,
    cols: int,
    time_limit_s: float = 2.0,
    restarts: int = 16,
    seed: int = 0,
    exact_threshold_k: int = 8,
) -> Tuple[np.ndarray, PairwisePlacementInfo]:
    """
    Pairwise-QAP-style placement used by graph baselines.

    This deliberately optimizes the classical collapsed pairwise objective, not the multicast
    objective. We then evaluate the resulting placement with multicast metrics to test whether
    graph/QAP baselines can explain away M-HySMap's gains.
    """
    F = partition_flow_matrix(out_neighbors, labels, rates=rates, include_self=False)
    k = int(labels.max()) + 1
    if k <= exact_threshold_k:
        return pairwise_exact_placement(F, rows, cols, time_limit_s=time_limit_s, fix_core0=True)

    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    best_perm = np.arange(k, dtype=np.int32)
    best_obj = pairwise_qap_objective(F, best_perm, rows, cols)
    evals_total = 0
    swaps_total = 0

    init_perms = [best_perm.copy()]
    for _ in range(max(0, restarts - 1)):
        p = np.arange(k, dtype=np.int32)
        rng.shuffle(p)
        init_perms.append(p)

    for p0 in init_perms:
        if time.perf_counter() - t0 > time_limit_s:
            break
        p, obj, evals, swaps = pairwise_local_swap_from_perm(
            F, rows, cols, p0, max_passes=3, time_limit_s=time_limit_s, t0=t0
        )
        evals_total += evals
        swaps_total += swaps
        if obj < best_obj - 1e-12:
            best_perm = p.copy()
            best_obj = obj

    info = PairwisePlacementInfo(
        method="anytime",
        status="feasible",
        qap_objective=float(best_obj),
        runtime_s=float(time.perf_counter() - t0),
        evaluations=int(evals_total),
        swaps=int(swaps_total),
    )
    return best_perm, info


def run_mhysmap_core_v04(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
) -> Dict[str, Dict[str, float]]:
    """
    v0.4 baseline suite.

    Key additions over v0.3:
      - edge-cut FM + pairwise-QAP placement
      - activity-edge FM + activity-weighted pairwise-QAP placement
      - M-HySMap partition + pairwise-QAP placement ablation
      - M-HySMap partition + multicast certified/anytime placement
    """
    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)

    rows_out: Dict[str, Dict[str, float]] = {}

    def add(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    # sanity floor
    add("random_rowmajor", labels_random, row_perm)

    # Classical graph partitioning baseline: optimize edge count, then give it a pairwise QAP placement.
    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    add("edgecut_refined_rowmajor", labels_edge, row_perm, info_edge)
    perm_edge_qap, info_edge_qap = pairwise_placement_search(
        outn, labels_edge, rates=None, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 101, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_edge)
    extra.update(info_edge_qap.as_dict())
    add("edgecut_refined_pairwise_qap", labels_edge, perm_edge_qap, extra)

    # Activity-aware graph baseline: optimize weighted edge cuts, then pairwise QAP weighted by source activity.
    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    add("activity_edge_refined_rowmajor", labels_wedge, row_perm, info_wedge)
    perm_wedge_qap, info_wedge_qap = pairwise_placement_search(
        outn, labels_wedge, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 202, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_wedge)
    extra.update(info_wedge_qap.as_dict())
    add("activity_edge_refined_pairwise_qap", labels_wedge, perm_wedge_qap, extra)

    # Our multicast-aware hypergraph refinement.
    labels_hyper, _, info_hyper = refine_by_objective(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        objective="multicast", weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_rowmajor", labels_hyper, row_perm, info_hyper)

    # Ablation: same hypergraph partition, but classical pairwise QAP placement.
    perm_hyper_pairwise, info_hyper_pairwise = pairwise_placement_search(
        outn, labels_hyper, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 303, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_hyper)
    extra.update(info_hyper_pairwise.as_dict())
    add("mhysmap_partition_pairwise_qap", labels_hyper, perm_hyper_pairwise, extra)

    # Final method: hypergraph partition + multicast/congestion-aware certified/anytime placement.
    perm_cert, metrics_cert, cert = anytime_placement_search(
        outn, labels_hyper, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 33, exact_threshold_k=exact_threshold_k,
    )
    m_cert = metrics_cert.as_dict()
    m_cert.update(info_hyper)
    m_cert.update(cert.as_dict())
    rows_out["mhysmap_partition_multicast_certified"] = m_cert

    return rows_out

# -----------------------------------------------------------------------------
# v0.5 joint / alternating partition-placement refinement
# -----------------------------------------------------------------------------

@dataclass
class JointRefineInfo:
    cycles: int
    partition_moves: int
    partition_evals: int
    placement_evals: int
    placement_restarts: int
    runtime_s: float
    initial_objective: float
    final_objective: float
    objective_improvement_pct: float

    def as_dict(self, prefix: str = "joint_") -> Dict[str, float]:
        return {
            prefix + "cycles": float(self.cycles),
            prefix + "partition_moves": float(self.partition_moves),
            prefix + "partition_evals": float(self.partition_evals),
            prefix + "placement_evals": float(self.placement_evals),
            prefix + "placement_restarts": float(self.placement_restarts),
            prefix + "runtime_s": float(self.runtime_s),
            prefix + "initial_objective": float(self.initial_objective),
            prefix + "final_objective": float(self.final_objective),
            prefix + "objective_improvement_pct": float(self.objective_improvement_pct),
        }


def joint_partition_placement_refinement(
    out_neighbors: List[np.ndarray],
    in_neighbors: List[np.ndarray],
    labels0: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    cycles: int = 3,
    partition_passes_per_cycle: int = 2,
    slack: float = 0.15,
    placement_time_per_cycle_s: float = 1.0,
    placement_restarts_per_cycle: int = 12,
    seed: int = 0,
    exact_threshold_k: int = 8,
) -> Tuple[np.ndarray, np.ndarray, MeshMetrics, JointRefineInfo, PlacementCert]:
    """
    Alternating co-optimization for v0.5.

    v0.4 did:
      partition with row-major placement -> place after the fact.

    v0.5 does:
      place current partition -> move neurons using that physical placement -> place again -> repeat.

    The neuron-move objective is still the multicast/congestion objective, but now its gain
    is evaluated under the current physical mesh placement. This is the lightest clean form of
    joint partition-placement refinement and directly answers the paper question: do we gain by
    letting partitioning see the hardware layout instead of treating placement as a final cleanup?
    """
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    labels = labels0.astype(np.int32, copy=True)
    k = int(labels.max()) + 1

    # Start from row-major, but immediately improve placement so first partition moves are topology-aware.
    perm = np.arange(k, dtype=np.int32)
    init_metrics = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights)
    initial_objective = float(init_metrics.objective)
    best_labels = labels.copy()
    best_perm = perm.copy()
    best_metrics = init_metrics

    total_moves = 0
    total_part_evals = 0
    total_place_evals = 0
    restarts_used = 0
    final_cert: Optional[PlacementCert] = None
    completed_cycles = 0

    for cyc in range(cycles):
        completed_cycles = cyc + 1

        # Placement refresh for current labels.
        perm, placed_metrics, cert = anytime_placement_search(
            out_neighbors, labels, rates, rows, cols, weights=weights,
            time_limit_s=placement_time_per_cycle_s,
            restarts=placement_restarts_per_cycle,
            seed=seed + 1000 + 17 * cyc,
            exact_threshold_k=exact_threshold_k,
        )
        final_cert = cert
        total_place_evals += int(cert.permutations_evaluated)
        restarts_used += int(placement_restarts_per_cycle)
        if placed_metrics.objective < best_metrics.objective - 1e-12:
            best_labels = labels.copy()
            best_perm = perm.copy()
            best_metrics = placed_metrics

        # Partition refresh under the current placement.
        labels_new, part_metrics, info = refine_by_objective(
            out_neighbors, in_neighbors, labels, rates, rows, cols, perm=perm,
            objective="multicast", weights=weights,
            passes=partition_passes_per_cycle, slack=slack,
            seed=seed + 2000 + 31 * cyc,
        )
        total_moves += int(info.get("moves", 0.0))
        total_part_evals += int(info.get("evaluated_moves", 0.0))

        # If no partition move happened and the current placed solution is not improving, stop early.
        labels = labels_new
        if part_metrics.objective < best_metrics.objective - 1e-12:
            best_labels = labels.copy()
            best_perm = perm.copy()
            best_metrics = part_metrics

        if int(info.get("moves", 0.0)) == 0 and cyc > 0:
            break

    # Final placement pass for the best partition found.
    perm_final, metrics_final, cert_final = anytime_placement_search(
        out_neighbors, best_labels, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_per_cycle_s,
        restarts=max(placement_restarts_per_cycle, 16),
        seed=seed + 9999,
        exact_threshold_k=exact_threshold_k,
    )
    final_cert = cert_final
    total_place_evals += int(cert_final.permutations_evaluated)
    if metrics_final.objective < best_metrics.objective - 1e-12:
        best_perm = perm_final.copy()
        best_metrics = metrics_final

    final_objective = float(best_metrics.objective)
    improvement_pct = (initial_objective - final_objective) / max(abs(initial_objective), 1e-12) * 100.0
    info = JointRefineInfo(
        cycles=completed_cycles,
        partition_moves=total_moves,
        partition_evals=total_part_evals,
        placement_evals=total_place_evals,
        placement_restarts=restarts_used,
        runtime_s=float(time.perf_counter() - t0),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        objective_improvement_pct=float(improvement_pct),
    )
    assert final_cert is not None
    return best_labels, best_perm, best_metrics, info, final_cert


def run_mhysmap_core_v05(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
    joint_cycles: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    v0.5 baseline suite.

    Keeps all v0.4 strong baselines and adds:
      - mhysmap_joint_multicast: alternating multicast placement + topology-aware neuron moves.
    """
    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)

    rows_out: Dict[str, Dict[str, float]] = {}

    def add(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    # sanity floor
    add("random_rowmajor", labels_random, row_perm)

    # v0.4 strong graph baselines.
    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    add("edgecut_refined_rowmajor", labels_edge, row_perm, info_edge)
    perm_edge_qap, info_edge_qap = pairwise_placement_search(
        outn, labels_edge, rates=None, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 101, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_edge); extra.update(info_edge_qap.as_dict())
    add("edgecut_refined_pairwise_qap", labels_edge, perm_edge_qap, extra)

    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    add("activity_edge_refined_rowmajor", labels_wedge, row_perm, info_wedge)
    perm_wedge_qap, info_wedge_qap = pairwise_placement_search(
        outn, labels_wedge, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 202, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_wedge); extra.update(info_wedge_qap.as_dict())
    add("activity_edge_refined_pairwise_qap", labels_wedge, perm_wedge_qap, extra)

    # v0.4 sequential M-HySMap: partition under row-major, then place.
    labels_hyper, _, info_hyper = refine_by_objective(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        objective="multicast", weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_rowmajor", labels_hyper, row_perm, info_hyper)

    perm_hyper_pairwise, info_hyper_pairwise = pairwise_placement_search(
        outn, labels_hyper, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 303, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_hyper); extra.update(info_hyper_pairwise.as_dict())
    add("mhysmap_partition_pairwise_qap", labels_hyper, perm_hyper_pairwise, extra)

    perm_cert, metrics_cert, cert = anytime_placement_search(
        outn, labels_hyper, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 33, exact_threshold_k=exact_threshold_k,
    )
    m_cert = metrics_cert.as_dict(); m_cert.update(info_hyper); m_cert.update(cert.as_dict())
    rows_out["mhysmap_partition_multicast_certified"] = m_cert

    # v0.5: start from the activity-edge partition, then alternate placement and topology-aware hyper-FM.
    labels_joint, perm_joint, metrics_joint, joint_info, joint_cert = joint_partition_placement_refinement(
        outn, inn, labels_wedge, rates, rows, cols, weights=weights,
        cycles=joint_cycles,
        partition_passes_per_cycle=max(1, passes_hyper // 2),
        slack=slack,
        placement_time_per_cycle_s=max(0.25, placement_time_limit_s / max(joint_cycles + 1, 1)),
        placement_restarts_per_cycle=max(4, placement_restarts // max(joint_cycles + 1, 1)),
        seed=seed + 505,
        exact_threshold_k=exact_threshold_k,
    )
    m_joint = metrics_joint.as_dict()
    m_joint.update(joint_info.as_dict())
    m_joint.update(joint_cert.as_dict())
    rows_out["mhysmap_joint_multicast"] = m_joint

    return rows_out

# -----------------------------------------------------------------------------
# v0.6 incremental multicast gain updates
# -----------------------------------------------------------------------------

class IncrementalMulticastState:
    """
    Cached evaluator for the fixed-placement multicast objective.

    It maintains per-source multicast trees and global link loads. A neuron move v:a->b
    can only change:
      - h_v, because the source core of v changes;
      - h_u for predecessors u with edge u->v, because v's destination core changes.

    Therefore candidate moves are evaluated by recomputing only this affected source set,
    rather than recomputing the whole SNN objective for every candidate.
    """

    def __init__(
        self,
        out_neighbors: List[np.ndarray],
        in_neighbors: List[np.ndarray],
        labels: np.ndarray,
        rates: np.ndarray,
        perm: np.ndarray,
        rows: int,
        cols: int,
        weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    ):
        self.out_neighbors = out_neighbors
        self.in_neighbors = in_neighbors
        self.labels = labels.astype(np.int32, copy=True)
        self.rates = rates.astype(float, copy=False)
        self.perm = perm.astype(np.int32, copy=True)
        self.rows = int(rows)
        self.cols = int(cols)
        self.weights = weights
        self.n = len(self.labels)
        self.k = int(self.labels.max()) + 1
        self.coords = grid_coords(rows, cols)

        self.source_links: List[frozenset] = [frozenset() for _ in range(self.n)]
        self.source_hops = np.zeros(self.n, dtype=float)
        self.source_remote = np.zeros(self.n, dtype=float)
        self.link_load: Counter = Counter()
        self.total_hops = 0.0
        self.remote_fanout = 0.0
        self.packet_count = 0.0

        self.counts = np.bincount(self.labels, minlength=self.k).astype(float)
        self.rate_per_core = np.bincount(self.labels, weights=self.rates, minlength=self.k).astype(float)

        for u in range(self.n):
            remote, hops, links = self._compute_source(u)
            self.source_remote[u] = remote
            self.source_hops[u] = hops
            self.source_links[u] = links
            self.remote_fanout += remote
            self.packet_count += remote
            self.total_hops += hops
            r = float(self.rates[u])
            if r > 0:
                for link in links:
                    self.link_load[link] += r
        self.current_objective = self._objective_from(self.total_hops, self.link_load, self.counts, self.rate_per_core)

    def _compute_source(self, u: int) -> Tuple[float, float, frozenset]:
        outs = self.out_neighbors[int(u)]
        r = float(self.rates[int(u)])
        if outs.size == 0 or r <= 0:
            return 0.0, 0.0, frozenset()
        src_part = int(self.labels[int(u)])
        dest_parts = set(int(x) for x in self.labels[outs])
        dest_parts.discard(src_part)
        if not dest_parts:
            return 0.0, 0.0, frozenset()
        src_xy = self.coords[int(self.perm[src_part])]
        links = set()
        for dst_part in dest_parts:
            dst_xy = self.coords[int(self.perm[int(dst_part)])]
            links.update(manhattan_path(src_xy, dst_xy))
        remote = r * float(len(dest_parts))
        hops = r * float(len(links))
        return float(remote), float(hops), frozenset(links)

    def _objective_from(self, total_hops: float, link_load: Counter, counts: np.ndarray, rate_per_core: np.ndarray) -> float:
        vals = [float(x) for x in link_load.values() if float(x) > 1e-12]
        if vals:
            loads = np.asarray(vals, dtype=float)
            max_link = float(loads.max())
            var_link = float(loads.var())
        else:
            max_link = 0.0
            var_link = 0.0
        size_imb = float(counts.max() / max(counts.mean(), 1e-9)) if counts.size else 0.0
        rate_imb = float(rate_per_core.max() / max(rate_per_core.mean(), 1e-9)) if rate_per_core.size else 0.0
        return float(
            self.weights.alpha_hops * float(total_hops)
            + self.weights.beta_max_link * max_link
            + self.weights.gamma_link_var * var_link
            + self.weights.delta_size_imbalance * size_imb
            + self.weights.delta_rate_imbalance * rate_imb
        )

    def affected_sources(self, v: int) -> List[int]:
        aff = set(int(x) for x in self.in_neighbors[int(v)])
        aff.add(int(v))
        return sorted(aff)

    @staticmethod
    def _add_link(counter: Counter, link, delta: float) -> None:
        newv = float(counter.get(link, 0.0)) + float(delta)
        if abs(newv) <= 1e-12:
            if link in counter:
                del counter[link]
        else:
            counter[link] = newv

    def evaluate_move(self, v: int, dst: int) -> Tuple[float, float]:
        """Return (gain, new_objective) for v -> dst without mutating the state."""
        v = int(v); dst = int(dst)
        src = int(self.labels[v])
        if src == dst:
            return -float("inf"), self.current_objective
        aff = self.affected_sources(v)
        tmp_load = Counter(self.link_load)
        old_hops = 0.0
        for u in aff:
            r = float(self.rates[u])
            old_hops += float(self.source_hops[u])
            if r > 0:
                for link in self.source_links[u]:
                    self._add_link(tmp_load, link, -r)

        old_label = int(self.labels[v])
        self.labels[v] = dst
        new_hops_aff = 0.0
        for u in aff:
            _, hops, links = self._compute_source(u)
            new_hops_aff += hops
            r = float(self.rates[u])
            if r > 0:
                for link in links:
                    self._add_link(tmp_load, link, r)
        self.labels[v] = old_label

        counts = self.counts.copy()
        rate_core = self.rate_per_core.copy()
        counts[src] -= 1.0
        counts[dst] += 1.0
        rv = float(self.rates[v])
        rate_core[src] -= rv
        rate_core[dst] += rv

        new_total_hops = self.total_hops - old_hops + new_hops_aff
        new_obj = self._objective_from(new_total_hops, tmp_load, counts, rate_core)
        return float(self.current_objective - new_obj), float(new_obj)

    def apply_move(self, v: int, dst: int) -> float:
        """Apply v -> dst and return the new current objective."""
        v = int(v); dst = int(dst)
        src = int(self.labels[v])
        if src == dst:
            return self.current_objective
        aff = self.affected_sources(v)

        old_hops = 0.0
        old_remote = 0.0
        for u in aff:
            r = float(self.rates[u])
            old_hops += float(self.source_hops[u])
            old_remote += float(self.source_remote[u])
            if r > 0:
                for link in self.source_links[u]:
                    self._add_link(self.link_load, link, -r)

        self.labels[v] = dst
        self.counts[src] -= 1.0
        self.counts[dst] += 1.0
        rv = float(self.rates[v])
        self.rate_per_core[src] -= rv
        self.rate_per_core[dst] += rv

        new_hops_aff = 0.0
        new_remote_aff = 0.0
        for u in aff:
            remote, hops, links = self._compute_source(u)
            self.source_remote[u] = remote
            self.source_hops[u] = hops
            self.source_links[u] = links
            new_hops_aff += hops
            new_remote_aff += remote
            r = float(self.rates[u])
            if r > 0:
                for link in links:
                    self._add_link(self.link_load, link, r)

        self.total_hops = float(self.total_hops - old_hops + new_hops_aff)
        self.remote_fanout = float(self.remote_fanout - old_remote + new_remote_aff)
        self.packet_count = self.remote_fanout
        self.current_objective = self._objective_from(self.total_hops, self.link_load, self.counts, self.rate_per_core)
        return self.current_objective

    def full_metrics(self) -> MeshMetrics:
        return multicast_metrics(
            self.out_neighbors, self.labels, self.rates, perm=self.perm,
            rows=self.rows, cols=self.cols, weights=self.weights,
        )


def refine_multicast_incremental(
    out_neighbors: List[np.ndarray],
    in_neighbors: List[np.ndarray],
    labels0: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    perm: Optional[np.ndarray] = None,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    passes: int = 5,
    slack: float = 0.12,
    seed: int = 0,
    time_limit_s: Optional[float] = None,
) -> Tuple[np.ndarray, MeshMetrics, Dict[str, float]]:
    """
    Hyper-FM-style multicast refinement with incremental gain evaluation.

    This should match the v0.5 full-recompute objective decisions closely, but candidate
    evaluation touches only h_v plus predecessor hyperedges instead of all source hyperedges.
    """
    rng = np.random.RandomState(seed)
    labels0 = labels0.astype(np.int32, copy=True)
    k = int(labels0.max()) + 1
    n = len(labels0)
    if perm is None:
        perm = np.arange(k, dtype=np.int32)
    lo, hi = balance_limits(n, k, slack)
    t0 = time.perf_counter()
    state = IncrementalMulticastState(out_neighbors, in_neighbors, labels0, rates, perm, rows, cols, weights)

    moves = 0
    evaluated = 0
    affected_total = 0
    affected_max = 0

    for _pass in range(passes):
        improved = False
        nodes = boundary_nodes(out_neighbors, in_neighbors, state.labels)
        rng.shuffle(nodes)
        for u in nodes:
            if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
                break
            best_dst = None
            best_gain = 0.0
            best_obj = state.current_objective
            cands = candidate_parts_for_node(u, out_neighbors, in_neighbors, state.labels, k, rng=rng)
            aff_size = len(state.affected_sources(int(u)))
            affected_total += aff_size
            affected_max = max(affected_max, aff_size)
            for dst in cands:
                if not can_move(state.labels, int(u), int(dst), lo, hi):
                    continue
                gain, new_obj = state.evaluate_move(int(u), int(dst))
                evaluated += 1
                if gain > best_gain + 1e-9:
                    best_gain = float(gain)
                    best_dst = int(dst)
                    best_obj = float(new_obj)
            if best_dst is not None:
                state.apply_move(int(u), int(best_dst))
                # Small guard against numerical drift in the cached objective.
                state.current_objective = best_obj
                moves += 1
                improved = True
        if not improved:
            break

    metrics = state.full_metrics()
    runtime = float(time.perf_counter() - t0)
    validation_abs = abs(float(metrics.objective) - float(state.current_objective))
    info = {
        "moves": float(moves),
        "evaluated_moves": float(evaluated),
        "runtime_s": runtime,
        "objective_used": "multicast_incremental",
        "incremental_evaluator": 1.0,
        "incremental_avg_affected_sources": float(affected_total / max(evaluated, 1)),
        "incremental_max_affected_sources": float(affected_max),
        "incremental_validation_abs_objective_error": float(validation_abs),
    }
    return state.labels.copy(), metrics, info


@dataclass
class JointRefineInfoV06:
    cycles: int
    partition_moves: int
    partition_evals: int
    placement_evals: int
    placement_restarts: int
    runtime_s: float
    initial_objective: float
    final_objective: float
    objective_improvement_pct: float
    partition_runtime_s: float
    avg_affected_sources: float
    max_affected_sources: float
    validation_abs_objective_error: float

    def as_dict(self, prefix: str = "joint_") -> Dict[str, float]:
        return {
            prefix + "cycles": float(self.cycles),
            prefix + "partition_moves": float(self.partition_moves),
            prefix + "partition_evals": float(self.partition_evals),
            prefix + "placement_evals": float(self.placement_evals),
            prefix + "placement_restarts": float(self.placement_restarts),
            prefix + "runtime_s": float(self.runtime_s),
            prefix + "initial_objective": float(self.initial_objective),
            prefix + "final_objective": float(self.final_objective),
            prefix + "objective_improvement_pct": float(self.objective_improvement_pct),
            prefix + "partition_runtime_s": float(self.partition_runtime_s),
            prefix + "avg_affected_sources": float(self.avg_affected_sources),
            prefix + "max_affected_sources": float(self.max_affected_sources),
            prefix + "validation_abs_objective_error": float(self.validation_abs_objective_error),
        }


def joint_partition_placement_refinement_incremental(
    out_neighbors: List[np.ndarray],
    in_neighbors: List[np.ndarray],
    labels0: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    cycles: int = 3,
    partition_passes_per_cycle: int = 2,
    slack: float = 0.15,
    placement_time_per_cycle_s: float = 1.0,
    placement_restarts_per_cycle: int = 12,
    seed: int = 0,
    exact_threshold_k: int = 8,
) -> Tuple[np.ndarray, np.ndarray, MeshMetrics, JointRefineInfoV06, PlacementCert]:
    """v0.6 alternating refinement using incremental multicast move gains."""
    t0 = time.perf_counter()
    labels = labels0.astype(np.int32, copy=True)
    k = int(labels.max()) + 1
    perm = np.arange(k, dtype=np.int32)
    init_metrics = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights)
    initial_objective = float(init_metrics.objective)
    best_labels = labels.copy()
    best_perm = perm.copy()
    best_metrics = init_metrics

    total_moves = 0
    total_part_evals = 0
    total_place_evals = 0
    restarts_used = 0
    part_runtime = 0.0
    affected_weighted_sum = 0.0
    affected_den = 0.0
    affected_max = 0.0
    validation_error = 0.0
    final_cert: Optional[PlacementCert] = None
    completed_cycles = 0

    for cyc in range(cycles):
        completed_cycles = cyc + 1
        perm, placed_metrics, cert = anytime_placement_search(
            out_neighbors, labels, rates, rows, cols, weights=weights,
            time_limit_s=placement_time_per_cycle_s,
            restarts=placement_restarts_per_cycle,
            seed=seed + 1000 + 17 * cyc,
            exact_threshold_k=exact_threshold_k,
        )
        final_cert = cert
        total_place_evals += int(cert.permutations_evaluated)
        restarts_used += int(placement_restarts_per_cycle)
        if placed_metrics.objective < best_metrics.objective - 1e-12:
            best_labels = labels.copy(); best_perm = perm.copy(); best_metrics = placed_metrics

        labels_new, part_metrics, info = refine_multicast_incremental(
            out_neighbors, in_neighbors, labels, rates, rows, cols, perm=perm,
            weights=weights, passes=partition_passes_per_cycle, slack=slack,
            seed=seed + 2000 + 31 * cyc,
        )
        total_moves += int(info.get("moves", 0.0))
        total_part_evals += int(info.get("evaluated_moves", 0.0))
        part_runtime += float(info.get("runtime_s", 0.0))
        evals = max(float(info.get("evaluated_moves", 0.0)), 1.0)
        affected_weighted_sum += float(info.get("incremental_avg_affected_sources", 0.0)) * evals
        affected_den += evals
        affected_max = max(affected_max, float(info.get("incremental_max_affected_sources", 0.0)))
        validation_error = max(validation_error, float(info.get("incremental_validation_abs_objective_error", 0.0)))

        labels = labels_new
        if part_metrics.objective < best_metrics.objective - 1e-12:
            best_labels = labels.copy(); best_perm = perm.copy(); best_metrics = part_metrics
        if int(info.get("moves", 0.0)) == 0 and cyc > 0:
            break

    perm_final, metrics_final, cert_final = anytime_placement_search(
        out_neighbors, best_labels, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_per_cycle_s,
        restarts=max(placement_restarts_per_cycle, 16),
        seed=seed + 9999,
        exact_threshold_k=exact_threshold_k,
    )
    final_cert = cert_final
    total_place_evals += int(cert_final.permutations_evaluated)
    if metrics_final.objective < best_metrics.objective - 1e-12:
        best_perm = perm_final.copy(); best_metrics = metrics_final

    final_objective = float(best_metrics.objective)
    improvement_pct = (initial_objective - final_objective) / max(abs(initial_objective), 1e-12) * 100.0
    info = JointRefineInfoV06(
        cycles=completed_cycles,
        partition_moves=total_moves,
        partition_evals=total_part_evals,
        placement_evals=total_place_evals,
        placement_restarts=restarts_used,
        runtime_s=float(time.perf_counter() - t0),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        objective_improvement_pct=float(improvement_pct),
        partition_runtime_s=float(part_runtime),
        avg_affected_sources=float(affected_weighted_sum / max(affected_den, 1.0)),
        max_affected_sources=float(affected_max),
        validation_abs_objective_error=float(validation_error),
    )
    assert final_cert is not None
    return best_labels, best_perm, best_metrics, info, final_cert


def run_mhysmap_core_v06(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
    joint_cycles: int = 3,
    include_full_joint_reference: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    v0.6 baseline suite.

    Keeps v0.5 baselines and adds an incremental-gain version of joint M-HySMap.
    The point of this version is not a new objective; it is making the same objective scalable
    by evaluating neuron moves through only affected multicast hyperedges.
    """
    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)
    rows_out: Dict[str, Dict[str, float]] = {}

    def add(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    add("random_rowmajor", labels_random, row_perm)

    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    add("edgecut_refined_rowmajor", labels_edge, row_perm, info_edge)
    perm_edge_qap, info_edge_qap = pairwise_placement_search(
        outn, labels_edge, rates=None, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 101, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_edge); extra.update(info_edge_qap.as_dict())
    add("edgecut_refined_pairwise_qap", labels_edge, perm_edge_qap, extra)

    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    add("activity_edge_refined_rowmajor", labels_wedge, row_perm, info_wedge)
    perm_wedge_qap, info_wedge_qap = pairwise_placement_search(
        outn, labels_wedge, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 202, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_wedge); extra.update(info_wedge_qap.as_dict())
    add("activity_edge_refined_pairwise_qap", labels_wedge, perm_wedge_qap, extra)

    # Sequential hypergraph partition: full recompute reference and incremental version under row-major.
    # v1.0 hybrid: start from the strong activity-edge + pairwise-QAP solution,
    # then do multicast-hypergraph neuron moves under that physical placement.
    # This directly answers the large-mesh failure mode exposed in v0.9.
    labels_activity_qap_hyper, metrics_activity_qap_hyper, info_activity_qap_hyper = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=perm_wedge_qap,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 404,
    )
    perm_activity_qap_hyper, metrics_activity_qap_hyper_placed, cert_activity_qap_hyper = anytime_placement_search_v10(
        outn, labels_activity_qap_hyper, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 405, exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_wedge_qap, perm_edge_qap],
    )
    m_activity_qap_hyper = metrics_activity_qap_hyper_placed.as_dict(); m_activity_qap_hyper.update(info_activity_qap_hyper); m_activity_qap_hyper.update(cert_activity_qap_hyper.as_dict())
    rows_out["mhysmap_activity_qap_seeded_hyper_refine"] = m_activity_qap_hyper

    labels_hyper_full, _, info_hyper_full = refine_by_objective(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        objective="multicast", weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_full_recompute_rowmajor", labels_hyper_full, row_perm, info_hyper_full)

    labels_hyper_inc, _, info_hyper_inc = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_incremental_rowmajor", labels_hyper_inc, row_perm, info_hyper_inc)

    # Place the incremental hypergraph partition with pairwise QAP and multicast-aware anytime placement.
    perm_hyper_pairwise, info_hyper_pairwise = pairwise_placement_search(
        outn, labels_hyper_inc, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 303, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_hyper_inc); extra.update(info_hyper_pairwise.as_dict())
    add("mhysmap_partition_incremental_pairwise_qap", labels_hyper_inc, perm_hyper_pairwise, extra)

    perm_cert, metrics_cert, cert = anytime_placement_search(
        outn, labels_hyper_inc, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 33, exact_threshold_k=exact_threshold_k,
    )
    m_cert = metrics_cert.as_dict(); m_cert.update(info_hyper_inc); m_cert.update(cert.as_dict())
    rows_out["mhysmap_partition_incremental_multicast_certified"] = m_cert

    if include_full_joint_reference:
        labels_joint_full, perm_joint_full, metrics_joint_full, joint_info_full, joint_cert_full = joint_partition_placement_refinement(
            outn, inn, labels_wedge, rates, rows, cols, weights=weights,
            cycles=joint_cycles,
            partition_passes_per_cycle=max(1, passes_hyper // 2),
            slack=slack,
            placement_time_per_cycle_s=max(0.25, placement_time_limit_s / max(joint_cycles + 1, 1)),
            placement_restarts_per_cycle=max(4, placement_restarts // max(joint_cycles + 1, 1)),
            seed=seed + 505,
            exact_threshold_k=exact_threshold_k,
        )
        m_joint_full = metrics_joint_full.as_dict(); m_joint_full.update(joint_info_full.as_dict()); m_joint_full.update(joint_cert_full.as_dict())
        rows_out["mhysmap_joint_full_recompute"] = m_joint_full

    labels_joint_inc, perm_joint_inc, metrics_joint_inc, joint_info_inc, joint_cert_inc = joint_partition_placement_refinement_incremental(
        outn, inn, labels_wedge, rates, rows, cols, weights=weights,
        cycles=joint_cycles,
        partition_passes_per_cycle=max(1, passes_hyper // 2),
        slack=slack,
        placement_time_per_cycle_s=max(0.25, placement_time_limit_s / max(joint_cycles + 1, 1)),
        placement_restarts_per_cycle=max(4, placement_restarts // max(joint_cycles + 1, 1)),
        seed=seed + 505,
        exact_threshold_k=exact_threshold_k,
    )
    m_joint_inc = metrics_joint_inc.as_dict(); m_joint_inc.update(joint_info_inc.as_dict()); m_joint_inc.update(joint_cert_inc.as_dict())
    rows_out["mhysmap_joint_incremental"] = m_joint_inc

    return rows_out

# -----------------------------------------------------------------------------
# v0.8 paper-lite guard: schedule-swept joint refinement + safe incumbent selection
# -----------------------------------------------------------------------------

def run_mhysmap_core_v08(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
    joint_cycles: int = 3,
    include_full_joint_reference: bool = False,
    enable_schedule_sweep: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    v0.8 baseline suite.

    The v0.7 suite exposed that alternating joint refinement can be schedule-sensitive,
    especially on larger meshes. v0.8 keeps the same objective and incremental evaluator,
    but adds two paper-friendly safeguards:

      1) A tiny schedule sweep over joint partition-placement budgets.
      2) A safe incumbent selector that reports the best among sequential M-HySMap and
         joint variants, so the final method never degrades if joint alternation stalls.

    This is not a different problem formulation; it is a robust search policy around the
    locked activity-weighted multicast hypergraph objective.
    """
    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)
    rows_out: Dict[str, Dict[str, float]] = {}

    def add(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    add("random_rowmajor", labels_random, row_perm)

    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    add("edgecut_refined_rowmajor", labels_edge, row_perm, info_edge)
    perm_edge_qap, info_edge_qap = pairwise_placement_search(
        outn, labels_edge, rates=None, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 101, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_edge); extra.update(info_edge_qap.as_dict())
    add("edgecut_refined_pairwise_qap", labels_edge, perm_edge_qap, extra)

    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    add("activity_edge_refined_rowmajor", labels_wedge, row_perm, info_wedge)
    perm_wedge_qap, info_wedge_qap = pairwise_placement_search(
        outn, labels_wedge, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 202, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_wedge); extra.update(info_wedge_qap.as_dict())
    add("activity_edge_refined_pairwise_qap", labels_wedge, perm_wedge_qap, extra)

    # v1.0 hybrid: start from the strong activity-edge + pairwise-QAP solution,
    # then do multicast-hypergraph neuron moves under that physical placement.
    # This directly answers the large-mesh failure mode exposed in v0.9.
    labels_activity_qap_hyper, metrics_activity_qap_hyper, info_activity_qap_hyper = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=perm_wedge_qap,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 404,
    )
    perm_activity_qap_hyper, metrics_activity_qap_hyper_placed, cert_activity_qap_hyper = anytime_placement_search_v10(
        outn, labels_activity_qap_hyper, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 405, exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_wedge_qap, perm_edge_qap],
    )
    m_activity_qap_hyper = metrics_activity_qap_hyper_placed.as_dict(); m_activity_qap_hyper.update(info_activity_qap_hyper); m_activity_qap_hyper.update(cert_activity_qap_hyper.as_dict())
    rows_out["mhysmap_activity_qap_seeded_hyper_refine"] = m_activity_qap_hyper

    labels_hyper_full, _, info_hyper_full = refine_by_objective(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        objective="multicast", weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_full_recompute_rowmajor", labels_hyper_full, row_perm, info_hyper_full)

    labels_hyper_inc, _, info_hyper_inc = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_incremental_rowmajor", labels_hyper_inc, row_perm, info_hyper_inc)

    perm_hyper_pairwise, info_hyper_pairwise = pairwise_placement_search(
        outn, labels_hyper_inc, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 303, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_hyper_inc); extra.update(info_hyper_pairwise.as_dict())
    add("mhysmap_partition_incremental_pairwise_qap", labels_hyper_inc, perm_hyper_pairwise, extra)

    perm_cert, metrics_cert, cert = anytime_placement_search(
        outn, labels_hyper_inc, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 33, exact_threshold_k=exact_threshold_k,
    )
    m_cert = metrics_cert.as_dict(); m_cert.update(info_hyper_inc); m_cert.update(cert.as_dict())
    rows_out["mhysmap_partition_incremental_multicast_certified"] = m_cert

    if include_full_joint_reference:
        labels_joint_full, perm_joint_full, metrics_joint_full, joint_info_full, joint_cert_full = joint_partition_placement_refinement(
            outn, inn, labels_wedge, rates, rows, cols, weights=weights,
            cycles=joint_cycles,
            partition_passes_per_cycle=max(1, passes_hyper // 2),
            slack=slack,
            placement_time_per_cycle_s=max(0.25, placement_time_limit_s / max(joint_cycles + 1, 1)),
            placement_restarts_per_cycle=max(4, placement_restarts // max(joint_cycles + 1, 1)),
            seed=seed + 505,
            exact_threshold_k=exact_threshold_k,
        )
        m_joint_full = metrics_joint_full.as_dict(); m_joint_full.update(joint_info_full.as_dict()); m_joint_full.update(joint_cert_full.as_dict())
        rows_out["mhysmap_joint_full_recompute"] = m_joint_full

    # Default alternating schedule, matching v0.7/v0.6.
    labels_joint_default, perm_joint_default, metrics_joint_default, joint_info_default, joint_cert_default = joint_partition_placement_refinement_incremental(
        outn, inn, labels_wedge, rates, rows, cols, weights=weights,
        cycles=joint_cycles,
        partition_passes_per_cycle=max(1, passes_hyper // 2),
        slack=slack,
        placement_time_per_cycle_s=max(0.25, placement_time_limit_s / max(joint_cycles + 1, 1)),
        placement_restarts_per_cycle=max(4, placement_restarts // max(joint_cycles + 1, 1)),
        seed=seed + 505,
        exact_threshold_k=exact_threshold_k,
    )
    m_joint_default = metrics_joint_default.as_dict(); m_joint_default.update(joint_info_default.as_dict()); m_joint_default.update(joint_cert_default.as_dict())
    rows_out["mhysmap_joint_incremental_default"] = m_joint_default
    # Keep the v0.7 name as an alias for compatibility with old plotting scripts.
    rows_out["mhysmap_joint_incremental"] = dict(m_joint_default)

    # v0.8 schedule variants: a little more placement budget and/or more alternation.
    schedule_candidates: List[Tuple[str, int, int, float, int, int]] = []
    if enable_schedule_sweep:
        schedule_candidates = [
            # name, cycles, partition_passes, placement_time_per_cycle, restarts_per_cycle, seed_offset
            ("mhysmap_joint_incremental_placeheavy", max(2, joint_cycles), max(1, passes_hyper // 2),
             max(0.35, placement_time_limit_s), max(6, placement_restarts * 2), 707),
            ("mhysmap_joint_incremental_deeper", max(3, joint_cycles + 1), max(1, passes_hyper // 2),
             max(0.25, placement_time_limit_s / max(joint_cycles, 1)), max(5, placement_restarts), 909),
        ]

    for name, cyc, part_passes, place_time, restarts_cycle, seed_offset in schedule_candidates:
        labels_j, perm_j, metrics_j, info_j, cert_j = joint_partition_placement_refinement_incremental(
            outn, inn, labels_wedge, rates, rows, cols, weights=weights,
            cycles=cyc,
            partition_passes_per_cycle=part_passes,
            slack=slack,
            placement_time_per_cycle_s=place_time,
            placement_restarts_per_cycle=restarts_cycle,
            seed=seed + seed_offset,
            exact_threshold_k=exact_threshold_k,
        )
        m_j = metrics_j.as_dict(); m_j.update(info_j.as_dict()); m_j.update(cert_j.as_dict())
        # make schedule metadata explicit in CSV
        m_j["v08_schedule_cycles"] = float(cyc)
        m_j["v08_schedule_partition_passes"] = float(part_passes)
        m_j["v08_schedule_placement_time_per_cycle_s"] = float(place_time)
        m_j["v08_schedule_restarts_per_cycle"] = float(restarts_cycle)
        rows_out[name] = m_j

    # Safe final method: take the best hardware-facing objective among sequential M-HySMap
    # and all joint variants. This avoids the k=25 pathology where joint alternation can
    # occasionally hurt if the schedule budget is unlucky.
    candidate_names = [
        "mhysmap_partition_incremental_multicast_certified",
        "mhysmap_joint_incremental_default",
        "mhysmap_joint_incremental_placeheavy",
        "mhysmap_joint_incremental_deeper",
    ]
    candidates = [(nm, rows_out[nm]) for nm in candidate_names if nm in rows_out]
    best_name, best_row = min(candidates, key=lambda kv: float(kv[1].get("objective", float("inf"))))
    safe = dict(best_row)
    safe["v08_safe_selected_code"] = float({
        "mhysmap_partition_incremental_multicast_certified": 1,
        "mhysmap_joint_incremental_default": 2,
        "mhysmap_joint_incremental_placeheavy": 3,
        "mhysmap_joint_incremental_deeper": 4,
    }.get(best_name, 0))
    safe["v08_safe_selected_name_hash"] = float(abs(hash(best_name)) % 1000000)
    safe["v08_safe_candidates"] = float(len(candidates))
    safe["v08_safe_best_is_joint"] = float(0 if best_name == "mhysmap_partition_incremental_multicast_certified" else 1)
    rows_out["mhysmap_v08_safe_best"] = safe

    return rows_out


# v0.9 alias: evidence-suite release keeps the v0.8 safe-best core.
def run_mhysmap_core_v09(*args, **kwargs):
    return run_mhysmap_core_v08(*args, **kwargs)


# -----------------------------------------------------------------------------
# v1.0: placement-portfolio hardening
# -----------------------------------------------------------------------------

def anytime_placement_search_v10(
    out_neighbors: List[np.ndarray],
    labels: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    time_limit_s: float = 2.0,
    restarts: int = 16,
    seed: int = 0,
    exact_threshold_k: int = 8,
    initial_perms: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, MeshMetrics, PlacementCert]:
    """
    v1.0 multicast placement search with a portfolio of strong initial placements.

    The v0.9 anytime mapper started from row-major and random permutations. On larger
    meshes, a pairwise-QAP placement can already be a very strong point under the actual
    multicast objective. v1.0 therefore admits user-provided initial permutations and then
    performs multicast-aware local swap refinement from all of them.
    """
    k = int(labels.max()) + 1
    if k <= exact_threshold_k:
        return exact_placement_bruteforce(
            out_neighbors, labels, rates, rows, cols, weights=weights,
            time_limit_s=time_limit_s, fix_core0=True,
        )

    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    lb = placement_lower_bound(out_neighbors, labels, rates, rows, cols, weights)
    best_perm = np.arange(k, dtype=np.int32)
    best_metrics = multicast_metrics(out_neighbors, labels, rates, perm=best_perm, rows=rows, cols=cols, weights=weights)
    evals_total = 0

    init_perms: List[np.ndarray] = []
    seen = set()

    def add_perm(p):
        arr = np.asarray(p, dtype=np.int32)
        if len(arr) != k:
            return
        # Keep only valid permutations over at least k mesh slots.
        if len(set(map(int, arr))) != k:
            return
        key = tuple(map(int, arr))
        if key not in seen:
            seen.add(key)
            init_perms.append(arr.copy())

    # Strong seeds first. In v0.9/v1.0 testing, row-major could consume the whole
    # tiny placement budget before the pairwise-QAP seed was even evaluated on large meshes.
    # Therefore user-provided seeds get priority, then row-major, then random restarts.
    if initial_perms:
        for p in initial_perms:
            add_perm(p)
    add_perm(np.arange(k, dtype=np.int32))
    # Random restarts fill the remaining budget.
    while len(init_perms) < max(1, restarts):
        p = np.arange(k, dtype=np.int32)
        rng.shuffle(p)
        add_perm(p)
        # Extremely unlikely, but avoid infinite loop if restarts is huge and k tiny.
        if len(seen) > max(restarts * 4, 64):
            break

    for p0 in init_perms:
        if (time.perf_counter() - t0) > time_limit_s:
            break
        perm, metrics, evals = local_swap_improve_from_perm(
            out_neighbors, labels, rates, rows, cols, p0,
            weights=weights, max_passes=3, time_limit_s=time_limit_s, t0=t0,
        )
        evals_total += evals
        if metrics.objective < best_metrics.objective - 1e-12:
            best_perm, best_metrics = perm.copy(), metrics

    inc = float(best_metrics.objective)
    abs_gap = max(0.0, inc - lb)
    rel_gap = abs_gap / max(abs(lb), 1e-12)
    cert = PlacementCert(
        status="feasible",
        incumbent_objective=inc,
        lower_bound=float(lb),
        absolute_gap=float(abs_gap),
        relative_gap=float(rel_gap),
        permutations_evaluated=int(evals_total),
        runtime_s=float(time.perf_counter() - t0),
        method="anytime_seeded_portfolio",
    )
    return best_perm, best_metrics, cert


def joint_partition_placement_refinement_incremental_v10(
    out_neighbors: List[np.ndarray],
    in_neighbors: List[np.ndarray],
    labels0: np.ndarray,
    rates: np.ndarray,
    rows: int,
    cols: int,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    cycles: int = 3,
    partition_passes_per_cycle: int = 2,
    slack: float = 0.15,
    placement_time_per_cycle_s: float = 1.0,
    placement_restarts_per_cycle: int = 12,
    seed: int = 0,
    exact_threshold_k: int = 8,
    seed_pairwise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, MeshMetrics, JointRefineInfoV06, PlacementCert]:
    """v1.0 alternating refinement with seeded multicast placement search."""
    t0 = time.perf_counter()
    labels = labels0.astype(np.int32, copy=True)
    k = int(labels.max()) + 1
    perm = np.arange(k, dtype=np.int32)
    init_metrics = multicast_metrics(out_neighbors, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights)
    initial_objective = float(init_metrics.objective)
    best_labels = labels.copy()
    best_perm = perm.copy()
    best_metrics = init_metrics

    total_moves = 0
    total_part_evals = 0
    total_place_evals = 0
    restarts_used = 0
    part_runtime = 0.0
    affected_weighted_sum = 0.0
    affected_den = 0.0
    affected_max = 0.0
    validation_error = 0.0
    final_cert: Optional[PlacementCert] = None
    completed_cycles = 0

    for cyc in range(cycles):
        completed_cycles = cyc + 1
        initial_perms = [perm.copy(), best_perm.copy()]
        if seed_pairwise:
            try:
                p_pair, _ = pairwise_placement_search(
                    out_neighbors, labels, rates=rates, rows=rows, cols=cols,
                    time_limit_s=max(0.05, placement_time_per_cycle_s * 0.35),
                    restarts=max(2, placement_restarts_per_cycle // 2),
                    seed=seed + 8000 + 19 * cyc,
                    exact_threshold_k=exact_threshold_k,
                )
                initial_perms.append(p_pair)
            except Exception:
                pass
        perm, placed_metrics, cert = anytime_placement_search_v10(
            out_neighbors, labels, rates, rows, cols, weights=weights,
            time_limit_s=placement_time_per_cycle_s,
            restarts=placement_restarts_per_cycle,
            seed=seed + 1000 + 17 * cyc,
            exact_threshold_k=exact_threshold_k,
            initial_perms=initial_perms,
        )
        final_cert = cert
        total_place_evals += int(cert.permutations_evaluated)
        restarts_used += int(placement_restarts_per_cycle)
        if placed_metrics.objective < best_metrics.objective - 1e-12:
            best_labels = labels.copy(); best_perm = perm.copy(); best_metrics = placed_metrics

        labels_new, part_metrics, info = refine_multicast_incremental(
            out_neighbors, in_neighbors, labels, rates, rows, cols, perm=perm,
            weights=weights, passes=partition_passes_per_cycle, slack=slack,
            seed=seed + 2000 + 31 * cyc,
        )
        total_moves += int(info.get("moves", 0.0))
        total_part_evals += int(info.get("evaluated_moves", 0.0))
        part_runtime += float(info.get("runtime_s", 0.0))
        evals = max(float(info.get("evaluated_moves", 0.0)), 1.0)
        affected_weighted_sum += float(info.get("incremental_avg_affected_sources", 0.0)) * evals
        affected_den += evals
        affected_max = max(affected_max, float(info.get("incremental_max_affected_sources", 0.0)))
        validation_error = max(validation_error, float(info.get("incremental_validation_abs_objective_error", 0.0)))

        labels = labels_new
        if part_metrics.objective < best_metrics.objective - 1e-12:
            best_labels = labels.copy(); best_perm = perm.copy(); best_metrics = part_metrics
        if int(info.get("moves", 0.0)) == 0 and cyc > 0:
            break

    final_initial_perms = [best_perm.copy()]
    if seed_pairwise:
        try:
            p_pair, _ = pairwise_placement_search(
                out_neighbors, best_labels, rates=rates, rows=rows, cols=cols,
                time_limit_s=max(0.05, placement_time_per_cycle_s * 0.5),
                restarts=max(3, placement_restarts_per_cycle),
                seed=seed + 9991,
                exact_threshold_k=exact_threshold_k,
            )
            final_initial_perms.append(p_pair)
        except Exception:
            pass
    perm_final, metrics_final, cert_final = anytime_placement_search_v10(
        out_neighbors, best_labels, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_per_cycle_s,
        restarts=max(placement_restarts_per_cycle, 16),
        seed=seed + 9999,
        exact_threshold_k=exact_threshold_k,
        initial_perms=final_initial_perms,
    )
    final_cert = cert_final
    total_place_evals += int(cert_final.permutations_evaluated)
    if metrics_final.objective < best_metrics.objective - 1e-12:
        best_perm = perm_final.copy(); best_metrics = metrics_final

    final_objective = float(best_metrics.objective)
    improvement_pct = (initial_objective - final_objective) / max(abs(initial_objective), 1e-12) * 100.0
    info = JointRefineInfoV06(
        cycles=completed_cycles,
        partition_moves=total_moves,
        partition_evals=total_part_evals,
        placement_evals=total_place_evals,
        placement_restarts=restarts_used,
        runtime_s=float(time.perf_counter() - t0),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        objective_improvement_pct=float(improvement_pct),
        partition_runtime_s=float(part_runtime),
        avg_affected_sources=float(affected_weighted_sum / max(affected_den, 1.0)),
        max_affected_sources=float(affected_max),
        validation_abs_objective_error=float(validation_error),
    )
    assert final_cert is not None
    return best_labels, best_perm, best_metrics, info, final_cert


def run_mhysmap_core_v10(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
    joint_cycles: int = 3,
    include_full_joint_reference: bool = False,
    enable_schedule_sweep: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    v1.0 evidence core.

    New policy: placement portfolio. The final M-HySMap method is allowed to keep the
    best multicast objective among multicast-seeded placement, pairwise-QAP placement,
    and joint refinements. This fixes the large-mesh failure mode where a strong pairwise
    placement seed was accidentally excluded from the safe-best set.
    """
    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)
    rows_out: Dict[str, Dict[str, float]] = {}

    def add(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    add("random_rowmajor", labels_random, row_perm)

    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    add("edgecut_refined_rowmajor", labels_edge, row_perm, info_edge)
    perm_edge_qap, info_edge_qap = pairwise_placement_search(
        outn, labels_edge, rates=None, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 101, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_edge); extra.update(info_edge_qap.as_dict())
    add("edgecut_refined_pairwise_qap", labels_edge, perm_edge_qap, extra)

    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    add("activity_edge_refined_rowmajor", labels_wedge, row_perm, info_wedge)
    perm_wedge_qap, info_wedge_qap = pairwise_placement_search(
        outn, labels_wedge, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 202, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_wedge); extra.update(info_wedge_qap.as_dict())
    add("activity_edge_refined_pairwise_qap", labels_wedge, perm_wedge_qap, extra)

    # v1.0 hybrid: start from the strong activity-edge + pairwise-QAP solution,
    # then do multicast-hypergraph neuron moves under that physical placement.
    # This directly answers the large-mesh failure mode exposed in v0.9.
    labels_activity_qap_hyper, metrics_activity_qap_hyper, info_activity_qap_hyper = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=perm_wedge_qap,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 404,
    )
    perm_activity_qap_hyper, metrics_activity_qap_hyper_placed, cert_activity_qap_hyper = anytime_placement_search_v10(
        outn, labels_activity_qap_hyper, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 405, exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_wedge_qap, perm_edge_qap],
    )
    m_activity_qap_hyper = metrics_activity_qap_hyper_placed.as_dict(); m_activity_qap_hyper.update(info_activity_qap_hyper); m_activity_qap_hyper.update(cert_activity_qap_hyper.as_dict())
    rows_out["mhysmap_activity_qap_seeded_hyper_refine"] = m_activity_qap_hyper

    labels_hyper_full, _, info_hyper_full = refine_by_objective(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        objective="multicast", weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_full_recompute_rowmajor", labels_hyper_full, row_perm, info_hyper_full)

    labels_hyper_inc, _, info_hyper_inc = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=row_perm,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 3,
    )
    add("mhysmap_partition_incremental_rowmajor", labels_hyper_inc, row_perm, info_hyper_inc)

    perm_hyper_pairwise, info_hyper_pairwise = pairwise_placement_search(
        outn, labels_hyper_inc, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 303, exact_threshold_k=exact_threshold_k,
    )
    extra = dict(info_hyper_inc); extra.update(info_hyper_pairwise.as_dict())
    add("mhysmap_partition_incremental_pairwise_qap", labels_hyper_inc, perm_hyper_pairwise, extra)

    # v1.0 seeded multicast-aware placement: start from the strong pairwise placements,
    # then optimize the true multicast/congestion objective by local swaps.
    perm_seeded, metrics_seeded, cert_seeded = anytime_placement_search_v10(
        outn, labels_hyper_inc, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 33, exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_hyper_pairwise, perm_wedge_qap, perm_edge_qap],
    )
    m_seeded = metrics_seeded.as_dict(); m_seeded.update(info_hyper_inc); m_seeded.update(cert_seeded.as_dict())
    rows_out["mhysmap_partition_incremental_seeded_multicast"] = m_seeded
    # Compatibility name from older versions, now backed by the seeded portfolio.
    rows_out["mhysmap_partition_incremental_multicast_certified"] = dict(m_seeded)

    if include_full_joint_reference:
        labels_joint_full, perm_joint_full, metrics_joint_full, joint_info_full, joint_cert_full = joint_partition_placement_refinement(
            outn, inn, labels_wedge, rates, rows, cols, weights=weights,
            cycles=joint_cycles,
            partition_passes_per_cycle=max(1, passes_hyper // 2),
            slack=slack,
            placement_time_per_cycle_s=max(0.25, placement_time_limit_s / max(joint_cycles + 1, 1)),
            placement_restarts_per_cycle=max(4, placement_restarts // max(joint_cycles + 1, 1)),
            seed=seed + 505,
            exact_threshold_k=exact_threshold_k,
        )
        m_joint_full = metrics_joint_full.as_dict(); m_joint_full.update(joint_info_full.as_dict()); m_joint_full.update(joint_cert_full.as_dict())
        rows_out["mhysmap_joint_full_recompute"] = m_joint_full

    labels_joint_default, perm_joint_default, metrics_joint_default, joint_info_default, joint_cert_default = joint_partition_placement_refinement_incremental_v10(
        outn, inn, labels_wedge, rates, rows, cols, weights=weights,
        cycles=joint_cycles,
        partition_passes_per_cycle=max(1, passes_hyper // 2),
        slack=slack,
        placement_time_per_cycle_s=max(0.25, placement_time_limit_s / max(joint_cycles + 1, 1)),
        placement_restarts_per_cycle=max(4, placement_restarts // max(joint_cycles + 1, 1)),
        seed=seed + 505,
        exact_threshold_k=exact_threshold_k,
    )
    m_joint_default = metrics_joint_default.as_dict(); m_joint_default.update(joint_info_default.as_dict()); m_joint_default.update(joint_cert_default.as_dict())
    rows_out["mhysmap_joint_incremental_default"] = m_joint_default
    rows_out["mhysmap_joint_incremental"] = dict(m_joint_default)

    schedule_candidates: List[Tuple[str, int, int, float, int, int]] = []
    if enable_schedule_sweep:
        schedule_candidates = [
            ("mhysmap_joint_incremental_placeheavy", max(2, joint_cycles), max(1, passes_hyper // 2),
             max(0.35, placement_time_limit_s), max(6, placement_restarts * 2), 707),
            ("mhysmap_joint_incremental_deeper", max(3, joint_cycles + 1), max(1, passes_hyper // 2),
             max(0.25, placement_time_limit_s / max(joint_cycles, 1)), max(5, placement_restarts), 909),
        ]

    for name, cyc, part_passes, place_time, restarts_cycle, seed_offset in schedule_candidates:
        labels_j, perm_j, metrics_j, info_j, cert_j = joint_partition_placement_refinement_incremental_v10(
            outn, inn, labels_wedge, rates, rows, cols, weights=weights,
            cycles=cyc,
            partition_passes_per_cycle=part_passes,
            slack=slack,
            placement_time_per_cycle_s=place_time,
            placement_restarts_per_cycle=restarts_cycle,
            seed=seed + seed_offset,
            exact_threshold_k=exact_threshold_k,
        )
        m_j = metrics_j.as_dict(); m_j.update(info_j.as_dict()); m_j.update(cert_j.as_dict())
        m_j["v10_schedule_cycles"] = float(cyc)
        m_j["v10_schedule_partition_passes"] = float(part_passes)
        m_j["v10_schedule_placement_time_per_cycle_s"] = float(place_time)
        m_j["v10_schedule_restarts_per_cycle"] = float(restarts_cycle)
        rows_out[name] = m_j

    candidate_names = [
        "mhysmap_activity_qap_seeded_hyper_refine",
        "mhysmap_partition_incremental_pairwise_qap",
        "mhysmap_partition_incremental_seeded_multicast",
        "mhysmap_joint_incremental_default",
        "mhysmap_joint_incremental_placeheavy",
        "mhysmap_joint_incremental_deeper",
    ]
    candidates = [(nm, rows_out[nm]) for nm in candidate_names if nm in rows_out]
    best_name, best_row = min(candidates, key=lambda kv: float(kv[1].get("objective", float("inf"))))
    safe = dict(best_row)
    safe["v10_safe_selected_code"] = float({
        "mhysmap_activity_qap_seeded_hyper_refine": 1,
        "mhysmap_partition_incremental_pairwise_qap": 2,
        "mhysmap_partition_incremental_seeded_multicast": 3,
        "mhysmap_joint_incremental_default": 4,
        "mhysmap_joint_incremental_placeheavy": 5,
        "mhysmap_joint_incremental_deeper": 6,
    }.get(best_name, 0))
    safe["v10_safe_selected_name_hash"] = float(abs(hash(best_name)) % 1000000)
    safe["v10_safe_candidates"] = float(len(candidates))
    safe["v10_safe_best_is_joint"] = float(1 if "joint" in best_name else 0)
    safe["v10_safe_best_uses_pairwise_seed"] = float(1 if best_name in ("mhysmap_activity_qap_seeded_hyper_refine", "mhysmap_partition_incremental_pairwise_qap", "mhysmap_partition_incremental_seeded_multicast") else 0)
    rows_out["mhysmap_v10_portfolio_best"] = safe
    # Compatibility alias for old aggregation scripts.
    rows_out["mhysmap_v08_safe_best"] = dict(safe)
    return rows_out


# -----------------------------------------------------------------------------
# v1.3 optional add-on: congestion/injection smoothing portfolio
# -----------------------------------------------------------------------------

def run_mhysmap_core_v13(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
    joint_cycles: int = 3,
    include_full_joint_reference: bool = False,
    enable_schedule_sweep: bool = True,
    enable_congestion_addon: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    v1.3 optional add-on around the stable v1.0/v1.2 evidence core.

    Motivation: the heavy v1.2 runs showed hop/objective wins are already strong, while
    max-link congestion is the most meaningful remaining improvement direction. v1.3 does
    NOT replace the locked M-HySMap algorithm. It adds a small, safe portfolio arm that
    reuses the strong Activity+QAP physical seed and tries a more congestion/injection-aware
    multicast refinement. The final method keeps the best original objective among v1.0 and
    the new variants, so this add-on cannot hurt the reported primary objective.

    New rows:
      - mhysmap_v13_congestion_place_polish: same partition style, congestion-biased placement search.
      - mhysmap_v13_injection_smooth_refine: neuron moves with rate-imbalance penalty.
      - mhysmap_v13_congestion_injection_refine: neuron moves with stronger link/injection smoothing.
      - mhysmap_v13_final_best: safe best over v10 portfolio and v13 variants under the original objective.
    """
    rows_out = run_mhysmap_core_v10(
        A_dir, rates, k=k, rows=rows, cols=cols, seed=seed,
        passes_edge=passes_edge, passes_hyper=passes_hyper, slack=slack,
        weights=weights, placement_time_limit_s=placement_time_limit_s,
        placement_restarts=placement_restarts, exact_threshold_k=exact_threshold_k,
        joint_cycles=joint_cycles, include_full_joint_reference=include_full_joint_reference,
        enable_schedule_sweep=enable_schedule_sweep,
    )
    if not enable_congestion_addon:
        rows_out["mhysmap_v13_final_best"] = dict(rows_out["mhysmap_v10_portfolio_best"])
        return rows_out

    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)

    # Reconstruct the strong graph seed used by v1.0. We keep this duplicated instead of
    # refactoring v1.0 so the v1.3 add-on is minimally invasive and easy to disable.
    labels_edge, _, info_edge = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    perm_edge_qap, info_edge_qap = pairwise_placement_search(
        outn, labels_edge, rates=None, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 101, exact_threshold_k=exact_threshold_k,
    )
    labels_wedge, _, info_wedge = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    perm_wedge_qap, info_wedge_qap = pairwise_placement_search(
        outn, labels_wedge, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 202, exact_threshold_k=exact_threshold_k,
    )

    def add_variant(name: str, labels: np.ndarray, perm: np.ndarray, extra: Optional[Dict[str, float]] = None):
        # Always evaluate the published metrics/objective under the original weights.
        m = multicast_metrics(outn, labels, rates, perm=perm, rows=rows, cols=cols, weights=weights).as_dict()
        if extra:
            m.update(extra)
        rows_out[name] = m

    # A light hypergraph refinement under the strong physical seed. This gives a useful
    # starting point for congestion placement polishing.
    labels_seeded, _, info_seeded = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=perm_wedge_qap,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 404,
    )

    # Variant 1: placement-only congestion polish. Search uses a stronger congestion weight,
    # but we report under the original objective. This is low-risk and often improves max-link.
    w_place_cong = MeshObjectiveWeights(
        alpha_hops=1.0,
        beta_max_link=max(weights.beta_max_link * 4.0, 0.40),
        gamma_link_var=max(weights.gamma_link_var * 3.0, 0.03),
        delta_size_imbalance=weights.delta_size_imbalance,
        delta_rate_imbalance=weights.delta_rate_imbalance,
    )
    perm_cpolish, metrics_cpolish_search, cert_cpolish = anytime_placement_search_v10(
        outn, labels_seeded, rates, rows, cols, weights=w_place_cong,
        time_limit_s=max(placement_time_limit_s, 0.45),
        restarts=max(placement_restarts * 2, 8),
        seed=seed + 1301,
        exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_wedge_qap, perm_edge_qap],
    )
    extra = dict(info_seeded); extra.update(cert_cpolish.as_dict(prefix="v13_cong_place_cert_"))
    extra["v13_search_beta_max_link"] = float(w_place_cong.beta_max_link)
    extra["v13_search_gamma_link_var"] = float(w_place_cong.gamma_link_var)
    add_variant("mhysmap_v13_congestion_place_polish", labels_seeded, perm_cpolish, extra)

    # Variant 2: injection smoothing. Add a source-rate imbalance penalty during neuron moves,
    # then place with the standard seeded multicast placement. This targets hot source cores.
    w_inject = MeshObjectiveWeights(
        alpha_hops=1.0,
        beta_max_link=weights.beta_max_link,
        gamma_link_var=weights.gamma_link_var,
        delta_size_imbalance=weights.delta_size_imbalance,
        delta_rate_imbalance=max(20.0, weights.delta_rate_imbalance),
    )
    labels_inj, _, info_inj = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=perm_wedge_qap,
        weights=w_inject, passes=max(1, passes_hyper), slack=slack, seed=seed + 1311,
    )
    perm_inj, metrics_inj_search, cert_inj = anytime_placement_search_v10(
        outn, labels_inj, rates, rows, cols, weights=weights,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 1312, exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_wedge_qap, perm_edge_qap],
    )
    extra = dict(info_inj); extra.update(cert_inj.as_dict(prefix="v13_injection_cert_"))
    extra["v13_search_delta_rate_imbalance"] = float(w_inject.delta_rate_imbalance)
    add_variant("mhysmap_v13_injection_smooth_refine", labels_inj, perm_inj, extra)

    # Variant 3: combined congestion + injection refinement. More aggressive, but still safe
    # because final_best only keeps it if it improves the original objective.
    w_combo = MeshObjectiveWeights(
        alpha_hops=1.0,
        beta_max_link=max(weights.beta_max_link * 3.0, 0.30),
        gamma_link_var=max(weights.gamma_link_var * 2.0, 0.02),
        delta_size_imbalance=weights.delta_size_imbalance,
        delta_rate_imbalance=max(10.0, weights.delta_rate_imbalance),
    )
    labels_combo, _, info_combo = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=perm_wedge_qap,
        weights=w_combo, passes=max(1, passes_hyper), slack=slack, seed=seed + 1321,
    )
    perm_combo, metrics_combo_search, cert_combo = anytime_placement_search_v10(
        outn, labels_combo, rates, rows, cols, weights=w_combo,
        time_limit_s=max(placement_time_limit_s, 0.45),
        restarts=max(placement_restarts * 2, 8),
        seed=seed + 1322, exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_wedge_qap, perm_edge_qap, perm_cpolish],
    )
    extra = dict(info_combo); extra.update(cert_combo.as_dict(prefix="v13_combo_cert_"))
    extra["v13_search_beta_max_link"] = float(w_combo.beta_max_link)
    extra["v13_search_gamma_link_var"] = float(w_combo.gamma_link_var)
    extra["v13_search_delta_rate_imbalance"] = float(w_combo.delta_rate_imbalance)
    add_variant("mhysmap_v13_congestion_injection_refine", labels_combo, perm_combo, extra)

    # Safe final selector under the original M-HySMap objective.
    candidate_names = [
        "mhysmap_v10_portfolio_best",
        "mhysmap_v13_congestion_place_polish",
        "mhysmap_v13_injection_smooth_refine",
        "mhysmap_v13_congestion_injection_refine",
    ]
    candidates = [(nm, rows_out[nm]) for nm in candidate_names if nm in rows_out]
    best_name, best_row = min(candidates, key=lambda kv: float(kv[1].get("objective", float("inf"))))
    safe = dict(best_row)
    safe["v13_selected_code"] = float({
        "mhysmap_v10_portfolio_best": 10,
        "mhysmap_v13_congestion_place_polish": 13.1,
        "mhysmap_v13_injection_smooth_refine": 13.2,
        "mhysmap_v13_congestion_injection_refine": 13.3,
    }.get(best_name, 0.0))
    safe["v13_selected_name_hash"] = float(abs(hash(best_name)) % 1000000)
    safe["v13_candidates"] = float(len(candidates))
    safe["v13_addon_selected"] = float(0 if best_name == "mhysmap_v10_portfolio_best" else 1)
    rows_out["mhysmap_v13_final_best"] = safe
    # Compatibility alias: downstream scripts can keep using final-method style names if desired.
    rows_out["mhysmap_final_best"] = dict(safe)
    return rows_out

# -----------------------------------------------------------------------------
# v1.3 streamlined override: keep the add-on paper-safe and runtime-safe.
# -----------------------------------------------------------------------------
def run_mhysmap_core_v13(
    A_dir: sp.csr_matrix,
    rates: np.ndarray,
    k: int = 16,
    rows: int = 4,
    cols: int = 4,
    seed: int = 0,
    passes_edge: int = 3,
    passes_hyper: int = 4,
    slack: float = 0.12,
    weights: MeshObjectiveWeights = MeshObjectiveWeights(),
    placement_time_limit_s: float = 2.0,
    placement_restarts: int = 16,
    exact_threshold_k: int = 8,
    joint_cycles: int = 3,
    include_full_joint_reference: bool = False,
    enable_schedule_sweep: bool = True,
    enable_congestion_addon: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    v1.3 optional add-on, streamlined.

    Heavy v1.2 results already justify finalizing the paper. v1.3 adds only one low-risk
    optional arm: a congestion-biased placement polish seeded by the strong Activity+QAP
    physical placement. The reported objective is still the original objective. The final
    method keeps the better of v1.0 portfolio-best and the add-on, so the primary result
    cannot regress.
    """
    rows_out = run_mhysmap_core_v10(
        A_dir, rates, k=k, rows=rows, cols=cols, seed=seed,
        passes_edge=passes_edge, passes_hyper=passes_hyper, slack=slack,
        weights=weights, placement_time_limit_s=placement_time_limit_s,
        placement_restarts=placement_restarts, exact_threshold_k=exact_threshold_k,
        joint_cycles=joint_cycles, include_full_joint_reference=include_full_joint_reference,
        enable_schedule_sweep=enable_schedule_sweep,
    )
    if not enable_congestion_addon:
        rows_out["mhysmap_v13_final_best"] = dict(rows_out["mhysmap_v10_portfolio_best"])
        rows_out["mhysmap_final_best"] = dict(rows_out["mhysmap_v13_final_best"])
        return rows_out

    outn = csr_out_neighbors(A_dir)
    inn = csr_in_neighbors(A_dir)
    labels_random = balanced_random_labels(A_dir.shape[0], k, seed)
    row_perm = np.arange(k, dtype=np.int32)

    labels_edge, _, _ = refine_by_objective(
        outn, inn, labels_random, rates, rows, cols, perm=row_perm,
        objective="edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 1,
    )
    perm_edge_qap, _ = pairwise_placement_search(
        outn, labels_edge, rates=None, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 101, exact_threshold_k=exact_threshold_k,
    )
    labels_wedge, _, _ = refine_by_objective(
        outn, inn, labels_edge, rates, rows, cols, perm=row_perm,
        objective="weighted_edge", weights=weights, passes=passes_edge, slack=slack, seed=seed + 2,
    )
    perm_wedge_qap, _ = pairwise_placement_search(
        outn, labels_wedge, rates=rates, rows=rows, cols=cols,
        time_limit_s=placement_time_limit_s, restarts=placement_restarts,
        seed=seed + 202, exact_threshold_k=exact_threshold_k,
    )

    labels_seeded, _, info_seeded = refine_multicast_incremental(
        outn, inn, labels_wedge, rates, rows, cols, perm=perm_wedge_qap,
        weights=weights, passes=passes_hyper, slack=slack, seed=seed + 404,
    )
    w_cong = MeshObjectiveWeights(
        alpha_hops=1.0,
        beta_max_link=max(weights.beta_max_link * 4.0, 0.40),
        gamma_link_var=max(weights.gamma_link_var * 3.0, 0.03),
        delta_size_imbalance=weights.delta_size_imbalance,
        delta_rate_imbalance=weights.delta_rate_imbalance,
    )
    perm_cpolish, _, cert_cpolish = anytime_placement_search_v10(
        outn, labels_seeded, rates, rows, cols, weights=w_cong,
        time_limit_s=max(placement_time_limit_s, 0.35),
        restarts=max(placement_restarts, 4),
        seed=seed + 1301,
        exact_threshold_k=exact_threshold_k,
        initial_perms=[perm_wedge_qap, perm_edge_qap],
    )
    m = multicast_metrics(outn, labels_seeded, rates, perm=perm_cpolish, rows=rows, cols=cols, weights=weights).as_dict()
    m.update(info_seeded)
    m.update(cert_cpolish.as_dict(prefix="v13_cong_place_cert_"))
    m["v13_addon_type"] = 1.0
    m["v13_search_beta_max_link"] = float(w_cong.beta_max_link)
    m["v13_search_gamma_link_var"] = float(w_cong.gamma_link_var)
    rows_out["mhysmap_v13_congestion_place_polish"] = m

    candidates = [("mhysmap_v10_portfolio_best", rows_out["mhysmap_v10_portfolio_best"]),
                  ("mhysmap_v13_congestion_place_polish", rows_out["mhysmap_v13_congestion_place_polish"])]
    best_name, best_row = min(candidates, key=lambda kv: float(kv[1].get("objective", float("inf"))))
    safe = dict(best_row)
    safe["v13_selected_code"] = float(10 if best_name == "mhysmap_v10_portfolio_best" else 13.1)
    safe["v13_addon_selected"] = float(0 if best_name == "mhysmap_v10_portfolio_best" else 1)
    safe["v13_candidates"] = 2.0
    rows_out["mhysmap_v13_final_best"] = safe
    rows_out["mhysmap_final_best"] = dict(safe)
    return rows_out
