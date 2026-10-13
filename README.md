# M-HySMap

**Activity-weighted, route-aware multicast hypergraph mapping for spiking neural networks on mesh NoCs.**

This repository contains the M-HySMap mapper and experiment harness associated with the extended
preprint **Beyond Edge Cuts: Activity-Weighted Multicast Hypergraph Mapping for Spiking Neural
Networks on Mesh NoCs**.

The central representation change is to model each source neuron and its postsynaptic fanout as one
directed, activity-weighted hyperedge. Mapping cost is evaluated on the union of deterministic mesh
routes to distinct destination cores, so shared links for one source spike are counted once.

## Core algorithmic ideas

- activity-aware graph/QAP seeds;
- source-rooted multicast/hypergraph objective;
- routed-link union cost plus worst-link and link-load-variance terms;
- exact affected-source locality for neuron moves;
- incremental partition refinement that reroutes only the moved neuron and its predecessors;
- multicast-aware core-location 2-swaps;
- alternating partition/placement schedules;
- safe-best portfolio selection.

The v1.3 working snapshot also contains optional congestion-oriented portfolio arms. The locked
paper interpretation remains conservative: the main published method should not be silently replaced
by an optional add-on unless the corresponding evidence supports that change.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick smoke validation

```bash
python run_heavy_v12.py   --config configs/smoke_v12.json   --out-dir results/local_smoke   --timeout-per-run 300
```

A clean validation run of this GitHub-ready snapshot completed the single smoke job successfully
(return code 0).

Generate plots with:

```bash
python plot_heavy_v12.py --results-dir results/local_smoke
```

## Paper-style experiment harness

### Paper-lite

```bash
python run_heavy_v12.py   --config configs/paperlite_10seed_v12.json   --out-dir results/paperlite_rerun   --timeout-per-run 1800
```

### Heavy ASP-DAC configuration

```bash
python run_heavy_v12.py   --config configs/heavy_aspdac_v12.json   --out-dir results/heavy_aspdac_rerun   --timeout-per-run 3600
```

The runner is resume-safe and stores one CSV per config/seed plus aggregated raw/summary/status
files. Existing archived evidence and plots from the working snapshot are retained in the repository.

## Main paper-level claims

The extended preprint reports a 115-job evidence suite over Potjans-inspired recurrent SNNs and
4x4--6x6 mesh NoCs, plus a 7x7 stress case. Under its deterministic XY route model, it reports
routed multicast-hop reductions of **10.6--19.6%** versus Activity+QAP and **19.7--41.1%** versus
Edge+QAP, with incremental move evaluation speedups of **4.7--12.7x** while matching full
recomputation to numerical precision.

Those are mapping-level traffic/profiling results, not calibrated hardware latency, throughput,
energy, or power measurements.

## Repository layout

```text
mhysmap.py                  core algorithms and objective implementation
run_single_v12.py           isolated single-run driver
run_heavy_v12.py            resume-safe multi-job experiment harness
run_evidence_suite_v12.py   evidence-suite driver
plot_heavy_v12.py           aggregation/plotting
configs/                    smoke, paper-lite, and heavy JSON configurations
results/                    archived run outputs/evidence
docs/legacy/                original v1.2/v1.3 working notes
```

Legacy v10/v11 aggregate tables and plots are retained at the repository root because they document
the development/evaluation path of the mapper.

## Modeling boundaries

- SNN activity is profiled over a finite interval.
- Routing uses deterministic XY paths in the reported formulation.
- The objective is a mapping-level traffic/congestion proxy.
- The search is heuristic; safe-best portfolio selection protects the best explicitly evaluated
  candidate but does not prove global optimality.
- Hardware multicast, adaptive routing, cycle timing, energy, and physical implementation require
  separate calibrated evaluation.

## arXiv vs GitHub

Git/GitHub is not required for arXiv. Keep the arXiv TeX source package separate from this
research-code repository.

## License

No software license is selected in this snapshot. Add one before advertising the repository as
open-source software.
