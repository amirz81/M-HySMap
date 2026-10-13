"""Run one M-HySMap v1.2 evidence-suite configuration and emit raw-method CSV rows."""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
from mhysmap import (
    MeshObjectiveWeights, build_potjans_mini_directed, profile_activity_event_driven,
    run_mhysmap_core_v13,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--rows', type=int, required=True)
    ap.add_argument('--cols', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--profile-T', type=float, default=0.16)
    ap.add_argument('--input-rate-hz', type=float, default=6.0)
    ap.add_argument('--placement-time', type=float, default=0.35)
    ap.add_argument('--placement-restarts', type=int, default=4)
    ap.add_argument('--joint-cycles', type=int, default=2)
    ap.add_argument('--passes-edge', type=int, default=2)
    ap.add_argument('--passes-hyper', type=int, default=3)
    ap.add_argument('--slack', type=float, default=0.15)
    ap.add_argument('--suite', type=str, default='evidence')
    ap.add_argument('--config-id', type=int, default=0)
    ap.add_argument('--config-name', type=str, default='')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    A, groups = build_potjans_mini_directed(scale=args.scale, seed=args.seed)
    rates, profile_stats = profile_activity_event_driven(
        A, groups, T=args.profile_T, input_rate_hz=args.input_rate_hz,
        seed=args.seed + 303, max_events=150_000,
    )
    weights = MeshObjectiveWeights(alpha_hops=1.0, beta_max_link=0.10, gamma_link_var=0.01)
    results = run_mhysmap_core_v13(
        A, rates, k=args.k, rows=args.rows, cols=args.cols, seed=args.seed,
        passes_edge=args.passes_edge, passes_hyper=args.passes_hyper, slack=args.slack,
        weights=weights, placement_time_limit_s=args.placement_time,
        placement_restarts=args.placement_restarts, exact_threshold_k=8,
        joint_cycles=args.joint_cycles, include_full_joint_reference=False,
        enable_schedule_sweep=True,
    )
    profile_dict = profile_stats.as_dict()
    rows = []
    for method, m in results.items():
        row = {
            'suite': args.suite, 'config_id': args.config_id, 'config_name': args.config_name, 'seed': args.seed,
            'scale': args.scale, 'k': args.k, 'rows': args.rows, 'cols': args.cols,
            'N': int(A.shape[0]), 'E': int(A.nnz), 'method': method,
            'run_wall_s': time.perf_counter() - t0,
        }
        row.update(m)
        row.update(profile_dict)
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = ['suite','config_id','seed','scale','k','rows','cols','N','E','method','edge_cut','weighted_edge_cut','remote_fanout','packet_count','multicast_hops','max_link_load','link_load_variance','objective','run_wall_s']
    fields = preferred + [f for f in fields if f not in preferred]
    with out.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} N={A.shape[0]} E={A.nnz} active={profile_stats.active_neurons} wall={time.perf_counter()-t0:.2f}s")

if __name__ == '__main__':
    main()
