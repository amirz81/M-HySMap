"""M-HySMap v1.2 heavy-run harness.

Features:
- JSON-driven experiment configs
- resume-safe per-run CSV outputs
- subprocess isolation
- per-run timeouts
- stdout/stderr logs per seed/config
- partial raw/summary/improvement CSVs updated after every completed run

Usage:
    python run_heavy_v12.py --config configs/paper_10seed.json --out-dir results/paper_10seed
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from run_evidence_suite_v12 import aggregate, improvements, read_csv, write_csv


DEFAULTS: Dict[str, Any] = {
    "profile_T": 0.16,
    "input_rate_hz": 6.0,
    "placement_time": 0.35,
    "placement_restarts": 4,
    "joint_cycles": 2,
    "passes_edge": 2,
    "passes_hyper": 3,
    "slack": 0.15,
}

RAW_FIELDS = [
    'suite','config_id','config_name','seed','scale','k','rows','cols','N','E','method',
    'edge_cut','weighted_edge_cut','remote_fanout','packet_count','multicast_hops',
    'max_link_load','link_load_variance','objective','runtime_s','evaluated_moves',
    'incremental_avg_affected_sources','incremental_max_affected_sources',
    'incremental_validation_abs_objective_error','joint_runtime_s','joint_partition_runtime_s',
    'joint_partition_evals','joint_avg_affected_sources','joint_max_affected_sources',
    'joint_validation_abs_objective_error','placement_cert_relative_gap',
    'v08_safe_best_is_joint','v08_safe_candidates','v10_safe_best_is_joint','v10_safe_candidates',
    'v10_safe_best_uses_pairwise_seed','v10_safe_selected_code','profile_external_events',
    'profile_internal_spikes','profile_total_recorded_spikes','profile_active_neurons','run_wall_s'
]


def load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if 'configs' not in data or not isinstance(data['configs'], list):
        raise ValueError("JSON config must contain a list field named 'configs'.")
    for i, cfg in enumerate(data['configs']):
        for key in ['scale', 'k', 'rows', 'cols', 'seeds']:
            if key not in cfg:
                raise ValueError(f"configs[{i}] missing required key: {key}")
        if not isinstance(cfg['seeds'], list) or not cfg['seeds']:
            raise ValueError(f"configs[{i}].seeds must be a non-empty list")
        cfg.setdefault('name', f"scale{cfg['scale']}_k{cfg['k']}")
    return data


def iter_jobs(data: Dict[str, Any]):
    for cfg_id, cfg in enumerate(data['configs']):
        for seed in cfg['seeds']:
            yield cfg_id, cfg, int(seed)


def stable_job_name(cfg_id: int, cfg: Dict[str, Any], seed: int) -> str:
    name = str(cfg.get('name', f'cfg{cfg_id}')).replace('/', '_').replace(' ', '_')
    return f"cfg{cfg_id:02d}_{name}_seed{seed}"


def csv_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline='') as f:
            return len(list(csv.DictReader(f))) > 0
    except Exception:
        return False


def collect_completed(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted((run_dir / 'per_run').glob('*.csv')):
        if csv_has_rows(p):
            rows.extend(read_csv(p))
    return rows


def write_status(status_path: Path, status_rows: List[Dict[str, Any]]) -> None:
    fields = ['job','config_id','config_name','seed','status','returncode','elapsed_s','csv_path','log_stdout','log_stderr','message']
    write_csv(status_path, status_rows, fields)


def refresh_aggregates(out_dir: Path) -> None:
    raw = collect_completed(out_dir)
    raw_path = out_dir / 'mhysmap_v12_heavy_raw.csv'
    summary_path = out_dir / 'mhysmap_v12_heavy_summary.csv'
    improvement_path = out_dir / 'mhysmap_v12_heavy_improvements.csv'
    if raw:
        # Add config_name to preferred raw fields if present.
        preferred = RAW_FIELDS
        write_csv(raw_path, raw, preferred)
        summary = aggregate(raw)
        imp = improvements(summary)
        write_csv(summary_path, summary, ['scale','k','rows','cols','method','num_runs'])
        write_csv(improvement_path, imp, ['scale','k','rows','cols','method'])
    else:
        for p in [raw_path, summary_path, improvement_path]:
            p.write_text('')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to JSON experiment config')
    ap.add_argument('--out-dir', required=True, help='Directory where results/logs are written')
    ap.add_argument('--timeout-per-run', type=float, default=None, help='Seconds per config/seed subprocess')
    ap.add_argument('--force', action='store_true', help='Rerun even if per-run CSV already exists')
    ap.add_argument('--dry-run', action='store_true', help='Only print jobs and commands')
    ap.add_argument('--stop-on-fail', action='store_true', help='Stop at first failed subprocess')
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    data = load_config(config_path)
    out_dir = Path(args.out_dir).resolve()
    per_run_dir = out_dir / 'per_run'
    log_dir = out_dir / 'logs'
    per_run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Merge defaults: top-level settings override DEFAULTS; per-config settings override top-level.
    top = {**DEFAULTS, **{k: data[k] for k in DEFAULTS if k in data}}
    suite_name = data.get('suite_name', config_path.stem)
    timeout = args.timeout_per_run if args.timeout_per_run is not None else data.get('timeout_per_run_s')

    jobs = list(iter_jobs(data))
    status_rows: List[Dict[str, Any]] = []
    t_suite = time.perf_counter()
    script = Path(__file__).with_name('run_single_v12.py')

    print(f"M-HySMap v1.2 heavy harness")
    print(f"config: {config_path}")
    print(f"out-dir: {out_dir}")
    print(f"jobs: {len(jobs)}")
    print(f"timeout-per-run: {timeout}")
    print()

    for idx, (cfg_id, cfg, seed) in enumerate(jobs, start=1):
        params = {**top, **{k: cfg[k] for k in DEFAULTS if k in cfg}}
        job = stable_job_name(cfg_id, cfg, seed)
        csv_path = per_run_dir / f'{job}.csv'
        stdout_path = log_dir / f'{job}.stdout.log'
        stderr_path = log_dir / f'{job}.stderr.log'
        if csv_has_rows(csv_path) and not args.force:
            print(f"[{idx}/{len(jobs)}] SKIP {job} (already done)", flush=True)
            status_rows.append({
                'job': job, 'config_id': cfg_id, 'config_name': cfg.get('name',''), 'seed': seed,
                'status': 'skipped_existing', 'returncode': 0, 'elapsed_s': 0.0,
                'csv_path': str(csv_path), 'log_stdout': str(stdout_path), 'log_stderr': str(stderr_path), 'message': ''
            })
            continue

        cmd = [
            sys.executable, str(script),
            '--scale', str(cfg['scale']), '--k', str(cfg['k']),
            '--rows', str(cfg['rows']), '--cols', str(cfg['cols']),
            '--seed', str(seed), '--profile-T', str(params['profile_T']),
            '--input-rate-hz', str(params['input_rate_hz']),
            '--placement-time', str(params['placement_time']),
            '--placement-restarts', str(params['placement_restarts']),
            '--joint-cycles', str(params['joint_cycles']),
            '--passes-edge', str(params['passes_edge']),
            '--passes-hyper', str(params['passes_hyper']),
            '--slack', str(params['slack']),
            '--suite', str(suite_name), '--config-id', str(cfg_id),
            '--config-name', str(cfg.get('name','')),
            '--out', str(csv_path),
        ]
        print(f"[{idx}/{len(jobs)}] RUN {job}: scale={cfg['scale']} k={cfg['k']} grid={cfg['rows']}x{cfg['cols']} seed={seed}", flush=True)
        if args.dry_run:
            print('  ' + ' '.join(cmd))
            continue
        t0 = time.perf_counter()
        try:
            with stdout_path.open('w') as so, stderr_path.open('w') as se:
                proc = subprocess.run(
                    cmd, cwd=str(Path(__file__).parent), stdout=so, stderr=se,
                    timeout=timeout, check=False,
                )
            elapsed = time.perf_counter() - t0
            status = 'ok' if proc.returncode == 0 and csv_has_rows(csv_path) else 'failed'
            msg = '' if status == 'ok' else 'nonzero return code or missing CSV rows'
            print(f"  -> {status} returncode={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
            status_rows.append({
                'job': job, 'config_id': cfg_id, 'config_name': cfg.get('name',''), 'seed': seed,
                'status': status, 'returncode': proc.returncode, 'elapsed_s': f'{elapsed:.3f}',
                'csv_path': str(csv_path), 'log_stdout': str(stdout_path), 'log_stderr': str(stderr_path), 'message': msg,
            })
            refresh_aggregates(out_dir)
            if status != 'ok' and args.stop_on_fail:
                break
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            print(f"  -> TIMEOUT elapsed={elapsed:.1f}s", flush=True)
            status_rows.append({
                'job': job, 'config_id': cfg_id, 'config_name': cfg.get('name',''), 'seed': seed,
                'status': 'timeout', 'returncode': '', 'elapsed_s': f'{elapsed:.3f}',
                'csv_path': str(csv_path), 'log_stdout': str(stdout_path), 'log_stderr': str(stderr_path),
                'message': f'timeout after {timeout} seconds',
            })
            refresh_aggregates(out_dir)
            if args.stop_on_fail:
                break

        write_status(out_dir / 'mhysmap_v12_heavy_status.csv', status_rows)

    refresh_aggregates(out_dir)
    meta = {
        'suite_name': suite_name,
        'config_path': str(config_path),
        'out_dir': str(out_dir),
        'total_jobs': len(jobs),
        'completed_rows': len(collect_completed(out_dir)),
        'wall_time_s': time.perf_counter() - t_suite,
        'defaults': top,
        'timeout_per_run_s': timeout,
        'configs': data['configs'],
    }
    (out_dir / 'mhysmap_v12_heavy_meta.json').write_text(json.dumps(meta, indent=2))
    write_status(out_dir / 'mhysmap_v12_heavy_status.csv', status_rows)
    print("\nDone. Key outputs:")
    for name in ['mhysmap_v12_heavy_raw.csv','mhysmap_v12_heavy_summary.csv','mhysmap_v12_heavy_improvements.csv','mhysmap_v12_heavy_status.csv','mhysmap_v12_heavy_meta.json']:
        print('  ', out_dir / name)


if __name__ == '__main__':
    main()
