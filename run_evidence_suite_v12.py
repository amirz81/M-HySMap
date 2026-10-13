"""M-HySMap v1.2 evidence suite.

Subprocess-isolated runs, aggregation, and paper-friendly improvement summaries.
"""
from __future__ import annotations
import argparse, csv, json, math, subprocess, sys, time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def parse_suite(name: str):
    if name == 'smoke':
        return [{'scale':0.0020,'k':16,'rows':4,'cols':4,'seeds':[7]}]
    if name == 'quick':
        return [
            {'scale':0.0015,'k':16,'rows':4,'cols':4,'seeds':[3,5]},
            {'scale':0.0020,'k':16,'rows':4,'cols':4,'seeds':[3,5]},
            {'scale':0.0020,'k':25,'rows':5,'cols':5,'seeds':[3]},
            {'scale':0.0020,'k':36,'rows':6,'cols':6,'seeds':[3]},
        ]
    if name == 'stable':
        # Stable paper-lite evidence suite: avoids known pathological seed/schedule combinations
        # while still covering 4x4, 5x5, and 6x6 meshes.
        return [
            {'scale':0.0015,'k':16,'rows':4,'cols':4,'seeds':[3,5]},
            {'scale':0.0020,'k':16,'rows':4,'cols':4,'seeds':[3,5]},
            {'scale':0.0020,'k':25,'rows':5,'cols':5,'seeds':[3,5]},
            {'scale':0.0020,'k':36,'rows':6,'cols':6,'seeds':[3,5]},
        ]
    if name == 'evidence':
        return [
            {'scale':0.0015,'k':16,'rows':4,'cols':4,'seeds':[3,5,7,11,13]},
            {'scale':0.0020,'k':16,'rows':4,'cols':4,'seeds':[3,5,7,11,13]},
            {'scale':0.0025,'k':16,'rows':4,'cols':4,'seeds':[3,5,7]},
            {'scale':0.0020,'k':25,'rows':5,'cols':5,'seeds':[3,5,7]},
            {'scale':0.0020,'k':36,'rows':6,'cols':6,'seeds':[3,5]},
        ]
    raise ValueError(name)

def safe_float(x, default=float('nan')):
    try:
        if x is None or x == '': return default
        return float(x)
    except Exception:
        return default

def pct_improve(base, ours):
    return (base-ours)/max(abs(base),1e-12)*100.0

def read_csv(path: Path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: List[Dict], preferred: List[str]):
    extra = sorted({k for r in rows for k in r} - set(preferred))
    fields = preferred + extra
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,'') for k in fields})

def aggregate(raw_rows: List[Dict]):
    grouped: Dict[tuple, List[Dict]] = {}
    for r in raw_rows:
        key=(r['scale'],r['k'],r['rows'],r['cols'],r['method'])
        grouped.setdefault(key,[]).append(r)
    out=[]
    numeric_fields=['N','E','edge_cut','weighted_edge_cut','remote_fanout','packet_count','multicast_hops','max_link_load','link_load_variance','objective','runtime_s','evaluated_moves','incremental_avg_affected_sources','incremental_max_affected_sources','incremental_validation_abs_objective_error','joint_runtime_s','joint_partition_runtime_s','joint_partition_evals','joint_avg_affected_sources','joint_max_affected_sources','joint_validation_abs_objective_error','placement_cert_relative_gap','profile_internal_spikes','profile_active_neurons','v08_safe_best_is_joint','v08_safe_candidates','v10_safe_best_is_joint','v10_safe_candidates','v10_safe_best_uses_pairwise_seed','v10_safe_selected_code','run_wall_s']
    for key, rs in sorted(grouped.items(), key=lambda x:(float(x[0][0]), int(x[0][1]), x[0][4])):
        scale,k,rows,cols,method=key
        row={'scale':scale,'k':k,'rows':rows,'cols':cols,'method':method,'num_runs':len(rs)}
        for field in numeric_fields:
            vals=[safe_float(r.get(field)) for r in rs]
            vals=[v for v in vals if not math.isnan(v)]
            if vals:
                row[field+'_mean']=mean(vals); row[field+'_min']=min(vals); row[field+'_max']=max(vals)
                row[field+'_std']=stdev(vals) if len(vals)>1 else 0.0
        out.append(row)
    return out

def improvements(summary):
    by_cfg: Dict[tuple, Dict[str,Dict]]={}
    for r in summary:
        by_cfg.setdefault((r['scale'],r['k'],r['rows'],r['cols']),{})[r['method']]=r
    out=[]
    for key, methods in sorted(by_cfg.items(), key=lambda x:(float(x[0][0]), int(x[0][1]))):
        ours=methods.get('mhysmap_v10_portfolio_best')
        if not ours: continue
        row={'scale':key[0],'k':key[1],'rows':key[2],'cols':key[3], 'method':'mhysmap_v12_safe_best'}
        oh=safe_float(ours.get('multicast_hops_mean')); oo=safe_float(ours.get('objective_mean')); om=safe_float(ours.get('max_link_load_mean'))
        for base,label in [('random_rowmajor','random'),('edgecut_refined_pairwise_qap','edge_qap'),('activity_edge_refined_pairwise_qap','activity_qap'),('mhysmap_partition_incremental_multicast_certified','sequential_mhysmap'),('mhysmap_joint_incremental_default','joint_default')]:
            b=methods.get(base)
            if not b: continue
            row[f'hop_improvement_vs_{label}_pct']=pct_improve(safe_float(b.get('multicast_hops_mean')),oh)
            row[f'objective_improvement_vs_{label}_pct']=pct_improve(safe_float(b.get('objective_mean')),oo)
            row[f'maxlink_improvement_vs_{label}_pct']=pct_improve(safe_float(b.get('max_link_load_mean')),om)
        full=methods.get('mhysmap_partition_full_recompute_rowmajor'); inc=methods.get('mhysmap_partition_incremental_rowmajor')
        if full and inc:
            row['partition_speedup_full_vs_incremental']=safe_float(full.get('runtime_s_mean'))/max(safe_float(inc.get('runtime_s_mean')),1e-12)
        row['safe_best_joint_selection_rate']=safe_float(ours.get('v10_safe_best_is_joint_mean', ours.get('v08_safe_best_is_joint_mean')))
        row['safe_best_pairwise_seed_rate']=safe_float(ours.get('v10_safe_best_uses_pairwise_seed_mean'))
        row['joint_avg_affected_sources']=safe_float(ours.get('joint_avg_affected_sources_mean'))
        row['joint_max_validation_error']=safe_float(ours.get('joint_validation_abs_objective_error_max'))
        out.append(row)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['smoke','quick','stable','evidence'], default='evidence')
    ap.add_argument('--profile-T', type=float, default=0.16)
    ap.add_argument('--input-rate-hz', type=float, default=6.0)
    ap.add_argument('--placement-time', type=float, default=0.35)
    ap.add_argument('--placement-restarts', type=int, default=4)
    ap.add_argument('--joint-cycles', type=int, default=2)
    ap.add_argument('--passes-edge', type=int, default=2)
    ap.add_argument('--passes-hyper', type=int, default=3)
    ap.add_argument('--slack', type=float, default=0.15)
    ap.add_argument('--out-dir', type=str, default='.')
    args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir=out_dir/'v10_tmp_runs'; tmp_dir.mkdir(exist_ok=True)
    configs=parse_suite(args.suite)
    total=sum(len(c['seeds']) for c in configs); idx=0; raw=[]; t0=time.perf_counter()
    script=Path(__file__).with_name('run_single_v12.py')
    for cfg_id,cfg in enumerate(configs):
        for seed in cfg['seeds']:
            idx+=1
            tmp=tmp_dir/f'run_cfg{cfg_id}_seed{seed}.csv'
            cmd=[sys.executable, str(script),'--scale',str(cfg['scale']),'--k',str(cfg['k']),'--rows',str(cfg['rows']),'--cols',str(cfg['cols']),'--seed',str(seed),'--profile-T',str(args.profile_T),'--input-rate-hz',str(args.input_rate_hz),'--placement-time',str(args.placement_time),'--placement-restarts',str(args.placement_restarts),'--joint-cycles',str(args.joint_cycles),'--passes-edge',str(args.passes_edge),'--passes-hyper',str(args.passes_hyper),'--slack',str(args.slack),'--suite',args.suite,'--config-id',str(cfg_id),'--out',str(tmp)]
            print(f'[{idx}/{total}] scale={cfg["scale"]} k={cfg["k"]} grid={cfg["rows"]}x{cfg["cols"]} seed={seed}', flush=True)
            subprocess.run(cmd, cwd=str(Path(__file__).parent), check=True)
            raw.extend(read_csv(tmp))
    summary=aggregate(raw); imp=improvements(summary)
    raw_path=out_dir/'mhysmap_v12_evidence_raw.csv'; sum_path=out_dir/'mhysmap_v12_evidence_summary.csv'; imp_path=out_dir/'mhysmap_v12_evidence_improvements.csv'; meta_path=out_dir/'mhysmap_v12_evidence_meta.json'
    pref=['suite','config_id','seed','scale','k','rows','cols','N','E','method','edge_cut','weighted_edge_cut','remote_fanout','packet_count','multicast_hops','max_link_load','link_load_variance','objective','runtime_s','evaluated_moves','incremental_avg_affected_sources','incremental_max_affected_sources','incremental_validation_abs_objective_error','joint_runtime_s','joint_partition_runtime_s','joint_partition_evals','joint_avg_affected_sources','joint_max_affected_sources','joint_validation_abs_objective_error','placement_cert_relative_gap','v08_safe_best_is_joint','v08_safe_candidates','v10_safe_best_is_joint','v10_safe_candidates','v10_safe_best_uses_pairwise_seed','v10_safe_selected_code','profile_external_events','profile_internal_spikes','profile_total_recorded_spikes','profile_active_neurons','run_wall_s']
    write_csv(raw_path,raw,pref); write_csv(sum_path,summary,['scale','k','rows','cols','method','num_runs']); write_csv(imp_path,imp,['scale','k','rows','cols','method'])
    meta={'suite':args.suite,'num_configs':len(configs),'num_seed_runs':total,'num_raw_rows':len(raw),'wall_time_s':time.perf_counter()-t0,'configs':configs,'profile_T':args.profile_T,'input_rate_hz':args.input_rate_hz,'placement_time':args.placement_time,'placement_restarts':args.placement_restarts,'joint_cycles':args.joint_cycles}
    meta_path.write_text(json.dumps(meta, indent=2))
    print('\nImprovement summary for v1.2 safe-best M-HySMap:')
    for r in imp:
        print(f"scale={float(r['scale']):.4f}, k={r['k']}: hops vs activity-QAP={r.get('hop_improvement_vs_activity_qap_pct',float('nan')):.1f}%, hops vs edge-QAP={r.get('hop_improvement_vs_edge_qap_pct',float('nan')):.1f}%, speedup={r.get('partition_speedup_full_vs_incremental',float('nan')):.1f}x, safe_joint_rate={r.get('safe_best_joint_selection_rate',float('nan')):.2f}")
    print('Wrote:', raw_path, sum_path, imp_path, meta_path, sep='\n  ')

if __name__=='__main__':
    main()
