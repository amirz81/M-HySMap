from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
summary = pd.read_csv(ROOT/'mhysmap_v10_evidence_summary.csv')
imp = pd.read_csv(ROOT/'mhysmap_v10_evidence_improvements.csv')
raw = pd.read_csv(ROOT/'mhysmap_v10_evidence_raw.csv')

# Rename copies to v11 naming
summary.to_csv(ROOT/'mhysmap_v11_evidence_summary.csv', index=False)
imp.to_csv(ROOT/'mhysmap_v11_evidence_improvements.csv', index=False)
raw.to_csv(ROOT/'mhysmap_v11_evidence_raw.csv', index=False)

# Utility
method_labels = {
    'random_rowmajor': 'Random',
    'edgecut_refined_pairwise_qap': 'Edge-cut + QAP',
    'activity_edge_refined_pairwise_qap': 'Activity-edge + QAP',
    'mhysmap_activity_qap_seeded_hyper_refine': 'QAP-seeded Hyper-refine',
    'mhysmap_joint_incremental_default': 'Joint M-HySMap',
    'mhysmap_v10_portfolio_best': 'M-HySMap portfolio',
    'mhysmap_partition_full_recompute_rowmajor': 'Full recompute partition',
    'mhysmap_partition_incremental_rowmajor': 'Incremental partition',
}

def cfg_label(scale,k):
    return f"s={scale:.4f}, k={int(k)}"

# Main paper table: key methods only
key_methods = ['edgecut_refined_pairwise_qap','activity_edge_refined_pairwise_qap','mhysmap_activity_qap_seeded_hyper_refine','mhysmap_v10_portfolio_best']
rows=[]
for (scale,k,rows_grid,cols), g in summary.groupby(['scale','k','rows','cols'], sort=True):
    for m in key_methods:
        r = g[g['method']==m]
        if r.empty: continue
        r = r.iloc[0]
        rows.append({
            'config': cfg_label(scale,k),
            'mesh': f'{int(rows_grid)}x{int(cols)}',
            'method': method_labels[m],
            'edge_cut': r['edge_cut_mean'],
            'weighted_cut': r['weighted_edge_cut_mean'],
            'routed_hops': r['multicast_hops_mean'],
            'max_link': r['max_link_load_mean'],
            'objective': r['objective_mean'],
            'runs': int(r['num_runs']),
        })
main_tbl = pd.DataFrame(rows)
main_tbl.to_csv(ROOT/'mhysmap_v11_main_results_table.csv', index=False)

# Improvement compact table
imp_rows=[]
for _, r in imp.iterrows():
    imp_rows.append({
        'config': cfg_label(r['scale'], r['k']),
        'mesh': f"{int(r['rows'])}x{int(r['cols'])}",
        'hop_impr_vs_edge_qap_pct': r['hop_improvement_vs_edge_qap_pct'],
        'hop_impr_vs_activity_qap_pct': r['hop_improvement_vs_activity_qap_pct'],
        'obj_impr_vs_activity_qap_pct': r['objective_improvement_vs_activity_qap_pct'],
        'maxlink_impr_vs_activity_qap_pct': r['maxlink_improvement_vs_activity_qap_pct'],
        'incremental_speedup_x': r['partition_speedup_full_vs_incremental'],
        'portfolio_joint_selection_rate': r['safe_best_joint_selection_rate'],
        'portfolio_pairwise_seed_rate': r['safe_best_pairwise_seed_rate'],
    })
imp_tbl = pd.DataFrame(imp_rows)
imp_tbl.to_csv(ROOT/'mhysmap_v11_improvement_table.csv', index=False)

# Ablation table: component deltas
abl_rows=[]
for (scale,k,rows_grid,cols), g in summary.groupby(['scale','k','rows','cols'], sort=True):
    def get(m, field):
        r=g[g.method==m]
        return float(r.iloc[0][field]) if not r.empty else np.nan
    edge = get('edgecut_refined_pairwise_qap','multicast_hops_mean')
    act = get('activity_edge_refined_pairwise_qap','multicast_hops_mean')
    seed = get('mhysmap_activity_qap_seeded_hyper_refine','multicast_hops_mean')
    joint = get('mhysmap_joint_incremental_default','multicast_hops_mean')
    final = get('mhysmap_v10_portfolio_best','multicast_hops_mean')
    full_rt = get('mhysmap_partition_full_recompute_rowmajor','runtime_s_mean')
    inc_rt = get('mhysmap_partition_incremental_rowmajor','runtime_s_mean')
    abl_rows.append({
        'config': cfg_label(scale,k),
        'edge_qap_hops': edge,
        'activity_qap_hops': act,
        'qap_seeded_hyper_hops': seed,
        'joint_default_hops': joint,
        'portfolio_hops': final,
        'activity_gain_over_edge_pct': (edge-act)/edge*100.0,
        'hyper_refine_gain_over_activity_pct': (act-seed)/act*100.0,
        'portfolio_gain_over_activity_pct': (act-final)/act*100.0,
        'portfolio_gain_over_edge_pct': (edge-final)/edge*100.0,
        'incremental_speedup_x': full_rt / inc_rt if inc_rt else np.nan,
    })
abl_tbl=pd.DataFrame(abl_rows)
abl_tbl.to_csv(ROOT/'mhysmap_v11_ablation_table.csv', index=False)

# Runtime table
runtime_rows=[]
for (scale,k,rows_grid,cols), g in summary.groupby(['scale','k','rows','cols'], sort=True):
    def get(m, field):
        r=g[g.method==m]
        return float(r.iloc[0][field]) if not r.empty else np.nan
    runtime_rows.append({
        'config': cfg_label(scale,k),
        'full_recompute_partition_s': get('mhysmap_partition_full_recompute_rowmajor','runtime_s_mean'),
        'incremental_partition_s': get('mhysmap_partition_incremental_rowmajor','runtime_s_mean'),
        'speedup_x': get('mhysmap_partition_full_recompute_rowmajor','runtime_s_mean')/get('mhysmap_partition_incremental_rowmajor','runtime_s_mean'),
        'avg_affected_sources': get('mhysmap_partition_incremental_rowmajor','incremental_avg_affected_sources_mean'),
        'max_validation_error': get('mhysmap_partition_incremental_rowmajor','incremental_validation_abs_objective_error_max'),
    })
runtime_tbl=pd.DataFrame(runtime_rows)
runtime_tbl.to_csv(ROOT/'mhysmap_v11_runtime_table.csv', index=False)

# Plots
plot_methods = ['edgecut_refined_pairwise_qap','activity_edge_refined_pairwise_qap','mhysmap_activity_qap_seeded_hyper_refine','mhysmap_v10_portfolio_best']
configs = []
for (scale,k,rows_grid,cols), _ in summary.groupby(['scale','k','rows','cols'], sort=True):
    configs.append((scale,k,rows_grid,cols))
labels = [cfg_label(s,k) for s,k,_,_ in configs]

# Hops ablation grouped
x=np.arange(len(configs)); width=0.18
fig,ax=plt.subplots(figsize=(10,5))
for i,m in enumerate(plot_methods):
    vals=[]
    for cfg in configs:
        s,k,rg,c=cfg
        r=summary[(summary.scale==s)&(summary.k==k)&(summary.rows==rg)&(summary.cols==c)&(summary.method==m)]
        vals.append(float(r.iloc[0]['multicast_hops_mean']))
    ax.bar(x+(i-1.5)*width, vals, width, label=method_labels[m])
ax.set_ylabel('Routed multicast hops')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_title('M-HySMap v1.1 ablation: routed multicast hops')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT/'mhysmap_v11_ablation_hops.png', dpi=180); plt.close(fig)

# Percent improvement over activity-QAP
fig,ax=plt.subplots(figsize=(8,4.5))
vals=imp_tbl['hop_impr_vs_activity_qap_pct'].values
ax.bar(np.arange(len(vals)), vals)
ax.axhline(0, linewidth=1)
ax.set_ylabel('Hop improvement vs Activity-edge + QAP (%)')
ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(imp_tbl['config'], rotation=20, ha='right')
ax.set_title('Final portfolio improvement over strongest graph baseline')
fig.tight_layout(); fig.savefig(ROOT/'mhysmap_v11_improvement_vs_activity_qap.png', dpi=180); plt.close(fig)

# Runtime speedup
fig,ax=plt.subplots(figsize=(8,4.5))
vals=runtime_tbl['speedup_x'].values
ax.bar(np.arange(len(vals)), vals)
ax.set_ylabel('Speedup: full recompute / incremental')
ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(runtime_tbl['config'], rotation=20, ha='right')
ax.set_title('Incremental local hyperedge update speedup')
fig.tight_layout(); fig.savefig(ROOT/'mhysmap_v11_runtime_speedup.png', dpi=180); plt.close(fig)

# Markdown helpers
def md(df, cols=None, precision=2):
    if cols: df=df[cols]
    return df.to_markdown(index=False, floatfmt=f'.{precision}f')

report = []
report.append('# M-HySMap v1.1 Paper Tables and Ablation Pack\n')
report.append('v1.1 packages the v1.0 stable evidence into paper-ready tables, ablations, and figures. It does **not** claim final camera-ready evidence yet; it is the draft-evidence pack for ASP-DAC writing.\n')
report.append('## Main result table\n')
report.append(md(main_tbl, precision=1))
report.append('\n\n## Improvement table\n')
report.append(md(imp_tbl, precision=2))
report.append('\n\n## Ablation table\n')
report.append(md(abl_tbl, precision=2))
report.append('\n\n## Runtime / incremental-update table\n')
report.append(md(runtime_tbl, precision=3))
report.append('\n\n## Draft interpretation\n')
report.append('- The final portfolio beats Edge-cut + QAP on all configs in routed multicast hops.\n')
report.append('- The final portfolio also beats Activity-edge + QAP on all configs in this stable evidence set, including the larger 6x6/k=36 case.\n')
report.append('- The QAP-seeded hypergraph path is the key robustness fix: the method starts from the strongest graph placement and then applies multicast-aware moves.\n')
report.append('- Incremental gain updates give roughly 6-8x partition-refinement speedup while keeping validation error near numerical noise.\n')
report.append('\n## Final run checklist before submission\n')
report.append('1. Expand seeds to at least 10 for k=16 and k=25.\n')
report.append('2. Expand k=36 to at least 5 seeds, ideally 10 if runtime allows.\n')
report.append('3. Add one synthetic fanout/congestion stress benchmark.\n')
report.append('4. Freeze alpha/beta/gamma and report sensitivity in appendix/supplement if space allows.\n')
report.append('5. Add link-load heatmap examples for one representative seed.\n')
(ROOT/'mhysmap_v11_paper_tables_report.md').write_text('\n'.join(report))

readme = """# M-HySMap v1.1\n\nPaper-table and ablation pack for the ASP-DAC draft.\n\nThis directory contains the v1.0 algorithm code plus v1.1 paper artifacts:\n\n- `mhysmap_v11_main_results_table.csv`\n- `mhysmap_v11_improvement_table.csv`\n- `mhysmap_v11_ablation_table.csv`\n- `mhysmap_v11_runtime_table.csv`\n- `mhysmap_v11_paper_tables_report.md`\n- `mhysmap_v11_ablation_hops.png`\n- `mhysmap_v11_improvement_vs_activity_qap.png`\n- `mhysmap_v11_runtime_speedup.png`\n\nThe raw/summary evidence CSVs are copied from the stable v1.0 evidence run and renamed/copied for v1.1 packaging.\n\nTo reproduce the paper-table generation:\n\n```bash\npython build_paper_tables_v11.py\n```\n\nTo run new evidence, use `run_evidence_suite_v11.py`, which is the v1.0 evidence runner copied forward.\n"""
(ROOT/'README.md').write_text(readme)
print('Wrote v1.1 tables, plots, and report into', ROOT)
