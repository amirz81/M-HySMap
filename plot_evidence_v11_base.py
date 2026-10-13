from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
summary = pd.read_csv(ROOT/'mhysmap_v10_evidence_summary.csv')
imp = pd.read_csv(ROOT/'mhysmap_v10_evidence_improvements.csv')

# cleaner labels/order
method_order = [
    'random_rowmajor',
    'edgecut_refined_pairwise_qap',
    'activity_edge_refined_pairwise_qap',
    'mhysmap_partition_incremental_multicast_certified',
    'mhysmap_v10_portfolio_best',
]
labels = {
    'random_rowmajor':'Random',
    'edgecut_refined_pairwise_qap':'Edge + QAP',
    'activity_edge_refined_pairwise_qap':'Activity-edge + QAP',
    'mhysmap_partition_incremental_multicast_certified':'Sequential M-HySMap',
    'mhysmap_v10_portfolio_best':'M-HySMap v1.0',
}
summary['cfg'] = summary.apply(lambda r: f"s={float(r['scale']):.4f}, k={int(r['k'])}", axis=1)
cfgs = list(dict.fromkeys(summary.sort_values(['scale','k'])['cfg']))

# Hops grouped bar
pivot = summary.pivot_table(index='cfg', columns='method', values='multicast_hops_mean', aggfunc='first').reindex(cfgs)
ax = pivot[[m for m in method_order if m in pivot.columns]].rename(columns=labels).plot(kind='bar', figsize=(11,5))
ax.set_ylabel('Mean routed multicast hops')
ax.set_xlabel('Benchmark configuration')
ax.set_title('M-HySMap v1.0 reduces routed multicast traffic')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(ROOT/'mhysmap_v10_hops_grouped.png', dpi=180)
plt.close()

# Improvement vs baselines
imp['cfg'] = imp.apply(lambda r: f"s={float(r['scale']):.4f}, k={int(r['k'])}", axis=1)
cols = ['hop_improvement_vs_edge_qap_pct','hop_improvement_vs_activity_qap_pct','hop_improvement_vs_sequential_mhysmap_pct']
plot_imp = imp.set_index('cfg')[cols].rename(columns={
    'hop_improvement_vs_edge_qap_pct':'vs Edge + QAP',
    'hop_improvement_vs_activity_qap_pct':'vs Activity-edge + QAP',
    'hop_improvement_vs_sequential_mhysmap_pct':'vs Sequential M-HySMap',
})
ax = plot_imp.plot(kind='bar', figsize=(11,5))
ax.set_ylabel('Mean hop improvement (%)')
ax.set_xlabel('Benchmark configuration')
ax.set_title('Relative improvement of safe-best M-HySMap')
plt.xticks(rotation=25, ha='right')
plt.axhline(0, linewidth=1)
plt.tight_layout()
plt.savefig(ROOT/'mhysmap_v10_improvements.png', dpi=180)
plt.close()

# Speedup
ax = imp.set_index('cfg')['partition_speedup_full_vs_incremental'].plot(kind='bar', figsize=(9,4))
ax.set_ylabel('Partition refinement speedup (×)')
ax.set_xlabel('Benchmark configuration')
ax.set_title('Incremental hyperedge updates accelerate refinement')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(ROOT/'mhysmap_v10_incremental_speedup.png', dpi=180)
plt.close()

# Max link load grouped bar
pivot2 = summary.pivot_table(index='cfg', columns='method', values='max_link_load_mean', aggfunc='first').reindex(cfgs)
ax = pivot2[[m for m in method_order if m in pivot2.columns]].rename(columns=labels).plot(kind='bar', figsize=(11,5))
ax.set_ylabel('Mean max link load')
ax.set_xlabel('Benchmark configuration')
ax.set_title('Congestion: maximum link load')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(ROOT/'mhysmap_v10_maxlink_grouped.png', dpi=180)
plt.close()

print('saved v1.0 plots')
