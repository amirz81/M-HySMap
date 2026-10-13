"""Create paper-friendly plots from a v1.2 heavy-run output directory."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METHOD_LABELS = {
    'edgecut_refined_pairwise_qap': 'Edge+QAP',
    'activity_edge_refined_pairwise_qap': 'Activity+QAP',
    'mhysmap_qap_seeded_refined_multicast': 'QAP-seeded multicast',
    'mhysmap_v10_portfolio_best': 'M-HySMap',
    'mhysmap_partition_incremental_multicast_certified': 'Sequential M-HySMap',
}
METHOD_ORDER = [
    'edgecut_refined_pairwise_qap',
    'activity_edge_refined_pairwise_qap',
    'mhysmap_qap_seeded_refined_multicast',
    'mhysmap_v10_portfolio_best',
]

def cfg_label(row) -> str:
    return f"s={float(row['scale']):.4f}\nk={int(float(row['k']))}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    args = ap.parse_args()
    d = Path(args.results_dir)
    summary_path = d / 'mhysmap_v12_heavy_summary.csv'
    imp_path = d / 'mhysmap_v12_heavy_improvements.csv'
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    imp = pd.read_csv(imp_path) if imp_path.exists() and imp_path.stat().st_size else pd.DataFrame()

    # Grouped hops plot
    keep = summary[summary['method'].isin(METHOD_ORDER)].copy()
    keep['cfg'] = keep.apply(cfg_label, axis=1)
    pivot = keep.pivot_table(index='cfg', columns='method', values='multicast_hops_mean', aggfunc='mean')
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]
    pivot = pivot.rename(columns=METHOD_LABELS)
    ax = pivot.plot(kind='bar', figsize=(10, 4.8))
    ax.set_ylabel('Mean routed multicast hops')
    ax.set_xlabel('Configuration')
    ax.set_title('M-HySMap v1.2 routed multicast hops')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(d / 'mhysmap_v12_hops_grouped.png', dpi=200)
    plt.close()

    # Improvement plot
    if not imp.empty:
        imp['cfg'] = imp.apply(cfg_label, axis=1)
        cols = [c for c in ['hop_improvement_vs_activity_qap_pct','hop_improvement_vs_edge_qap_pct'] if c in imp.columns]
        if cols:
            plotdf = imp.set_index('cfg')[cols].rename(columns={
                'hop_improvement_vs_activity_qap_pct': 'vs Activity+QAP',
                'hop_improvement_vs_edge_qap_pct': 'vs Edge+QAP',
            })
            ax = plotdf.plot(kind='bar', figsize=(10, 4.8))
            ax.set_ylabel('Hop improvement (%)')
            ax.set_xlabel('Configuration')
            ax.set_title('M-HySMap v1.2 hop improvement')
            plt.axhline(0, linewidth=1)
            ax.legend(loc='best', fontsize=8)
            plt.tight_layout()
            plt.savefig(d / 'mhysmap_v12_improvements.png', dpi=200)
            plt.close()
        if 'partition_speedup_full_vs_incremental' in imp.columns:
            sp = imp.set_index('cfg')['partition_speedup_full_vs_incremental']
            ax = sp.plot(kind='bar', figsize=(8, 4.2))
            ax.set_ylabel('Speedup (×)')
            ax.set_xlabel('Configuration')
            ax.set_title('Incremental update speedup')
            plt.tight_layout()
            plt.savefig(d / 'mhysmap_v12_incremental_speedup.png', dpi=200)
            plt.close()

    print('Wrote plots to', d)

if __name__ == '__main__':
    main()
