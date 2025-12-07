#!/usr/bin/env python3
"""
PhiBench Analysis Script

Analyzes benchmark results and generates statistics, figures, and visualization data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats


def load_results(results_path: str = "results/phi_results/all_results.csv") -> pd.DataFrame:
    """Load benchmark results from CSV."""
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df):,} networks")
    return df


def summary_statistics(df: pd.DataFrame) -> dict:
    """Compute overall summary statistics."""
    total = len(df)
    nonzero = (df['phi'] > 0).sum()

    summary = {
        'total_networks': total,
        'nonzero_phi_count': int(nonzero),
        'nonzero_phi_rate': round(nonzero / total * 100, 2),
        'mean_phi': round(df['phi'].mean(), 6),
        'max_phi': round(df['phi'].max(), 6),
        'mean_computation_time': round(df['computation_time'].mean(), 2),
        'total_computation_hours': round(df['computation_time'].sum() / 3600, 2)
    }
    return summary


def analyze_by_tpm_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results grouped by TPM rule."""
    results = []

    for rule in df['tpm_rule'].unique():
        subset = df[df['tpm_rule'] == rule]
        nonzero = subset[subset['phi'] > 0]

        results.append({
            'tpm_rule': rule,
            'count': len(subset),
            'nonzero_count': len(nonzero),
            'nonzero_rate': round(len(nonzero) / len(subset) * 100, 2),
            'mean_phi': round(subset['phi'].mean(), 6),
            'mean_phi_nonzero': round(nonzero['phi'].mean(), 6) if len(nonzero) > 0 else 0,
            'max_phi': round(subset['phi'].max(), 6),
            'mean_time': round(subset['computation_time'].mean(), 2)
        })

    return pd.DataFrame(results).sort_values('mean_phi', ascending=False)


def analyze_by_topology(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results grouped by topology."""
    results = []

    for topo in df['topology'].unique():
        subset = df[df['topology'] == topo]
        nonzero = subset[subset['phi'] > 0]

        results.append({
            'topology': topo,
            'count': len(subset),
            'nonzero_count': len(nonzero),
            'nonzero_rate': round(len(nonzero) / len(subset) * 100, 2),
            'mean_phi': round(subset['phi'].mean(), 6),
            'mean_phi_nonzero': round(nonzero['phi'].mean(), 6) if len(nonzero) > 0 else 0,
            'max_phi': round(subset['phi'].max(), 6),
            'mean_time': round(subset['computation_time'].mean(), 2)
        })

    return pd.DataFrame(results).sort_values('nonzero_rate', ascending=False)


def analyze_by_node_size(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results grouped by node size."""
    results = []

    for n in sorted(df['n_nodes'].unique()):
        subset = df[df['n_nodes'] == n]
        nonzero = subset[subset['phi'] > 0]

        results.append({
            'n_nodes': n,
            'count': len(subset),
            'nonzero_count': len(nonzero),
            'nonzero_rate': round(len(nonzero) / len(subset) * 100, 2),
            'mean_phi': round(subset['phi'].mean(), 6),
            'mean_phi_nonzero': round(nonzero['phi'].mean(), 6) if len(nonzero) > 0 else 0,
            'max_phi': round(subset['phi'].max(), 6),
            'mean_time': round(subset['computation_time'].mean(), 2),
            'max_time': round(subset['computation_time'].max(), 2)
        })

    return pd.DataFrame(results)


def create_heatmap_data(df: pd.DataFrame) -> dict:
    """Create heatmap data for TPM rule x Topology."""
    heatmap = {}

    for rule in ['and', 'or', 'majority', 'xor']:
        heatmap[rule] = {}
        for topo in df['topology'].unique():
            subset = df[(df['tpm_rule'] == rule) & (df['topology'] == topo)]
            if len(subset) > 0:
                rate = round((subset['phi'] > 0).sum() / len(subset) * 100, 1)
                heatmap[rule][topo] = rate
            else:
                heatmap[rule][topo] = None

    return heatmap


def statistical_tests(df: pd.DataFrame) -> dict:
    """Perform statistical significance tests."""
    results = {}

    # Compare AND vs OR (non-zero phi values only)
    and_phi = df[(df['tpm_rule'] == 'and') & (df['phi'] > 0)]['phi']
    or_phi = df[(df['tpm_rule'] == 'or') & (df['phi'] > 0)]['phi']

    if len(and_phi) > 0 and len(or_phi) > 0:
        stat, pval = stats.mannwhitneyu(and_phi, or_phi, alternative='greater')
        results['and_vs_or'] = {
            'test': 'Mann-Whitney U',
            'statistic': round(stat, 2),
            'p_value': f"{pval:.2e}",
            'significant': pval < 0.05,
            'interpretation': 'AND produces significantly higher phi than OR' if pval < 0.05 else 'No significant difference'
        }

    # Compare watts_strogatz vs ring_lattice
    ws_phi = df[(df['topology'] == 'watts_strogatz') & (df['phi'] > 0)]['phi']
    rl_phi = df[(df['topology'] == 'ring_lattice') & (df['phi'] > 0)]['phi']

    if len(ws_phi) > 0 and len(rl_phi) > 0:
        stat, pval = stats.mannwhitneyu(ws_phi, rl_phi, alternative='greater')
        results['watts_strogatz_vs_ring_lattice'] = {
            'test': 'Mann-Whitney U',
            'statistic': round(stat, 2),
            'p_value': f"{pval:.2e}",
            'significant': pval < 0.05,
            'interpretation': 'Watts-Strogatz produces significantly higher phi' if pval < 0.05 else 'No significant difference'
        }

    return results


def export_visualization_data(df: pd.DataFrame, output_path: str = "results/visualization_data.json"):
    """Export data for visualization in JSON format."""

    # TPM rule chart data
    tpm_stats = analyze_by_tpm_rule(df)
    tpm_chart = {
        'labels': tpm_stats['tpm_rule'].str.upper().tolist(),
        'mean_phi': tpm_stats['mean_phi'].tolist(),
        'nonzero_rate': tpm_stats['nonzero_rate'].tolist()
    }

    # Topology chart data
    topo_stats = analyze_by_topology(df)
    topo_chart = {
        'labels': topo_stats['topology'].tolist(),
        'nonzero_rate': topo_stats['nonzero_rate'].tolist(),
        'mean_phi': topo_stats['mean_phi'].tolist()
    }

    # Node size chart data
    node_stats = analyze_by_node_size(df)
    node_chart = {
        'labels': [f"n={n}" for n in node_stats['n_nodes'].tolist()],
        'mean_time': node_stats['mean_time'].tolist(),
        'max_time': node_stats['max_time'].tolist()
    }

    # Node size by TPM rule (for line chart)
    node_by_tpm = {}
    for rule in ['and', 'or', 'majority']:
        rule_data = []
        for n in sorted(df['n_nodes'].unique()):
            subset = df[(df['tpm_rule'] == rule) & (df['n_nodes'] == n) & (df['phi'] > 0)]
            if len(subset) > 0:
                rule_data.append(round(subset['phi'].mean(), 4))
            else:
                rule_data.append(0)
        node_by_tpm[rule] = rule_data
    node_chart['by_tpm'] = node_by_tpm

    # Heatmap data
    heatmap = create_heatmap_data(df)

    # Summary stats
    summary = summary_statistics(df)

    viz_data = {
        'summary': summary,
        'tpm_chart': tpm_chart,
        'topology_chart': topo_chart,
        'node_size_chart': node_chart,
        'heatmap': heatmap
    }

    with open(output_path, 'w') as f:
        json.dump(viz_data, f, indent=2)

    print(f"Visualization data exported to {output_path}")
    return viz_data


def print_report(df: pd.DataFrame):
    """Print a formatted analysis report."""
    print("\n" + "="*70)
    print("PHIBENCH ANALYSIS REPORT")
    print("="*70)

    # Summary
    summary = summary_statistics(df)
    print(f"\nSUMMARY")
    print(f"  Total networks: {summary['total_networks']:,}")
    print(f"  Non-zero phi: {summary['nonzero_phi_count']:,} ({summary['nonzero_phi_rate']}%)")
    print(f"  Mean phi: {summary['mean_phi']}")
    print(f"  Max phi: {summary['max_phi']}")
    print(f"  Total compute time: {summary['total_computation_hours']} hours")

    # By TPM rule
    print(f"\nBY TPM RULE")
    tpm_stats = analyze_by_tpm_rule(df)
    print(tpm_stats.to_string(index=False))

    # By topology
    print(f"\nBY TOPOLOGY")
    topo_stats = analyze_by_topology(df)
    print(topo_stats.to_string(index=False))

    # By node size
    print(f"\nBY NODE SIZE")
    node_stats = analyze_by_node_size(df)
    print(node_stats.to_string(index=False))

    # Statistical tests
    print(f"\nSTATISTICAL TESTS")
    stat_tests = statistical_tests(df)
    for name, result in stat_tests.items():
        print(f"  {name}:")
        print(f"    {result['test']}: stat={result['statistic']}, p={result['p_value']}")
        print(f"    {result['interpretation']}")

    # Key findings
    print(f"\nKEY FINDINGS")

    # Find best TPM rule
    best_tpm = tpm_stats.iloc[0]
    print(f"  - Best TPM rule: {best_tpm['tpm_rule']} (mean phi={best_tpm['mean_phi']}, {best_tpm['nonzero_rate']}% non-zero)")

    # Find best topology
    best_topo = topo_stats.iloc[0]
    print(f"  - Best topology: {best_topo['topology']} ({best_topo['nonzero_rate']}% non-zero)")

    # XOR observation
    xor_stats = df[df['tpm_rule'] == 'xor']
    xor_nonzero = (xor_stats['phi'] > 0).sum()
    print(f"  - XOR produces {xor_nonzero}/{len(xor_stats)} non-zero phi (mathematical property)")

    # Zero-integration topologies
    zero_topos = topo_stats[topo_stats['nonzero_rate'] == 0]['topology'].tolist()
    if zero_topos:
        print(f"  - Zero-integration topologies: {', '.join(zero_topos)}")

    print("\n" + "="*70)


def main():
    """Main analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze PhiBench results')
    parser.add_argument('--results', default='results/phi_results/all_results.csv',
                        help='Path to results CSV')
    parser.add_argument('--export-json', action='store_true',
                        help='Export visualization data to JSON')
    parser.add_argument('--json-output', default='results/visualization_data.json',
                        help='JSON output path')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed report')

    args = parser.parse_args()

    # Load data
    df = load_results(args.results)

    # Print report
    if not args.quiet:
        print_report(df)

    # Export JSON if requested
    if args.export_json:
        export_visualization_data(df, args.json_output)

    return df


if __name__ == '__main__':
    main()
