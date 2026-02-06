import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

"""
Visualization script for comprehensive performance benchmark results.

Usage:
    python visualize_performance.py comprehensive_performance_results.csv

Generates:
    1. GFLOPS comparison across algorithms
    2. Efficiency vs theoretical peak
    3. Scaling with matrix size
    4. GEMM variants comparison
"""

def load_results(filename):
    """Load benchmark results from CSV."""
    df = pd.read_csv(filename)
    return df

def plot_gflops_comparison(df, output_prefix='perf'):
    """Plot GFLOPS across all algorithms."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by algorithm and plot
    algorithms = df['Algorithm'].unique()
    x_pos = range(len(algorithms))

    # Average GFLOPS per algorithm
    avg_gflops = df.groupby('Algorithm')['GFLOPS'].mean()

    colors = plt.cm.viridis(range(len(algorithms)))
    bars = ax.bar(x_pos, avg_gflops.values, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average GFLOPS', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Average GFLOPS Across Algorithms',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(avg_gflops.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_gflops_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_gflops_comparison.png")
    plt.close()

def plot_efficiency_vs_peak(df, output_prefix='perf'):
    """Plot efficiency vs theoretical peak."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate efficiency percentage
    df['EfficiencyPct'] = df['Efficiency'] * 100

    algorithms = df['Algorithm'].unique()
    x_pos = range(len(algorithms))
    avg_efficiency = df.groupby('Algorithm')['EfficiencyPct'].mean()

    colors = plt.cm.RdYlGn(avg_efficiency.values / 100)
    bars = ax.bar(x_pos, avg_efficiency.values, color=colors, edgecolor='black', linewidth=1.5)

    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Theoretical Peak')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=1, label='80% Efficiency', alpha=0.7)
    ax.axhline(y=50, color='yellow', linestyle='--', linewidth=1, label='50% Efficiency', alpha=0.7)

    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency vs Theoretical Hardware Peak',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(avg_efficiency.index, rotation=45, ha='right')
    ax.set_ylim(0, min(110, avg_efficiency.max() * 1.2))
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_efficiency.png")
    plt.close()

def plot_scaling_with_size(df, output_prefix='perf'):
    """Plot how performance scales with matrix size."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter to key algorithms
    key_algorithms = ['Optimized GEMM', 'Blocked GEMM', 'CUDA GEMM',
                     'Householder QR', 'Hessenberg', 'LU', 'SVD']

    for algo in key_algorithms:
        algo_data = df[df['Algorithm'] == algo].sort_values('Size')
        if len(algo_data) > 0:
            ax.plot(algo_data['Size'], algo_data['GFLOPS'],
                   marker='o', linewidth=2, markersize=8, label=algo)

    ax.set_xlabel('Matrix Size (n×n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GFLOPS', fontsize=12, fontweight='bold')
    ax.set_title('Performance Scaling with Matrix Size',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_scaling.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_scaling.png")
    plt.close()

def plot_gemm_variants(df, output_prefix='perf'):
    """Compare GEMM variants specifically."""
    gemm_data = df[df['Algorithm'].str.contains('GEMM', case=False)]

    if len(gemm_data) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: GFLOPS by size
    for algo in gemm_data['Algorithm'].unique():
        algo_data = gemm_data[gemm_data['Algorithm'] == algo].sort_values('Size')
        ax1.plot(algo_data['Size'], algo_data['GFLOPS'],
                marker='o', linewidth=2, markersize=8, label=algo)

    ax1.set_xlabel('Matrix Size (n×n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GFLOPS', fontsize=12, fontweight='bold')
    ax1.set_title('GEMM Variants: Raw Performance', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: Efficiency comparison
    gemm_avg = gemm_data.groupby('Algorithm').agg({
        'GFLOPS': 'mean',
        'Efficiency': 'mean'
    }).reset_index()

    x_pos = range(len(gemm_avg))
    bars = ax2.bar(x_pos, gemm_avg['Efficiency'] * 100,
                   color=plt.cm.viridis(range(len(gemm_avg))),
                   edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('GEMM Variant', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Efficiency (%)', fontsize=12, fontweight='bold')
    ax2.set_title('GEMM Variants: Efficiency vs Peak', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(gemm_avg['Algorithm'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_gemm_variants.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_gemm_variants.png")
    plt.close()

def plot_decomposition_comparison(df, output_prefix='perf'):
    """Compare decomposition algorithms."""
    decomp_algos = ['Householder QR', 'Hessenberg', 'Bidiagonal', 'LU', 'SVD', 'Polar', 'Schur (Implicit QR)']
    decomp_data = df[df['Algorithm'].isin(decomp_algos)]

    if len(decomp_data) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Average GFLOPS per decomposition
    avg_perf = decomp_data.groupby('Algorithm')['GFLOPS'].mean().sort_values(ascending=False)

    x_pos = range(len(avg_perf))
    colors = plt.cm.plasma(range(len(avg_perf)))
    bars = ax.bar(x_pos, avg_perf.values, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Decomposition Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average GFLOPS', fontsize=12, fontweight='bold')
    ax.set_title('Decomposition Algorithms: Performance Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(avg_perf.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_decompositions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_decompositions.png")
    plt.close()

def generate_summary_stats(df, output_file='performance_summary.txt'):
    """Generate text summary of key statistics."""
    with open(output_file, 'w') as f:
        f.write("=== Comprehensive Performance Benchmark Summary ===\n\n")

        # Overall stats
        f.write("Overall Statistics:\n")
        f.write(f"  Total algorithms tested: {df['Algorithm'].nunique()}\n")
        f.write(f"  Matrix sizes tested: {sorted(df['Size'].unique())}\n")
        f.write(f"  Peak GFLOPS achieved: {df['GFLOPS'].max():.2f}\n")
        f.write(f"  Average GFLOPS: {df['GFLOPS'].mean():.2f}\n\n")

        # Best performers
        f.write("Top 5 Performers (by average GFLOPS):\n")
        top5 = df.groupby('Algorithm')['GFLOPS'].mean().sort_values(ascending=False).head(5)
        for i, (algo, gflops) in enumerate(top5.items(), 1):
            f.write(f"  {i}. {algo}: {gflops:.2f} GFLOPS\n")
        f.write("\n")

        # Most efficient
        f.write("Most Efficient (vs theoretical peak):\n")
        top5_eff = df.groupby('Algorithm')['Efficiency'].mean().sort_values(ascending=False).head(5)
        for i, (algo, eff) in enumerate(top5_eff.items(), 1):
            f.write(f"  {i}. {algo}: {eff*100:.1f}%\n")
        f.write("\n")

        # GEMM comparison
        gemm_data = df[df['Algorithm'].str.contains('GEMM', case=False)]
        if len(gemm_data) > 0:
            f.write("GEMM Variants Comparison:\n")
            gemm_stats = gemm_data.groupby('Algorithm').agg({
                'GFLOPS': 'mean',
                'Efficiency': 'mean'
            })
            for algo, row in gemm_stats.iterrows():
                f.write(f"  {algo}:\n")
                f.write(f"    GFLOPS: {row['GFLOPS']:.2f}\n")
                f.write(f"    Efficiency: {row['Efficiency']*100:.1f}%\n")

    print(f"Summary statistics written to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_performance.py <results.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_prefix = csv_file.replace('.csv', '')

    print(f"Loading results from: {csv_file}")
    df = load_results(csv_file)

    print(f"\nGenerating visualizations...")
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'

    plot_gflops_comparison(df, output_prefix)
    plot_efficiency_vs_peak(df, output_prefix)
    plot_scaling_with_size(df, output_prefix)
    plot_gemm_variants(df, output_prefix)
    plot_decomposition_comparison(df, output_prefix)

    generate_summary_stats(df, output_prefix + '_summary.txt')

    print(f"\n✅ All visualizations generated successfully!")
    print(f"Files created:")
    print(f"  - {output_prefix}_gflops_comparison.png")
    print(f"  - {output_prefix}_efficiency.png")
    print(f"  - {output_prefix}_scaling.png")
    print(f"  - {output_prefix}_gemm_variants.png")
    print(f"  - {output_prefix}_decompositions.png")
    print(f"  - {output_prefix}_summary.txt")

if __name__ == '__main__':
    main()
