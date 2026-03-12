import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(repo_root):
    consolidated_paths = [
        os.path.join(repo_root, 'build', 'reports', 'gemm_pes_consolidated.csv'),
        os.path.join(repo_root, 'consolidated_gemm_pes.csv')
    ]
    # prefer build/reports copy
    consolidated = None
    for p in consolidated_paths:
        if os.path.exists(p):
            consolidated = pd.read_csv(p)
            break
    if consolidated is None:
        raise FileNotFoundError('Consolidated CSV not found')

    roofline_path = os.path.join(repo_root, 'build', 'reports', 'roofline_20260311175904', 'portable_efficiency_results.csv')
    if not os.path.exists(roofline_path):
        raise FileNotFoundError(roofline_path)
    roofline = pd.read_csv(roofline_path)

    gemm_analysis_path = os.path.join(repo_root, 'build', 'reports', 'gemm-analysis-quick-rerun', 'gemm_analysis_runs.csv')
    gemm_analysis = None
    if os.path.exists(gemm_analysis_path):
        gemm_analysis = pd.read_csv(gemm_analysis_path)

    return consolidated, roofline, gemm_analysis


def infer_ai(row, roofline_df, gemm_df):
    # Try match in roofline
    mask = (
        (roofline_df['kernel'] == row['kernel']) &
        (roofline_df['m'] == row['m']) &
        (roofline_df['n'] == row['n']) &
        (roofline_df['k'] == row['k'])
    )
    if mask.any():
        return float(roofline_df.loc[mask, 'arithmetic_intensity'].iloc[0])

    # Try gemm analysis: match by source_identifier case_id substring
    if gemm_df is not None and isinstance(row.get('source_identifier'), str):
        sid = row['source_identifier']
        # extract case_id=... if present
        if 'case_id=' in sid:
            case_id = sid.split('case_id=')[-1].split(';')[0]
            gmask = gemm_df['case_id'].astype(str) == case_id
            if gmask.any():
                g = gemm_df.loc[gmask].iloc[0]
                # prefer modeled AI if present
                if 'arithmetic_intensity_modeled' in g and not pd.isna(g['arithmetic_intensity_modeled']):
                    return float(g['arithmetic_intensity_modeled'])
                # fallback to flops/bytes_modeled
                if ('flops' in g and 'bytes_modeled' in g) and (g['bytes_modeled'] > 0):
                    return float(g['flops']) / float(g['bytes_modeled'])

    # last resort: estimate using common values from roofline
    # use median AI from roofline
    return float(roofline_df['arithmetic_intensity'].median())


def make_roofline_plot(consolidated, roofline, gemm_analysis, outpath):
    # prepare AI for consolidated rows
    consolidated = consolidated.copy()
    consolidated[['m','n','k']] = consolidated[['m','n','k']].astype(float)
    consolidated['m'] = consolidated['m'].astype(int)
    consolidated['n'] = consolidated['n'].astype(int)
    consolidated['k'] = consolidated['k'].astype(int)

    consolidated['arithmetic_intensity'] = consolidated.apply(lambda r: infer_ai(r, roofline, gemm_analysis), axis=1)

    # pick representative compute_roof and memory_roofs
    compute_roof = float(roofline['compute_roof_gflops'].median())
    memory_roofs_gbps = roofline['memory_roof_gbps'].dropna().unique()
    if len(memory_roofs_gbps) == 0:
        memory_roofs_gbps = np.array([roofline['memory_roof_gbps'].median()])

    # x range
    ais = consolidated['arithmetic_intensity'].replace(0, np.nan).dropna()
    xmin = max(1e-3, ais.min() * 0.5)
    xmax = max(ais.max() * 5.0, xmin * 10)
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 512)

    plt.figure(figsize=(8,6))
    # plot memory roof lines
    for mb in memory_roofs_gbps:
        plt.plot(xs, mb * xs, linestyle='--', linewidth=1, label=f'mem_roof {mb:.3g} GB/s')
    # compute roof horizontal
    plt.plot(xs, [compute_roof]*len(xs), color='black', linewidth=1.5, label=f'compute_roof {compute_roof:.3g} GF/s')

    # plot measured consolidated points
    x = consolidated['arithmetic_intensity'].values
    y = consolidated['measured_gflops_mean'].astype(float).values
    sizes = consolidated['n'].values
    proxy_mask = consolidated.get('proxy', False).astype(str).map({'True':True,'true':True,'TRUE':True,'1':True}).fillna(False)

    plt.scatter(x[~proxy_mask], y[~proxy_mask], c='tab:blue', s=50, label='GEMM')
    if proxy_mask.any():
        plt.scatter(x[proxy_mask], y[proxy_mask], c='tab:orange', marker='s', s=70, label='GEMM (proxy)')

    # annotate by n
    for xi, yi, ni in zip(x, y, sizes):
        plt.text(xi, yi, str(int(ni)), fontsize=8, ha='left', va='bottom')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOPs / byte)')
    plt.ylabel('Measured GFLOPS')
    plt.title('Roofline: GEMM measurements')
    plt.legend()
    plt.grid(True, which='both', ls=':', linewidth=0.5)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def make_pes_vs_size_plot(consolidated, outpath):
    df = consolidated.copy()
    df['n'] = df['n'].astype(int)
    df['PES'] = df['PES'].astype(float)

    proxy_mask = df.get('proxy', False).astype(str).map({'True':True,'true':True,'TRUE':True,'1':True}).fillna(False)

    plt.figure(figsize=(8,5))
    plt.plot(df.loc[~proxy_mask, 'n'], df.loc[~proxy_mask, 'PES'], 'o-', label='PES')
    if proxy_mask.any():
        plt.plot(df.loc[proxy_mask, 'n'], df.loc[proxy_mask, 'PES'], 's', color='orange', label='proxy')

    plt.xscale('log', base=2)
    plt.xlabel('Matrix size n (log2)')
    plt.ylabel('PES')
    plt.title('Portable Efficiency Score vs matrix size')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    consolidated, roofline, gemm_analysis = load_data(repo_root)

    # ensure numeric columns
    for c in ['measured_gflops_mean','compute_roof_gflops','effective_roof_gflops','PES']:
        if c in consolidated.columns:
            consolidated[c] = pd.to_numeric(consolidated[c], errors='coerce')

    outdir = os.path.join(repo_root, 'build', 'reports', 'roofline_20260311175904', 'plots')
    roofline_png = os.path.join(outdir, 'roofline.png')
    pes_png = os.path.join(outdir, 'pes_vs_size.png')

    make_roofline_plot(consolidated, roofline, gemm_analysis, roofline_png)
    make_pes_vs_size_plot(consolidated, pes_png)

    print('Saved:', roofline_png, pes_png)


if __name__ == '__main__':
    main()
