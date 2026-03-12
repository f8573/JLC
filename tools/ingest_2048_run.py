import os
import sys
import csv
import shutil
import datetime
import argparse
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD_REPORTS = os.path.join(REPO_ROOT, 'build', 'reports')
CONSOLIDATED_PATHS = [
    os.path.join(REPO_ROOT, 'build', 'reports', 'gemm_pes_consolidated.csv'),
    os.path.join(REPO_ROOT, 'consolidated_gemm_pes.csv')
]


def find_roofline_dir(explicit=None):
    if explicit:
        p = os.path.join(REPO_ROOT, explicit)
        if os.path.isdir(p):
            return p
    # prefer latest roofline_* archive
    if os.path.isdir(BUILD_REPORTS):
        candidates = [d for d in os.listdir(BUILD_REPORTS) if d.startswith('roofline_')]
        candidates_sorted = sorted(candidates, reverse=True)
        for cand in candidates_sorted:
            p = os.path.join(BUILD_REPORTS, cand)
            f = os.path.join(p, 'portable_efficiency_results.csv')
            if os.path.exists(f):
                return p
    # fallback
    p = os.path.join(BUILD_REPORTS, 'roofline')
    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'portable_efficiency_results.csv')):
        return p
    return None


def read_portable_csv(roofline_dir):
    path = os.path.join(roofline_dir, 'portable_efficiency_results.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path), path


def find_gemm_row_for_n(df, n):
    # match kernel GEMM and m==n==k==n
    mask = (df['kernel'] == 'GEMM') & (df['m'] == n) & (df['n'] == n) & (df['k'] == n)
    found = df.loc[mask]
    if found.shape[0] == 0:
        return None
    return found.iloc[0]


def replace_proxy_row_in_csvs(new_row, source_archive_path, commit_message=None):
    updated_files = []
    for path in CONSOLIDATED_PATHS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # find proxy 2048 row
        mask = (df['kernel'] == 'GEMM') & (df['n'] == 2048) & (df.get('proxy', False) == True)
        # Some proxy rows mark proxy as 'true' string; try alternative
        if mask.sum() == 0:
            # try string match in source_identifier
            mask = (df['kernel'] == 'GEMM') & (df['n'] == 2048) & df['source_identifier'].astype(str).str.contains('PROXY')
        if mask.sum() == 0:
            # nothing to replace; append
            append = True
        else:
            append = False
            df = df.loc[~mask]
        # build new consolidated row fields matching existing header
        new_entry = {
            'kernel': 'GEMM',
            'm': int(new_row['m']),
            'n': int(new_row['n']),
            'k': int(new_row['k']),
            'measured_gflops_mean': float(new_row.get('measured_gflops', new_row.get('measured_gflops_mean', 0.0))),
            'compute_roof_gflops': float(new_row.get('compute_roof_gflops', new_row.get('compute_roof_gflops', 0.0))),
            'effective_roof_gflops': float(new_row.get('roof_gflops', new_row.get('effective_roof_gflops', 0.0))),
            'PES': float(new_row.get('portable_efficiency_score', 0.0)),
            'bound_type': new_row.get('bound_type', ''),
            'memory_level': new_row.get('memory_level', ''),
            'source_file': os.path.relpath(source_archive_path, REPO_ROOT),
            'source_identifier': f"case=GEMM;n={int(new_row['n'])}",
            'proxy': False
        }
        # enforce numeric formatting to 4 significant digits when writing later
        # insert
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        # write back
        df.to_csv(path, index=False)
        updated_files.append(path)
    return updated_files


def archive_roofline(roofline_dir):
    # create timestamped copy under build/reports/roofline_<ts>
    ts = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    dest = os.path.join(BUILD_REPORTS, f'roofline_{ts}')
    if not os.path.exists(dest):
        os.makedirs(dest)
    # copy portable_efficiency_results.csv and json if present
    for name in ('portable_efficiency_results.csv', 'portable_efficiency_results.json'):
        s = os.path.join(roofline_dir, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dest, name))
    return dest


def run_plots():
    # invoke the plotting script
    script = os.path.join(REPO_ROOT, 'tools', 'plot_gemm_pes.py')
    rc = os.system(f'"{sys.executable}" "{script}"')
    if rc != 0:
        raise RuntimeError('Plot script failed')


def update_report_md(archive_dir, new_row):
    md = os.path.join(REPO_ROOT, 'REPORT_GEMM_PES.md')
    if not os.path.exists(md):
        return None
    rel = os.path.relpath(archive_dir, REPO_ROOT).replace('\\','/')
    insert_line = f"- Explicit square 2048 measurement added: [portable_efficiency_results.csv]({rel}/portable_efficiency_results.csv) (GEMM,2048,2048,2048)\n"
    with open(md, 'r', encoding='utf8') as f:
        text = f.read()
    if insert_line in text:
        return md
    # prepend under Primary sources section if present; otherwise append
    if '**Primary sources**' in text:
        parts = text.split('\n')
        # find location after Primary sources line
        out = []
        inserted = False
        for i, line in enumerate(parts):
            out.append(line)
            if line.strip().startswith('**Primary sources**') and not inserted:
                # next line add note
                out.append(insert_line)
                inserted = True
        new_text = '\n'.join(out)
    else:
        new_text = text + '\n' + insert_line
    with open(md, 'w', encoding='utf8') as f:
        f.write(new_text)
    return md


def main(argv):
    parser = argparse.ArgumentParser(description='Ingest an explicit GEMM n=2048 run and update consolidated CSVs and plots')
    parser.add_argument('--roofline-dir', help='Existing roofline directory (e.g. build/reports/roofline)')
    parser.add_argument('--commit', action='store_true', help='If set, commit CSVs and plots with the standard message')
    args = parser.parse_args(argv[1:])

    roofline_dir = find_roofline_dir(args.roofline_dir)
    if not roofline_dir:
        print('ERROR: roofline directory with portable_efficiency_results.csv not found', file=sys.stderr)
        sys.exit(2)
    df, src_path = read_portable_csv(roofline_dir)
    row = find_gemm_row_for_n(df, 2048)
    if row is None:
        print('ERROR: GEMM 2048 row not found in', src_path, file=sys.stderr)
        sys.exit(3)

    # archive
    archive = archive_roofline(roofline_dir)
    print('Archived roofline to', archive)

    updated = replace_proxy_row_in_csvs(row, archive)
    print('Updated consolidated files:', updated)

    # regenerate plots
    run_plots()
    print('Regenerated plots')

    md = update_report_md(archive, row)
    if md:
        print('Updated', md)

    if args.commit:
        # commit changes with requested message
        msg = 'Add explicit square 2048 GEMM measurement and update consolidated PES/plots'
        os.system(f'git add {" ".join(["build/reports/gemm_pes_consolidated.csv","consolidated_gemm_pes.csv","build/reports/roofline_*\/plots/roofline.png","build/reports/roofline_*\/plots/pes_vs_size.png","REPORT_GEMM_PES.md"]) }')
        os.system(f'git commit -m "{msg}"')
        # print short commit id
        os.system('git rev-parse --short HEAD')

if __name__ == "__main__":
    main(sys.argv)
