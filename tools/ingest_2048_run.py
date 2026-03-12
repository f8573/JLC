import os
import sys
import csv
import shutil
import datetime
import argparse
# avoid heavy third-party deps during ingestion; use stdlib csv for small edits

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
    with open(path, newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, path


def find_gemm_row_for_n(rows, n):
    # match kernel GEMM and m==n==k==n (rows are dicts from csv.DictReader)
    for r in rows:
        try:
            if r.get('kernel') == 'GEMM' and int(float(r.get('m', 0))) == n and int(float(r.get('n', 0))) == n and int(float(r.get('k', 0))) == n:
                return r
        except Exception:
            continue
    return None


def replace_proxy_row_in_csvs(new_row, source_archive_path, commit_message=None):
    updated_files = []
    for path in CONSOLIDATED_PATHS:
        if not os.path.exists(path):
            continue
        # read existing CSV
        with open(path, newline='', encoding='utf8') as f:
            reader = list(csv.DictReader(f))
        # find proxy 2048 row(s)
        new_rows = []
        removed = False
        for r in reader:
            try:
                if r.get('kernel') == 'GEMM' and int(float(r.get('n', 0))) == 2048 and (str(r.get('proxy', '')).lower() == 'true' or 'PROXY' in str(r.get('source_identifier', ''))):
                    removed = True
                    continue
            except Exception:
                pass
            new_rows.append(r)

        # build new consolidated row fields matching existing header order
        new_entry = {
            'kernel': 'GEMM',
            'm': str(int(float(new_row.get('m', new_row.get('n', 2048))))),
            'n': str(int(float(new_row.get('n', 2048)))),
            'k': str(int(float(new_row.get('k', new_row.get('n', 2048))))),
            'measured_gflops_mean': f"{float(new_row.get('measured_gflops', new_row.get('measured_gflops_mean', 0.0))):.4g}",
            'compute_roof_gflops': f"{float(new_row.get('compute_roof_gflops', new_row.get('compute_roof_gflops', 0.0))):.4g}",
            'effective_roof_gflops': f"{float(new_row.get('roof_gflops', new_row.get('effective_roof_gflops', 0.0))):.4g}",
            'PES': f"{float(new_row.get('portable_efficiency_score', 0.0)):.4g}",
            'bound_type': new_row.get('bound_type', ''),
            'memory_level': new_row.get('memory_level', ''),
            'source_file': os.path.relpath(source_archive_path, REPO_ROOT).replace('\\', '/'),
            'source_identifier': f"case=GEMM;n={int(float(new_row.get('n', 2048)))}",
            'proxy': 'false'
        }

        new_rows.append(new_entry)

        # determine fieldnames: preserve existing columns if possible, else use this order
        if reader:
            fieldnames = list(reader[0].keys())
            for k in new_entry.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        else:
            fieldnames = list(new_entry.keys())

        # write back
        with open(path, 'w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in new_rows:
                writer.writerow(r)
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

    # regenerate plots (best-effort)
    try:
        run_plots()
        print('Regenerated plots')
    except Exception as e:
        print('Plot generation skipped or failed:', e, file=sys.stderr)

    md = update_report_md(archive, row)
    if md:
        print('Updated', md)

    if args.commit:
        # commit changes with requested message
        msg = 'Add explicit square 2048 GEMM measurement and update consolidated PES/plots'
        files = [
            'build/reports/gemm_pes_consolidated.csv',
            'consolidated_gemm_pes.csv',
            'REPORT_GEMM_PES.md'
        ]
        # include any plot files under roofline_* archives
        import glob
        plot_paths = glob.glob(os.path.join(BUILD_REPORTS, 'roofline_*', 'plots', 'roofline.png'))
        plot_paths += glob.glob(os.path.join(BUILD_REPORTS, 'roofline_*', 'plots', 'pes_vs_size.png'))
        for p in plot_paths:
            files.append(os.path.relpath(p, REPO_ROOT).replace('\\', '/'))

        add_cmd = 'git add ' + ' '.join(f'"{f}"' for f in files)
        rc = os.system(add_cmd)
        if rc != 0:
            print('git add failed', file=sys.stderr)
        rc2 = os.system(f'git commit -m "{msg}"')
        if rc2 != 0:
            print('git commit failed', file=sys.stderr)
        # print short commit id
        os.system('git rev-parse --short HEAD')

if __name__ == "__main__":
    main(sys.argv)
