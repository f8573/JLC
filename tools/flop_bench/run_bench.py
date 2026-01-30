#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from multiprocessing import cpu_count

import platform

# Files live in the same directory as this script
HERE = os.path.dirname(__file__)
CPU_SRC = os.path.join(HERE, 'bench.cpp')
CUDA_SRC = os.path.join(HERE, 'bench_cuda.cu')

def find_compiler():
    # prefer g++/clang++, fall back to MSVC `cl`
    # allow override via CXX environment variable
    env_cxx = os.environ.get('CXX')
    if env_cxx:
        env_path = shutil.which(env_cxx) or env_cxx
        if env_path and os.path.exists(env_path):
            bn = os.path.basename(env_path).lower()
            if bn.startswith('cl'):
                return ('msvc', env_path)
            return ('gcc', env_path)
    for name in ('g++', 'clang++'):
        path = shutil.which(name)
        if path:
            return ('gcc', path)
    path = shutil.which('cl')
    if path:
        return ('msvc', path)
    return (None, None)

def compile_cpu(out_path, vectorize=True):
    kind, compiler = find_compiler()
    if kind is None:
        raise RuntimeError('No suitable C++ compiler found (need g++, clang++ or cl).')

    if kind == 'gcc':
        flags = ['-O3', '-std=c++17', '-fopenmp']
        if vectorize:
            flags += ['-march=native']
        else:
            flags += ['-fno-tree-vectorize']
        cmd = [compiler, CPU_SRC, '-o', out_path] + flags
    else:  # msvc
        # MSVC: cl /Fe:out.exe source.cpp /O2 /std:c++17 /openmp
        flags = ['/O2', '/std:c++17', '/openmp']
        if not vectorize:
            # no direct MSVC flag to disable vectorization; keep flags as-is
            pass
        cmd = [compiler, CPU_SRC, '/Fe' + out_path] + flags

    print('Compiling:', ' '.join(cmd))
    subprocess.check_call(cmd)

def compile_cuda(out_path, ccbin=None):
    cmd = ['nvcc', CUDA_SRC, '-lcublas', '-O3', '-o', out_path]
    if ccbin:
        cmd += ['-ccbin', ccbin]
    print('Compiling:', ' '.join(cmd))
    subprocess.check_call(cmd)

def run_proc(cmd, env=None):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    if proc.returncode != 0:
        print('Command failed:', ' '.join(cmd))
        print(proc.stderr)
        raise RuntimeError('process failed')
    return proc.stdout.strip()

def parse_json_line(line):
    try:
        return json.loads(line)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-N', type=int, default=1024)
    parser.add_argument('--iterations', '-i', type=int, default=3)
    parser.add_argument('--iter-list', type=str, default=None, help='Comma-separated list of iteration counts to sweep (overrides other iteration settings)')
    parser.add_argument('--cpu-iterations', type=int, default=1, help='Number of iterations to run for CPU tests (default 1)')
    parser.add_argument('--gpu-iter-start', type=int, default=16, help='Starting iteration count for GPU (power-of-two sequence)')
    parser.add_argument('--gpu-iter-max', type=int, default=262144, help='Maximum iteration count for GPU (power-of-two sequence)')
    parser.add_argument('--threads', '-t', nargs='*', type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--ccbin', type=str, default=None, help='Path to host compiler for nvcc (cl.exe)')
    args = parser.parse_args()

    sizes = [args.size]
    # Determine iteration lists for CPU and GPU separately
    if args.iter_list:
        common_list = [int(x) for x in args.iter_list.split(',') if x.strip()]
        cpu_iters_list = common_list
        gpu_iters_list = common_list
    else:
        cpu_iters_list = [args.cpu_iterations]
        # GPU uses powers of two from start to max
        gpu_iters_list = []
        v = args.gpu_iter_start
        while v <= args.gpu_iter_max:
            gpu_iters_list.append(v)
            v *= 2
    max_threads = cpu_count()
    if args.threads:
        thread_list = args.threads
    else:
        thread_list = [1, min(2, max_threads), min(4, max_threads), min(8, max_threads), max_threads]
        thread_list = sorted(set([t for t in thread_list if t >= 1]))

    work = []

    # Compile CPU variants
    bin_vec = os.path.join(HERE, 'bench_vec')
    bin_novec = os.path.join(HERE, 'bench_novec')
    compile_cpu(bin_vec, vectorize=True)
    compile_cpu(bin_novec, vectorize=False)

    for precision in ['float', 'double']:
        for variant, exe in [('vectorized', bin_vec), ('no-vector', bin_novec)]:
            for t in thread_list:
                for iters in cpu_iters_list:
                    cmd = [exe, '-p', precision, '-N', str(args.size), '-i', str(iters), '-t', str(t)]
                    env = os.environ.copy()
                    env['OMP_NUM_THREADS'] = str(t)
                    print('Running:', ' '.join(cmd))
                    out = run_proc(cmd, env=env)
                    j = parse_json_line(out.splitlines()[-1])
                    if j is None:
                        print('Failed to parse output:', out)
                        continue
                    j.update({'backend':'cpu','variant':variant, 'iterations': iters})
                    work.append(j)

    # CUDA
    if not args.no_cuda and shutil.which('nvcc'):
        bin_cuda = os.path.join(HERE, 'bench_cuda')
        ccbin = args.ccbin or os.environ.get('NVCC_CC')
        compile_cuda(bin_cuda, ccbin=ccbin)
        for precision in ['float', 'double']:
            for iters in gpu_iters_list:
                cmd = [bin_cuda, '-p', precision, '-N', str(args.size), '-i', str(iters)]
                print('Running GPU:', ' '.join(cmd))
                out = run_proc(cmd)
                j = parse_json_line(out.splitlines()[-1])
                if j is None:
                    print('Failed to parse GPU output:', out)
                else:
                    j.update({'backend':'cuda','variant':'cublas', 'iterations': iters})
                    work.append(j)
    else:
        print('nvcc not found or CUDA disabled; skipping GPU tests.')

    # Output results
    summary = {'results': work}
    print('\nJSON Summary:')
    print(json.dumps(summary, indent=2))

    # Generate HTML report with plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io, base64
        import numpy as np

        # Prepare dataframes
        cpu_results = [r for r in work if r.get('backend') == 'cpu']
        gpu_results = [r for r in work if r.get('backend') == 'cuda']

        # Unique thread counts and iterations
        threads = sorted({int(r['threads']) for r in cpu_results})
        iterations_vals = sorted({int(r['iterations']) for r in work})

        # 1) CPU line plot: avg, max, min GFLOPs vs threads
        cpu_stats = {}
        for t in threads:
            vals = [r['gflops'] for r in cpu_results if int(r['threads']) == t]
            if vals:
                cpu_stats[t] = {'avg': float(np.mean(vals)), 'max': float(np.max(vals)), 'min': float(np.min(vals))}

        fig1, ax1 = plt.subplots()
        xs = list(cpu_stats.keys())
        ax1.plot(xs, [cpu_stats[x]['avg'] for x in xs], label='avg')
        ax1.plot(xs, [cpu_stats[x]['max'] for x in xs], label='max')
        ax1.plot(xs, [cpu_stats[x]['min'] for x in xs], label='min')
        ax1.set_xlabel('Threads')
        ax1.set_ylabel('GFLOPs')
        ax1.set_title('CPU GFLOPs vs Threads')
        ax1.legend()
        buf1 = io.BytesIO(); fig1.savefig(buf1, format='png'); buf1.seek(0)
        img1 = base64.b64encode(buf1.read()).decode('ascii')
        plt.close(fig1)

        # 2) Heatmap: iterations (x) vs threads (y) -> GFLOPs (mean)
        iters_sorted = iterations_vals
        t_sorted = threads
        heat = np.zeros((len(t_sorted), len(iters_sorted)))
        for i,t in enumerate(t_sorted):
            for j,it in enumerate(iters_sorted):
                vals = [r['gflops'] for r in cpu_results if int(r['threads'])==t and int(r['iterations'])==it]
                heat[i,j] = float(np.mean(vals)) if vals else 0.0

        fig2, ax2 = plt.subplots()
        im = ax2.imshow(heat, aspect='auto', cmap='RdYlGn_r')
        ax2.set_xticks(np.arange(len(iters_sorted))); ax2.set_xticklabels(iters_sorted)
        ax2.set_yticks(np.arange(len(t_sorted))); ax2.set_yticklabels(t_sorted)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Threads')
        ax2.set_title('CPU GFLOPs Heatmap (Red=min, Green=max)')
        fig2.colorbar(im, ax=ax2, label='GFLOPs')
        buf2 = io.BytesIO(); fig2.savefig(buf2, format='png'); buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode('ascii')
        plt.close(fig2)

        # 3) Compare max CPU per thread to max GPU
        cpu_maxs = [cpu_stats[t]['max'] if t in cpu_stats else 0.0 for t in t_sorted]
        gpu_max = 0.0
        if gpu_results:
            gpu_max = max([r['gflops'] for r in gpu_results])

        fig3, ax3 = plt.subplots()
        ax3.bar([str(t) for t in t_sorted], cpu_maxs, label='CPU max per thread')
        ax3.plot([ -1 + i for i in range(len(t_sorted))], [gpu_max]*len(t_sorted), color='red', linestyle='--', label='GPU max')
        ax3.set_xlabel('Threads')
        ax3.set_ylabel('GFLOPs')
        ax3.set_title('CPU max GFLOPs vs GPU max GFLOPs')
        ax3.legend()
        buf3 = io.BytesIO(); fig3.savefig(buf3, format='png'); buf3.seek(0)
        img3 = base64.b64encode(buf3.read()).decode('ascii')
        plt.close(fig3)

        # Write HTML
        html_path = os.path.join(HERE, 'flop_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write('<html><head><meta charset="utf-8"><title>FLOPs Report</title></head><body>')
            f.write('<h1>FLOPs Benchmark Report</h1>')
            f.write('<h2>Summary (JSON)</h2>')
            f.write('<pre>')
            f.write(json.dumps(summary, indent=2))
            f.write('</pre>')
            f.write('<h2>CPU GFLOPs vs Threads</h2>')
            f.write(f'<img src="data:image/png;base64,{img1}"/>')
            f.write('<h2>CPU GFLOPs Heatmap (iterations x threads)</h2>')
            f.write(f'<img src="data:image/png;base64,{img2}"/>')
            f.write('<h2>CPU max per thread vs GPU max</h2>')
            f.write(f'<img src="data:image/png;base64,{img3}"/>')
            f.write('</body></html>')

        print('\nHTML report written to', html_path)
    except Exception as e:
        print('Failed to generate plots or HTML report:', e)

if __name__ == '__main__':
    main()
