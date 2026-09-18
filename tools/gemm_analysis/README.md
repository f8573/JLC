# GEMM Analysis Tools

## Plotting

```bash
py -3 tools/gemm_analysis/plot_gemm_analysis.py \
  --csv build/reports/gemm-analysis/gemm_analysis_runs.csv \
  --summary build/reports/gemm-analysis/gemm_analysis_summary.json \
  --out-dir build/reports/gemm-analysis
```

Dependencies:

- `pandas`
- `matplotlib`
- `numpy`

Generated figures:

- `roofline.png`
- `scaling.png`
- `cache_transitions.png`
- `simd_utilization.png`
- `stability.png`
