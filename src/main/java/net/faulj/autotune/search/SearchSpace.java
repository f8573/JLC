package net.faulj.autotune.search;

/**
 * Lazy search domain for BLIS-style GEMM blocking parameters.
 *
 * <p>Contains independent parameter ranges plus a {@link ConstraintSet}
 * encoding coupling predicates between parameters. This is a
 * <b>declarative</b> representation — no combinations are enumerated
 * or materialized.</p>
 *
 * <p>The parameters form a hierarchy for Phase 4 search:</p>
 * <ol>
 *   <li><b>Microkernel:</b> MR, NR (register pressure)</li>
 *   <li><b>Inner K blocking:</b> KC (L1 tier)</li>
 *   <li><b>Panel blocking:</b> NC (L2 tier)</li>
 *   <li><b>Outer blocking:</b> MC (L3 tier / thread-level)</li>
 *   <li><b>Unroll:</b> kUnroll (pipeline latency hiding)</li>
 * </ol>
 *
 * <p>Phase 4 traverses this hierarchy coarse-to-fine, using
 * {@link ConstraintSet#isAdmissible} to prune at each level.</p>
 */
public final class SearchSpace {

    // ── Microkernel tier ────────────────────────────────────────────────

    /** Microkernel rows. Constrained by register pressure. */
    public final ParameterRange mr;

    /** Microkernel columns. Typically fixed to vector length. */
    public final ParameterRange nr;

    // ── Blocking tiers (hierarchical: KC → NC → MC) ────────────────────

    /** K blocking (columns of A / rows of B). L1-constrained. */
    public final ParameterRange kc;

    /** N blocking (columns of B). L2-constrained. */
    public final ParameterRange nc;

    /** M blocking (rows of A). L3-constrained. */
    public final ParameterRange mc;

    // ── Pipeline tier ───────────────────────────────────────────────────

    /** K-loop unroll factor. Derived from FMA latency and issue width. */
    public final ParameterRange kUnroll;

    // ── Coupling constraints ────────────────────────────────────────────

    /** Formal predicates governing parameter admissibility. */
    public final ConstraintSet constraints;

    public SearchSpace(ParameterRange mr, ParameterRange nr,
                       ParameterRange kc, ParameterRange nc, ParameterRange mc,
                       ParameterRange kUnroll, ConstraintSet constraints) {
        this.mr = mr;
        this.nr = nr;
        this.kc = kc;
        this.nc = nc;
        this.mc = mc;
        this.kUnroll = kUnroll;
        this.constraints = constraints;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Search Domain {\n");
        sb.append("\n  Microkernel:\n");
        sb.append("    ").append(nr).append('\n');
        sb.append("    ").append(mr).append('\n');
        sb.append("\n  Inner K blocking:\n");
        sb.append("    ").append(kc).append('\n');
        sb.append("\n  Panel blocking:\n");
        sb.append("    ").append(nc).append('\n');
        sb.append("\n  Outer blocking:\n");
        sb.append("    ").append(mc).append('\n');
        sb.append("\n  Unroll:\n");
        sb.append("    ").append(kUnroll).append('\n');
        sb.append('\n');
        sb.append("  ").append(constraints.toString().replace("\n", "\n  "));
        sb.append("}");
        return sb.toString();
    }
}
