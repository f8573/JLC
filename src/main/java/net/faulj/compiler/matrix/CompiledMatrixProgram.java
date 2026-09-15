package net.faulj.compiler.matrix;

import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.DependenceGraph;
import net.faulj.compiler.matrix.cpu.CpuExecutionPlan;
import net.faulj.compiler.matrix.cpu.CpuLowerer;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.matrix.Matrix;

/**
 * Inspectable result of the complete M1-to-M4 compiler pipeline.
 *
 * <p>The stages are retained as immutable references. Calling
 * {@link #execute()} runs only the lowered CPU plan; it never interprets the
 * affine IR and it continues to route matrix multiplication through the
 * existing production GEMM facade.</p>
 */
public final class CompiledMatrixProgram {
    private final ExecutionPlan expressionPlan;
    private final AffineProgram affineProgram;
    private final SchedulePlan schedulePlan;
    private final CpuExecutionPlan cpuExecutionPlan;

    private CompiledMatrixProgram(ExecutionPlan expressionPlan,
                                  AffineProgram affineProgram,
                                  SchedulePlan schedulePlan,
                                  CpuExecutionPlan cpuExecutionPlan) {
        this.expressionPlan = Objects.requireNonNull(expressionPlan, "Expression plan must not be null");
        this.affineProgram = Objects.requireNonNull(affineProgram, "Affine program must not be null");
        this.schedulePlan = Objects.requireNonNull(schedulePlan, "Schedule plan must not be null");
        this.cpuExecutionPlan = Objects.requireNonNull(
            cpuExecutionPlan, "CPU execution plan must not be null");
    }

    static CompiledMatrixProgram from(ExecutionPlan expressionPlan) {
        if (expressionPlan == null) {
            throw new IllegalArgumentException("Expression plan must not be null");
        }
        AffineProgram affine = AffineProgram.lower(expressionPlan);
        SchedulePlan schedule = CpuLowerer.defaultSchedule(affine);
        CpuExecutionPlan cpu = CpuLowerer.lower(schedule, expressionPlan);
        return new CompiledMatrixProgram(expressionPlan, affine, schedule, cpu);
    }

    public ExecutionPlan expressionPlan() {
        return expressionPlan;
    }

    public ExecutionPlan plan() {
        return expressionPlan;
    }

    public AffineProgram affineProgram() {
        return affineProgram;
    }

    public DependenceGraph dependenceGraph() {
        return affineProgram.dependenceGraph();
    }

    public DependenceGraph dependencies() {
        return dependenceGraph();
    }

    public SchedulePlan schedule() {
        return schedulePlan;
    }

    public SchedulePlan schedulePlan() {
        return schedulePlan;
    }

    public CpuExecutionPlan cpuPlan() {
        return cpuExecutionPlan;
    }

    public CpuExecutionPlan cpuExecutionPlan() {
        return cpuExecutionPlan;
    }

    /** Execute the lowered CPU plan against the graph's current input values. */
    public Matrix execute() {
        return cpuExecutionPlan.execute();
    }

    /**
     * Render each compiler stage in deterministic order for review and
     * debugging.
     */
    public String dump() {
        return "expression plan:\n" + expressionPlan.dump()
            + "\n\naffine program:\n" + affineProgram.dump()
            + "\n\nschedule:\n" + schedulePlan.dump()
            + "\n\n" + cpuExecutionPlan.dump();
    }

    @Override
    public String toString() {
        return dump();
    }
}
