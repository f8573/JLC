package net.faulj.compiler.matrix.schedule;

/**
 * Immutable node in an M3 schedule tree.
 *
 * <p>A schedule is separate from the computation represented by an
 * {@code AffineProgram}. Nodes only describe ordering and annotations; they
 * never execute the statements they reference.</p>
 */
public sealed interface ScheduleNode
    permits ScheduleSequence, ScheduleBand, ScheduleStatement {

    /**
     * Render this node with the supplied indentation.
     */
    String dump(String indentation);

    default String dump() {
        return dump("");
    }
}
