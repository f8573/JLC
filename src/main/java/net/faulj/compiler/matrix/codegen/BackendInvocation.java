package net.faulj.compiler.matrix.codegen;

/** One complete runtime-boundary invocation used by backend calibration. */
@FunctionalInterface
public interface BackendInvocation {
    /** Return false only for a normal, pre-invocation availability miss. */
    boolean invoke();
}
