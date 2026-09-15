package net.faulj.compiler.matrix.schedule;

/**
 * Direction of a dependence component from source to sink.
 */
public enum DependenceDirection {
    LESS("<"),
    EQUAL("="),
    GREATER(">"),
    UNKNOWN("?");

    private final String symbol;

    DependenceDirection(String symbol) {
        this.symbol = symbol;
    }

    public String symbol() {
        return symbol;
    }

    @Override
    public String toString() {
        return symbol;
    }
}
