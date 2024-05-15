abstract class Irrational extends Real<Irrational> {
    abstract Irrational add(Irrational other);
    abstract Irrational multiply(Irrational other);
    abstract Irrational divide(Irrational other);
    abstract Irrational negate();
    abstract Irrational inverse();
    abstract double eval();
}