package number;

public abstract class Ring {
    public abstract Ring add(Ring ring);
    public abstract Ring multiply(Ring ring);

    public abstract boolean isZero();
}
