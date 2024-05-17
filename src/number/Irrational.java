package number;

import java.util.List;

public class Irrational extends Field {
    List<Irrational> terms;

    public Irrational(Rational b, Rational e) {
        terms.add(new IrrationalSinglet(b, e));
    }

    public Irrational(Integer b, Rational e) {
        terms.add(new IrrationalSinglet(b, e));
    }

    public Irrational() {
        throw new RuntimeException("The irrational class can not be instantiated with a default.");
    }

    public Irrational(Irrational... items) {
        for (Irrational irrationalSinglet : items) {
            assert false;
            terms.add(irrationalSinglet);
        }
    }

    static class IrrationalSinglet extends Irrational {
        /**
         * basically a rational to the power of a rational
         * potential forms:
         * default:
         * irrational num
         * (a/b)^(c/d)
         * if c/d = e
         * we have rational num
         * (a^e/b^e)
         * if a/b = f
         * we have irrational num
         * f^(c/d)
         * if a/b = f and c/d = e
         * we have integer
         * f^e
         */

        Rational base;
        Rational exponent;
        Boolean sign; //false: negative     true: positive      null: zero

        public IrrationalSinglet(Rational b, Rational e) {
            base = b;
            exponent = e;
        }

        public IrrationalSinglet(Integer b, Rational e) {
            base = b.toRational();
            exponent = e;
        }

        public boolean isInteger() {
            return exponent.isInteger() && base.isInteger();
        }

        public boolean isRational() {
            return exponent.isInteger();
        }

        @Override
        public Field additiveInverse() {
            return null;
        }

        @Override
        public Field multiplicativeInverse() {
            return new IrrationalSinglet((Rational) base.multiplicativeInverse(), exponent);
        }

        @Override
        public Ring add(Ring ring) {
            return new Irrational();
        }

        @Override
        public Ring multiply(Ring ring) {
            return null;
        }
    }


    @Override
    public Field additiveInverse() {
        return null;
    }

    @Override
    public Field multiplicativeInverse() {
        return null;
    }

    @Override
    public Ring add(Ring ring) {
        return null;
    }

    @Override
    public Ring multiply(Ring ring) {
        return null;
    }
}
