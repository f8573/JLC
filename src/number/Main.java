package number;

import linear.Vector;

public class Main {
    public static void main(String[] args) {
        Integer integer = new Integer(2);
        Rational rational = new Rational(2,3);
        Ring sum = rational.add(integer);
        System.out.println(integer.power(integer));
        System.out.println(rational.power(integer));
        System.out.println(rational.multiply(new SquareRoot((Rational) sum)));
        //System.out.println(new SquareRoot((Rational) sum).multiply(new SquareRoot((Rational) sum)));
        Vector vector = new Vector(new Integer(1).toRational(),new Integer(2).toRational());
        System.out.println(vector.unitary());
    }
}
