package linear;

import number.*;
import number.Integer;

import java.util.Arrays;

public class Vector {
    Field[] numbers;

    public Vector(Field... numbers) {
        this.numbers = numbers;
    }

    public Vector copy() {
        return new Vector(numbers);
    }

    public Vector add(Vector vector) {
        Vector v = copy();
        for (int i = 0; i < v.numbers.length; i++) {
            v.numbers[i] = (Field) v.numbers[i].add(vector.numbers[i]);
        }
        return v;
    }

    public Vector subtract(Vector vector) {
        Vector v = copy();
        for (int i = 0; i < v.numbers.length; i++) {
            v.numbers[i] = (Field) v.numbers[i].subtract(vector.numbers[i]);
        }
        return v;
    }

    public Vector dot(Vector vector) {
        Rational result = new Rational(0,1);
        Vector v = copy();
        for (int i = 0; i < v.numbers.length; i++) {
            result = (Rational) result.add(v.numbers[i].multiply(vector.numbers[i]));
        }
        return v;
    }

    public SquareRoot length() {
        Rational result = new Rational(0,1);
        Vector v = copy();
        for (int i = 0; i < v.numbers.length; i++) {
            result = (Rational) result.add(((Rational)v.numbers[i]).power(new Integer(2)));
        }
        return new SquareRoot(result);
    }

    public Vector multiply(Rational number) {
        Vector v = copy();
        for (int i = 0; i < v.numbers.length; i++) {
            v.numbers[i] = (Field) v.numbers[i].multiply(number);
        }
        return v;
    }

    public Vector unitary() {
        Vector v = copy();
        SquareRoot length = v.length();
        for (int i = 0; i < v.numbers.length; i++) {
            v.numbers[i] = (Field) length.multiplicativeInverse().multiply(v.numbers[i]);
        }
        return v;
    }

    @Override
    public String toString() {
        return Arrays.toString(numbers);
    }
}
