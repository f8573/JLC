package linear;

import number.*;
import number.Integer;

import java.util.Arrays;

public class Vector {
    Field[] numbers;
    int length;

    public Vector(Field... numbers) {
        this.numbers = numbers;
        length = numbers.length;
    }

    public Vector(Integer... numbers) {
        Field[] data = new Field[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            data[i] = numbers[i].toRational();
        }
        this.numbers = data;
        length = this.numbers.length;
    }

    public Vector(int... numbers) {
        Field[] data = new Field[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            data[i] = new Integer(numbers[i]).toRational();
        }
        this.numbers = data;
        length = this.numbers.length;
    }

    public Vector(int i) {
        Field[] data = new Field[i];
        for (int j = 0; j < i; j++) {
            data[j] = new Integer(0).toRational();
        }
        this.length = i;
        this.numbers = data;
    }

    public boolean isZero() {
        for (int i = 0; i < length; i++) {
            if (!numbers[i].isZero()) {
                return false;
            }
        }
        return true;
    }

    public Vector copy() {
        Field[] newNumbers = new Field[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            newNumbers[i] = numbers[i]; // Assuming Field is immutable, otherwise use a deep copy method for Field
        }
        return new Vector(newNumbers);
    }


    public Vector add(Vector vector) {
        Vector v = copy();
        if (vector.length > v.length) {
            Field[] resized = new Field[vector.length];
            for (int i = 0; i < resized.length; i++) {
                resized[i] = new Integer(0).toRational();
            }
            for (int i = 0; i < v.length; i++) {
                resized[i] = v.numbers[i];
            }
            v.numbers = resized;
        } else if (vector.length < v.length) {
            Field[] resized = new Field[v.length];
            for (int i = 0; i < resized.length; i++) {
                resized[i] = new Integer(0).toRational();
            }
            for (int i = 0; i < vector.length; i++) {
                resized[i] = vector.numbers[i];
            }
            vector.numbers = resized;
        }
        for (int i = 0; i < v.numbers.length; i++) {
            v.numbers[i] = (Field) v.numbers[i].add(vector.numbers[i]);
        }
        return v;
    }

    public Vector subtract(Vector vector) {
        Vector v = copy();
        if (vector.length > v.length) {
            Field[] resized = new Field[vector.length];
            for (int i = 0; i < resized.length; i++) {
                resized[i] = new Integer(0).toRational();
            }
            for (int i = 0; i < v.length; i++) {
                resized[i] = v.numbers[i];
            }
            v.numbers = resized;
        } else if (vector.length < v.length) {
            Field[] resized = new Field[v.length];
            for (int i = 0; i < resized.length; i++) {
                resized[i] = new Integer(0).toRational();
            }
            for (int i = 0; i < vector.length; i++) {
                resized[i] = vector.numbers[i];
            }
            vector.numbers = resized;
        }
        for (int i = 0; i < v.numbers.length; i++) {
            v.numbers[i] = v.numbers[i].subtract(vector.numbers[i]);
        }
        return v;
    }

    public Field dot(Vector vector) {
        Rational result = new Rational(0,1);
        Vector v = copy();
        for (int i = 0; i < Math.min(v.numbers.length, vector.numbers.length); i++) {
            result = (Rational) result.add(v.numbers[i].multiply(vector.numbers[i]));
        }
        return result;
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

    public Vector projection(Set set) {
        Vector vector = new Vector(set.length);
        for (int i = 0; i < length; i++) {
            vector = vector.add(projection(set.set[i]));
        }
        return vector;
    }

    public Vector projection(Vector vector) {
        //projecting this onto a vector (proj_a (b), b=this, a=vector)
        Vector a = vector.copy();
        Vector b = copy();
        Rational coefficient = (Rational) b.dot(a).divide(a.dot(a));
        return a.multiply(coefficient);
    }

    public Vector bestApproximation(Set set) {
        return projection(set);
    }

    public SquareRoot distance(Set set) {
        return copy().subtract(projection(set)).length();
    }

    public Vector[] orthogonalDecomposition(Set set) {
        Vector yHat = projection(set);
        Vector z = copy().subtract(yHat);
        return new Vector[]{yHat,z};
    }

    public void reduce() {
        for(Field field : numbers) {
            if (field instanceof Rational) {
                ((Rational) field).reduce();
            }
        }
    }

    public void print() {
        System.out.println(this);
    }

    public Vector resize(int newSize) {
        Vector newVec = new Vector(newSize);
        Vector vector = copy();
        for (int i = 0; i < vector.length; i++) {
            newVec.numbers[i] = vector.numbers[i];
        }
        return newVec;
    }

    //TODO: implement
    public Vector changeBasis(Set b, Set c) {
        return null;
    }

    @Override
    public String toString() {
        return Arrays.toString(numbers);
    }
}
