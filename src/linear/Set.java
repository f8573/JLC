package linear;

import number.Field;
import number.Rational;

public class Set {
    public Vector[] set;
    int length;

    public Set(Vector... vectors) {
        this.set = vectors;
        this.length = vectors.length;
    }

    public boolean linearlyIndependent() {
        return !(new Matrix(set).triangularDeterminant().isZero());
    }

    public Set[] gramSchmidt() {
        if (isOrthogonal()) {
            return new Set[]{this};
        }
        Vector[] u = new Vector[length];
        Vector[] e = new Vector[length];
        Vector[] v = set;

        u[0] = v[0];
        e[0] = u[0].unitary();

        for (int i = 1; i < length; i++) {
            Vector sum = new Vector(length);
            for (int j = 0; j < i; j++) {
                sum = sum.add(v[i].projection(u[j]));
            }
            u[i] = v[i].subtract(sum);
            e[i] = u[i].unitary();
        }

//        u[1] = v[1].subtract(v[1].projection(u[0]));
//        e[1] = u[1].unitary();
//        u[2] = v[2].subtract(v[2].projection(u[0])).subtract(v[2].projection(u[1]));
        //v[2].projection(u[0]).print();
        //v[2].projection(u[1]).print();
        //e[2] = u[2].unitary();
        return new Set[]{new Set(u), new Set(e)};
    }

    public Set orthogonalize() {
        return gramSchmidt()[0];
    }

    public Set orthonormalize() {
        return gramSchmidt()[1];
    }

    public boolean isOrthogonal() {
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length; j++) {
                if (i != j && !set[i].dot(set[j]).isZero()) {
                    return false;
                }
            }
        }
        return true;
    }

    public void print() {
        for (int i = 0; i < length; i++) {
            set[i].print();
        }
    }
}
