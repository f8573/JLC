package linear;

import number.Constants;
import number.Field;
import number.Integer;
import number.Rational;

import java.security.InvalidParameterException;
import java.util.*;

public class Matrix {
    Vector[] set;
    int rows;
    int columns;

    public Matrix(Vector... vectors) {
        set = vectors;
        columns = vectors.length;
        rows = vectors[0].length;
    }

    public Matrix(int i, int j) {
        //i rows j columns
        Vector[] data = new Vector[j];
        for (int k = 0; k < j; k++) {
            data[k] = new Vector(i);
        }
        set = data;
        rows = i;
        columns = j;
    }

    public Set nullSpaceBasis() {
        Matrix ns = solve(new Vector(rows))[2];
        if (ns.columns == 1 && ns.set[0].isZero()) {
            return new Set(ns.set);
        }
        for (int i = ns.columns-1; i >= 0; i--) {
            if (ns.getColumn(i).isZero()) {
                ns = ns.removeColumn(i);
            }
        }
        return new Set(ns.set);
    }

    public Set columnSpaceBasis() {
        Matrix cs = solve(new Vector(rows))[1];
        for (int i = cs.columns-1; i >= 0; i--) {
            if (cs.getColumn(i).isZero()) {
                cs = cs.removeColumn(i);
            }
        }
        return new Set(cs.set);
    }

    //TODO: implement
    public Matrix[] diagonalize() {
        return null;
    }

    public Matrix[] solve(Vector b) {
        //also gives the basis for the column space and null space
        Matrix matrix = copy();
        ArrayList<Vector> newset = new ArrayList<>(List.of(matrix.set));
        newset.add(b);
        Matrix augment = new Matrix(newset.toArray(new Vector[0]));
        if (matrix.triangularDeterminant().isZero()) {
            //for each zero row in the reduced matrix, there should be a zero row in the augmented matrix
            //if not, no solution
            int matrixZeros = 0;
            int augmentZeros = 0;
            for (int i = 0; i < matrix.rows; i++) {
                if (matrix.getRow(i).isZero()) {
                    matrixZeros++;
                }
                if (augment.getRow(i).isZero()) {
                    augmentZeros++;
                }
            }
            if (matrixZeros != augmentZeros) {
                System.out.println("No solution");
                //throw new UnsupportedOperationException("Unsolvable matrix");
                return null;
            } else {
                //System.out.println("onto");
                //identify columns with leading 1s
                Matrix reduced = augment.rref();
                ArrayList<java.lang.Integer> pivotColumns = new ArrayList<>();
                for (int i = 0; i < reduced.rows; i++) {
                    for (int j = 0; j < reduced.columns; j++) {
                        if (reduced.getValue(i,j).toInteger().value == 1) {
                            pivotColumns.add(j);
                            break;
                        }
                    }
                }
                //now, create an empty matrix with non pivot columns
                Matrix solution = new Matrix(augment.rows,augment.columns);
                for (int i = 0; i < augment.columns; i++) {
                    if (!pivotColumns.contains(i)) {
                        solution = solution.setColumn(i,reduced.getColumn(i));
                    }
                }
                for (int i = 0; i < solution.columns; i++) {
                    solution.set[i] = solution.set[i].resize(matrix.rows);
                }
                //dont include the augmented column
                for (int i = 0; i < solution.columns-1; i++) {
                    for (int j = 0; j < solution.rows; j++) {
                        solution = solution.setValue(j,i, (Rational) solution.getValue(j,i).multiply(new Integer(-1)));
                    }
                }
                for (int i = 0; i < solution.set.length-1; i++) {
                    if (!solution.getColumn(i).isZero()) {
                        solution = solution.setValue(i,i,new Rational(1));
                    }
                }
                Matrix columnSpaceBasis = new Matrix(matrix.rows,pivotColumns.size());
                Matrix nullSpaceBasis = solution.removeColumn(solution.columns-1);
                for (int i = 0; i < nullSpaceBasis.columns; i++) {
                    if (nullSpaceBasis.getColumn(i).isZero()) {
                        columnSpaceBasis.set[i] = matrix.set[i];
                    }
                }
                return new Matrix[]{solution,columnSpaceBasis,nullSpaceBasis};
            }
        } else {
            //System.out.println("one-to-one");
            //augment.ref().print();
            //augment.rref().print();
            Matrix columnSpaceBasis = matrix;
            Matrix nullSpaceBasis = new Matrix(new Vector(matrix.rows));
            return new Matrix[]{new Matrix(augment.rref().set[augment.columns-1]),columnSpaceBasis,nullSpaceBasis};
        }
    }

    public Vector leastSquaresSolution(Vector vector) {
        Matrix left = copy().transpose().multiply(copy());
        Matrix right = copy().transpose().multiply(new Matrix(vector));
        Vector b = right.set[0];
        return left.solve(b)[0].set[0];
    }

    public Matrix leastSquaresError(Vector vector) {
        Vector xHat = copy().leastSquaresSolution(vector);
        return null;
    }

    public Matrix copy() {
        Vector[] newVectors = new Vector[set.length];
        for (int i = 0; i < set.length; i++) {
            newVectors[i] = set[i].copy();
        }
        return new Matrix(newVectors);
    }

    public Matrix add(Matrix matrix) {
        dimensionMatch(matrix);
        Matrix source = copy();
        for (int i = 0; i < set.length; i++) {
            source.set[i] = source.set[i].add(matrix.set[i]);
        }
        return source;
    }

    public Matrix subtract(Matrix matrix) {
        dimensionMatch(matrix);
        Matrix source = copy();
        for (int i = 0; i < set.length; i++) {
            source.set[i] = source.set[i].subtract(matrix.set[i]);
        }
        return source;
    }

    public Matrix transpose() {
        Vector[] data = new Vector[rows];
        for (int i = 0; i < rows; i++) {
            Field[] vectorData = new Field[columns];
            for (int j = 0; j < columns; j++) {
                vectorData[j] = this.set[j].numbers[i];
            }
            data[i] = new Vector(vectorData);
        }
        return new Matrix(data);
    }

    public Matrix multiply(Matrix matrix) {
        //columns of a must match rows of b
        if (columns != matrix.rows) {
            throw new UnsupportedOperationException("Invalid dimensions for multiplication");
        }
        Matrix m = copy();
        Matrix multiplied = new Matrix(rows,matrix.columns);
        for (int i = 0; i < multiplied.rows; i++) {
            for (int j = 0; j < multiplied.columns; j++) {
                multiplied.set[j].numbers[i] = m.getRow(i).dot(matrix.getColumn(j));
            }
        }
        return multiplied;
    }

    public boolean zeroColumn(int i) {
          return getColumn(i).isZero();
    }

    public boolean zeroRow(int i) {
        return getRow(i).isZero();
    }

    public Matrix rowSwap(int source, int target) {
        Matrix matrix = copy().transpose();
        Vector temp = matrix.set[source];
        matrix.set[source] = matrix.set[target];
        matrix.set[target] = temp;
        return matrix.transpose();
    }

    public Matrix rowMultiply(Rational number, int row) {
        Matrix matrix = copy().transpose();
        matrix.set[row] = matrix.set[row].multiply(number);
        return matrix.transpose();
    }

    public Matrix rowMultiplyAdd(Rational number, int source, int target) {
        Matrix matrix = copy().transpose();
        matrix.set[target] = matrix.set[target].add(matrix.set[source].multiply(number));
        return matrix.transpose();
    }

    //TODO: actually make this return the right answer
    public Rational cofactorDeterminant() {
        requireSquare();
        Matrix matrix = copy();
        if (rows == 1) {
            return matrix.getValue(0,0);
        } else if (rows == 2) {
            return (Rational) ((Rational) matrix.getValue(0,0).multiply(matrix.getValue(1,1))).subtract((Field) matrix.getValue(0,1).multiply(matrix.getValue(1,0)));
        } else {
            Rational determinant = new Integer(0).toRational();
            for (int i = 0; i < columns; i++) {
                Integer sign = (Integer) new Integer(1).power(new Integer(i));
                Rational coefficient = matrix.getValue(0,i);
                Rational cofactor = matrix.minor(0,i).cofactorDeterminant();
                Rational fin = (Rational) coefficient.multiply(sign);
                fin = (Rational) fin.multiply(cofactor);
                determinant = (Rational) determinant.add(fin);
            }
            return determinant;
        }
    }

    public Rational triangularDeterminant() {
        requireSquare();
        Matrix matrix = copy().ref();
        Rational one = Constants.ONE;
        for (int i = 0; i < matrix.columns; i++) {
            one = (Rational) one.multiply(matrix.getValue(i,i));
        }
        return one;
    }

    public Vector cramer(Vector vector) {
        Vector result = new Vector(vector.length);
        Matrix matrix = copy();
        for (int i = 0; i < vector.length; i++) {
            Matrix replaced = copy();
            replaced.set[i] = vector;
            result.numbers[i] = replaced.triangularDeterminant().divide(matrix.triangularDeterminant());
        }
        return result;
    }

    public Matrix minor(int i, int j) {
        return copy().removeRow(i).removeColumn(j);
    }

    public Matrix removeColumn(int i) {
        Matrix matrix = copy();
        ArrayList<Vector> vectors = new ArrayList<>(List.of(matrix.set));
        vectors.remove(i);
        return new Matrix(vectors.toArray(new Vector[0]));
    }

    public Matrix removeRow(int i) {
        return copy().transpose().removeColumn(i).transpose();
    }

    public Matrix[] reduced() {
        Matrix[] matrices = copy().LU();
        Matrix matrix = matrices[1];
        Matrix inverted = matrices[2];
        //inverted.print();
        int px = 0;
        int py = 0;
        for (int i = 0; i < matrix.columns; i++) {
            if (py >= rows || px >= columns) {
                return new Matrix[]{matrix,inverted};
            }
            if (matrix.zeroRow(py)) {
                break;
            } else {
                if (matrix.zeroColumn(i) || matrix.getValue(py,i).isZero()) {
                    px++;
                } else {
                    inverted = inverted.rowMultiply((Rational) Constants.ONE.divide(matrix.getValue(px,px)),py);
                    matrix = matrix.rowMultiply((Rational) Constants.ONE.divide(matrix.getValue(px,px)),py);
                    for (int j = 0; j < matrix.rows; j++) {
                        if (j != py) {
                            inverted = inverted.rowMultiplyAdd((Rational) matrix.getValue(j,i).additiveInverse(),py,j);
                            matrix = matrix.rowMultiplyAdd((Rational) matrix.getValue(j,i).additiveInverse(),py,j);
                        }
                    }
                    px++;
                    py++;
                    //inverted.print();
                    //matrix.print();
                }
            }
        }
        return new Matrix[]{matrix,inverted};
    }

    public Matrix rref() {
        return reduced()[0];
    }

    public Matrix inverse() {
        requireSquare();
        return reduced()[1];
    }

    public Matrix ref() {
        return LU()[1];
    }

    public Matrix[] LU() {
        Matrix matrix = copy();
        Matrix L = identity(matrix.rows);
        Matrix I = identity(matrix.rows);
        //if its square
        int currentPivotX = 0;
        int currentPivotY = 0;
        for (int i = 0; i < matrix.set.length; i++) {
            //for each column, check if it is zero first
            if (matrix.zeroColumn(i)) {
                currentPivotX++;
            } else {
                //if no number at pivot position find one
                if (currentPivotX >= matrix.rows || currentPivotY >= matrix.columns) {
                    return new Matrix[]{L,matrix,I};
                }
                if (matrix.set[currentPivotX].numbers[currentPivotY].isZero()) {
                    for (int j = currentPivotY; j < matrix.set[0].numbers.length; j++) {
                        if (!matrix.set[i].numbers[j].isZero()) {
                            matrix = matrix.rowSwap(currentPivotY, j);
                            L = L.rowSwap(currentPivotY, j);
                            I = I.rowSwap(currentPivotY, j);
                            break;
                        }
                    }
                    //check if the to-be pivot row is a zero row. if so return the matrix
                    if (matrix.zeroRow(currentPivotY)) {
                        return new Matrix[]{L,matrix,I};
                    }
                    //now if it's still zero, continue to the next column
                    if (matrix.set[currentPivotX].numbers[currentPivotY].isZero()) {
                        currentPivotX++;
                    }
                }
                for (int j = currentPivotY+1; j < matrix.set[0].length; j++) {
                    //column i, row j
                    L = L.setValue(j,i, (Rational) matrix.getValue(j,i).divide(matrix.getValue(currentPivotY,currentPivotX)));
                    Rational coefficient = (Rational) matrix.getValue(j,i).additiveInverse().divide(matrix.getValue(currentPivotY,currentPivotX));
                    matrix = matrix.rowMultiplyAdd(coefficient,currentPivotY,j);
                    I = I.rowMultiplyAdd(coefficient,currentPivotY,j);

                }
                currentPivotX++;
                currentPivotY++;
            }
        }
        return new Matrix[]{L,matrix,I};
    }

    public Rational getValue(int i, int j) {
        //row i col j
        return (Rational) set[j].numbers[i];
    }

    public Matrix setValue(int i, int j, Rational value) {
        //row i col j
        Matrix matrix = copy();
        matrix.set[j].numbers[i] = value;
        return matrix;
    }

    public Vector getColumn(int i) {
        return set[i];
    }

    public Vector getRow(int i) {
        return transpose().set[i];
    }

    public Matrix setRow(int i, Vector vector) {
        Matrix matrix = copy();
        //System.out.println(matrix.transpose().set[i]);
        matrix.transpose().set[i] = vector;
        return matrix.transpose();
    }

    public Matrix setColumn(int i, Vector vector) {
        Matrix matrix = copy();
        matrix.set[i] = vector;
        return matrix;
    }

    public void dimensionMatch(Matrix matrix) {
        if (rows != matrix.rows || columns != matrix.columns) {
            throw new UnsupportedOperationException("Non-matching dimensions of matrices");
        }
    }

    public void requireSquare() {
        if (rows != columns) {
            throw new UnsupportedOperationException("Non-square matrix");
        }
    }

    public static Matrix zero(int i, int j) {
        Vector[] vectors = new Vector[j];
        for (int k = 0; k < j; k++) {
            Field[] vec = new Field[i];
            for (int l = 0; l < i; l++) {
                vec[l] = new Integer(0).toRational();
            }
            vectors[k] = new Vector(vec);
        }
        return new Matrix(vectors);
    }

    public static Matrix identity(int i) {
        Matrix matrix = Matrix.zero(i,i);
        for (int j = 0; j < i; j++) {
            matrix.set[j].numbers[j] = new Integer(1).toRational();
        }
        return matrix;
    }

    public static Vector e(int i) {
        Field[] data = new Field[i];
        for (int j = 0; j < i; j++) {
            data[j] = new Integer(0).toRational();
        }
        data[i-1] = new Integer(1).toRational();
        return new Vector(data);
    }

    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.print(set[j].numbers[i].toString()+" ");
            }
            System.out.println();
        }
    }
}
