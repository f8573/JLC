package number;

import linear.Matrix;
import linear.Set;
import linear.Vector;

public class Main {
    public static void main(String[] args) {
        Vector v1 = new Vector(new Integer(1),new Integer(2),new Integer(3));
        Vector v2 = new Vector(new Integer(4),new Integer(5),new Integer(6));
        Vector v3 = new Vector(new Integer(7),new Integer(8),new Integer(9));
        //System.out.println(v1.dot(v2));
        Matrix matrix = new Matrix(v1,v2,v3);
        //matrix.print();
        //matrix.transpose().print();
        //matrix.multiply(matrix.transpose()).print();
        Matrix[] matrices = matrix.LU();
//        matrices[0].print(); //L
//        matrices[1].print(); //reduced
        //matrices[2].print(); //?
        //matrices[0].multiply(matrices[1]).print();
        //matrix.rref().print();
        Vector v4 = new Vector(new Integer(1),new Integer(2),new Integer(3));
        Vector v5 = new Vector(new Integer(4),new Integer(5),new Integer(6));
        Vector v6 = new Vector(new Integer(7),new Integer(8),new Integer(10));
        Matrix matrix1 = new Matrix(v4,v5,v6);
        //matrix1.print();
        //matrix1.rref().print();
        //matrix1.inverse().print();
        matrix1.reduced();
        //System.out.println(matrix1.triangularDeterminant());
        //System.out.println(matrix1.cofactorDeterminant());
        //System.out.println(matrix1.minor(0,0).determinant());
        Set set = new Set(v4,v5,v6);
        set.print();
        //System.out.println(set.set[0].dot(set.set[1]));
        //System.out.println(set.set[0].dot(set.set[0]));
        set.orthogonalize().print();
        set.orthonormalize().print();
    }
}
