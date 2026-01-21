package net.faulj.solve;

import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Solves linear systems Ax = b using Cramer's Rule.
 * <p>
 * This class implements a direct method for solving systems of linear equations
 * where the unique solution is expressed in terms of determinants.
 * </p>
 *
 * <h2>Mathematical Formulation:</h2>
 * <p>
 * For a system Ax = b with an n×n invertible matrix A, the components x<sub>i</sub>
 * of the solution vector are given by:
 * </p>
 * <pre>
 * x<sub>i</sub> = det(A<sub>i</sub>) / det(A)
 * </pre>
 * <p>
 * Where:
 * </p>
 * <ul>
 * <li><b>det(A):</b> The determinant of the coefficient matrix A (must be non-zero).</li>
 * <li><b>A<sub>i</sub>:</b> The matrix formed by replacing the i-th column of A with the vector b.</li>
 * </ul>
 *
 * <h2>Computational Complexity:</h2>
 * <p>
 * Cramer's rule is computationally expensive compared to decomposition methods like LU or QR.
 * </p>
 * <ul>
 * <li><b>Determinant Calculation:</b> Typically O(n³) using LU decomposition.</li>
 * <li><b>Total Cost:</b> Requires n+1 determinant calculations, leading to O(n⁴) complexity.</li>
 * <li><b>Memory:</b> Requires copying matrices to create A<sub>i</sub>, leading to high allocation rates.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Define a 2x2 system
 * Matrix A = new Matrix(new double[][]{{3, 1}, {1, 2}});
 * Vector b = new Vector(new double[]{9, 8});
 *
 * // Solve using Cramer's Rule
 * CramerSolver solver = new CramerSolver();
 * Vector x = solver.solve(A, b);
 *
 * System.out.println(x); // Output: [2.0, 3.0]
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Solving very small systems (n ≤ 3) where overhead of other methods is negligible.</li>
 * <li>Theoretical derivations and algebraic proofs.</li>
 * <li>Explicit formula generation for symbolic computation.</li>
 * </ul>
 *
 * <h2>Limitations:</h2>
 * <p>
 * This implementation is <b>not recommended</b> for large matrices (n &gt; 4) due to numerical instability
 * and high computational cost compared to {@link LUSolver}.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.determinant.LUDeterminant
 * @see LUSolver
 */
public class CramerSolver implements LinearSolver {

    /**
     * Solves the linear system Ax = b.
     *
     * <h2>Algorithm:</h2>
     * <ol>
     * <li>Verify A is square and dimensions match b.</li>
     * <li>Compute det(A). If close to zero, throw ArithmeticException.</li>
     * <li>For each column i from 0 to n-1:
     * <ul>
     * <li>Construct matrix A<sub>i</sub> by replacing column i of A with b.</li>
     * <li>Compute det(A<sub>i</sub>).</li>
     * <li>Calculate x<sub>i</sub> = det(A<sub>i</sub>) / det(A).</li>
     * </ul>
     * </li>
     * </ol>
     *
     * @param A The coefficient matrix (must be square and non-singular).
     * @param b The constant vector.
     * @return The solution vector x satisfying Ax = b.
     * @throws IllegalArgumentException if A is not square or dimensions mismatch.
     * @throws ArithmeticException if A is singular (det(A) ≈ 0).
     */
    @Override
    public Vector solve(Matrix A, Vector b) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Cramer's rule requires a square matrix");
        }
        if (b.dimension() != A.getRowCount()) {
            throw new IllegalArgumentException("Dimension mismatch");
        }

        double detA = LUDeterminant.compute(A);
        if (Math.abs(detA) < 1e-12) {
            throw new ArithmeticException("Matrix is singular");
        }

        int n = A.getRowCount();
        double[] x = new double[n];

        for (int i = 0; i < n; i++) {
            // Construct Ai: Matrix A with column i replaced by b
            Matrix Ai = replaceColumn(A, i, b);
            x[i] = LUDeterminant.compute(Ai) / detA;
        }

        return new Vector(x);
    }

    /**
     * Helper method to create the matrix A<sub>i</sub>.
     * <p>
     * Creates a deep copy of matrix A where the specified column is replaced
     * by the vector b.
     * </p>
     *
     * @param A The original matrix.
     * @param colIndex The index of the column to replace.
     * @param b The vector to insert into the column.
     * @return A new Matrix instance representing A<sub>i</sub>.
     */
    private Matrix replaceColumn(Matrix A, int colIndex, Vector b) {
        Vector[] originalData = A.getData();
        Vector[] newData = new Vector[originalData.length];

        for (int i = 0; i < originalData.length; i++) {
            if (i == colIndex) {
                newData[i] = b.copy();
            } else {
                newData[i] = originalData[i].copy();
            }
        }
        return new Matrix(newData);
    }
}