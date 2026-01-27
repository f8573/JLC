package net.faulj.matrix;

/**
 * Factory class for creating various types of {@link Matrix} instances.
 * <p>
 * This class provides static convenience methods to generate common matrix structures
 * used in linear algebra, testing, and initialization. It abstracts the underlying
 * storage implementation (dense vs sparse) where applicable.
 * </p>
 *
 * <h2>Supported Matrix Types:</h2>
 * <ul>
 * <li><b>Zero Matrix:</b> All elements are 0.0.</li>
 * <li><b>Identity Matrix (I):</b> Square matrix with 1.0 on the diagonal and 0.0 elsewhere.</li>
 * <li><b>Diagonal Matrix:</b> Elements provided by a vector, 0.0 off-diagonal.</li>
 * <li><b>Random Matrix:</b> Elements drawn from a uniform or normal distribution.</li>
 * <li><b>Ones Matrix:</b> All elements are 1.0.</li>
 * </ul>
 *
 * <h2>Creation Patterns:</h2>
 * <p>
 * The factory ensures that created matrices are dimensionally valid.
 * </p>
 * <pre>{@code
 * // 3x3 Identity
 * Matrix I = MatrixFactory.identity(3);
 *
 * // 5x5 Matrix with random values between 0 and 1
 * Matrix R = MatrixFactory.random(5, 5);
 *
 * // 4x4 Diagonal matrix from array
 * Matrix D = MatrixFactory.diagonal(new double[]{1, 2, 3, 4});
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Initializing weights in neural networks (Random).</li>
 * <li>Setting up systems of equations (Identity/Zero).</li>
 * <li>Unit testing with predictable structures.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.matrix.Matrix
 */
public class MatrixFactory {
	private MatrixFactory() {
	}

	/**
	 * Create a dense heap-backed matrix.
	 *
	 * @param rows number of rows
	 * @param cols number of columns
	 * @return heap-backed matrix
	 */
	public static Matrix create(int rows, int cols) {
		return new Matrix(rows, cols);
	}

	/**
	 * Create an off-heap matrix with 64-byte alignment.
	 *
	 * @param rows number of rows
	 * @param cols number of columns
	 * @return off-heap matrix
	 */
	public static OffHeapMatrix createOffHeap(int rows, int cols) {
		return new OffHeapMatrix(rows, cols);
	}
}