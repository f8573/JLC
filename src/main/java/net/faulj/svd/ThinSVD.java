package net.faulj.svd;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SVDResult;

/**
 * Computes the Thin (Economy) Singular Value Decomposition.
 * <p>
 * This class represents the compact factorization A = U<sub>n</sub>Σ<sub>n</sub>V<sup>T</sup> for an m×n matrix (where m &ge; n):
 * </p>
 * <ul>
 * <li><b>U<sub>n</sub></b> - m×n matrix with orthonormal columns (U<sup>T</sup>U = I)</li>
 * <li><b>Σ<sub>n</sub></b> - n×n Square diagonal matrix of singular values</li>
 * <li><b>V</b> - n×n Orthogonal matrix of right singular vectors</li>
 * </ul>
 *
 * <h2>Thin vs Full SVD:</h2>
 * <table border="1">
 * <tr><th>Aspect</th><th>Full SVD</th><th>Thin SVD</th></tr>
 * <tr><td>Matrix U Size</td><td>m × m</td><td>m × n</td></tr>
 * <tr><td>Matrix Σ Size</td><td>m × n</td><td>n × n</td></tr>
 * <tr><td>Reconstruction</td><td>Exact (A = UΣVᵀ)</td><td>Exact (A = UₙΣₙVᵀ)</td></tr>
 * <tr><td>Nullspace of Aᵀ</td><td>Included in U columns (n+1 to m)</td><td>Discarded</td></tr>
 * <tr><td>Memory Usage</td><td>High (O(m²))</td><td>Efficient (O(mn))</td></tr>
 * </table>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Tall matrix (e.g., 1000 rows, 50 columns)
 * Matrix A = loadLargeMatrix();
 *
 * // Compute Economy SVD
 * ThinSVD thinSvd = new ThinSVD();
 * SVDResult result = thinSvd.decompose(A);
 *
 * // U is 1000x50 instead of 1000x1000
 * Matrix U = result.getU();
 * }</pre>
 *
 * <h2>Use Cases:</h2>
 * <ul>
 * <li>Least squares problems where m &gt; n</li>
 * <li>Image compression (only first k singular vectors needed)</li>
 * <li>Latent Semantic Indexing (LSI) in NLP</li>
 * </ul>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is thread-safe and stateless.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SVDecomposition
 */
public class ThinSVD {
	public ThinSVD() {
		throw new RuntimeException("Class unfinished");
	}
}