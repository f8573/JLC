package net.faulj.orthogonal;

import net.faulj.vector.Vector;
import java.util.List;

/**
 * Algorithms for converting a set of vectors into an orthonormal basis.
 * <p>
 * Orthonormalization transforms a set of linearly independent vectors {x₁, ..., xₖ}
 * into a set of orthonormal vectors {q₁, ..., qₖ} that span the same subspace.
 * </p>
 *
 * <h2>Orthonormal Basis Properties:</h2>
 * <ul>
 * <li><b>Unit Length:</b> ||qᵢ|| = 1 for all i</li>
 * <li><b>Orthogonality:</b> qᵢ • qⱼ = 0 for all i ≠ j</li>
 * <li><b>Span Preservation:</b> Span{x₁, ..., xⱼ} = Span{q₁, ..., qⱼ} for all j</li>
 * </ul>
 *
 * <h2>Gram-Schmidt Process:</h2>
 * <p>
 * The classical algorithm constructs qₖ by subtracting components in the direction
 * of previous basis vectors:
 * </p>
 * <pre>
 * 1. v₁ = x₁
 * 2. v₂ = x₂ - proj<sub>v₁</sub>(x₂)
 * 3. vₖ = xₖ - Σ proj<sub>vᵢ</sub>(xₖ)  (for i=1 to k-1)
 *
 * Normalize: qₖ = vₖ / ||vₖ||
 * </pre>
 *
 * <h2>Modified Gram-Schmidt (MGS):</h2>
 * <p>
 * The classical algorithm is numerically unstable due to rounding errors. This class
 * typically implements MGS, where projections are removed sequentially from the
 * current vector to maintain orthogonality in finite precision arithmetic.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * List<Vector> rawBasis = Arrays.asList(v1, v2, v3);
 *
 * // Convert to orthonormal basis
 * List<Vector> orthoBasis = Orthonormalization.gramSchmidt(rawBasis);
 *
 * // Verify properties
 * double norm = orthoBasis.get(0).norm2(); // Should be 1.0
 * double dot = orthoBasis.get(0).dot(orthoBasis.get(1)); // Should be ~0.0
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>QR Decomposition</li>
 * <li>Simplifying projection calculations (P = QQ<sup>T</sup>)</li>
 * <li>Eigenvalue algorithms (Arnoldi/Lanczos iterations)</li>
 * <li>Improving numerical stability of basis sets</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.result.QRResult
 * @see net.faulj.spaces.SubspaceBasis
 */
public class Orthonormalization {
}