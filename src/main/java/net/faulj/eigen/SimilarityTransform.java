package net.faulj.eigen;

import net.faulj.matrix.Matrix;

/**
 * Represents and performs similarity transformations on matrices.
 * <p>
 * A similarity transformation of a matrix A by an invertible matrix P is defined as:
 * </p>
 * <pre>
 * B = P<sup>-1</sup>AP
 * </pre>
 * <p>
 * If P is orthogonal (P<sup>T</sup> = P<sup>-1</sup>), the transformation is an orthogonal similarity:
 * </p>
 * <pre>
 * B = P<sup>T</sup>AP
 * </pre>
 *
 * <h2>Invariant Properties:</h2>
 * <p>
 * Matrices A and B are "similar". Similar matrices share several fundamental properties:
 * </p>
 * <ul>
 * <li><b>Eigenvalues:</b> Characteristic polynomials are identical (det(A-λI) = det(B-λI)).</li>
 * <li><b>Determinant:</b> det(A) = det(B).</li>
 * <li><b>Trace:</b> tr(A) = tr(B).</li>
 * <li><b>Rank:</b> rank(A) = rank(B).</li>
 * <li><b>Jordan Normal Form:</b> Both share the same Jordan canonical form.</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>Simplification:</b> Transforming dense matrices into simpler forms (Diagonal, Hessenberg, Triangular)
 * to ease computations (like eigenvalue finding).</li>
 * <li><b>Change of Basis:</b> Represents the same linear operator viewed from a different coordinate basis P.</li>
 * <li><b>Iterative Algorithms:</b> The QR algorithm consists of a sequence of orthogonal similarity transformations.</li>
 * </ul>
 *
 * <h2>Standard Forms:</h2>
 * <p>
 * Common target forms for similarity transformations include:
 * </p>
 * <table border="1">
 * <tr><th>Target Form</th><th>Description</th><th>Transformation Matrix P</th></tr>
 * <tr><td><b>Hessenberg</b></td><td>Zeroes below first subdiagonal</td><td>Orthogonal (Householder)</td></tr>
 * <tr><td><b>Schur</b></td><td>Upper triangular (or quasi)</td><td>Orthogonal (Schur vectors)</td></tr>
 * <tr><td><b>Diagonal</b></td><td>Values only on diagonal</td><td>Eigenvectors (if diagonalizable)</td></tr>
 * <tr><td><b>Jordan</b></td><td>Jordan blocks on diagonal</td><td>Generalized Eigenvectors</td></tr>
 * </table>
 *
 * <h2>Immutability & Usage:</h2>
 * <p>
 * This class is designed to be a stateless utility or an immutable representation of a specific
 * transform instance.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.hessenberg.HessenbergReduction
 * @see net.faulj.decomposition.result.SchurResult
 */
public class SimilarityTransform {
}