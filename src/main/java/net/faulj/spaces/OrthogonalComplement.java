package net.faulj.spaces;

import net.faulj.vector.Vector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Computes and represents the orthogonal complement of a subspace.
 * <p>
 * For a subspace \( W \) of an inner product space \( V \), the orthogonal complement
 * \( W^\perp \) (pronounced "W perp") is the set of all vectors in \( V \) that are
 * orthogonal to every vector in \( W \).
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * \( W^\perp = \{ x \in V : \langle x, w \rangle = 0 \text{ for all } w \in W \} \)
 * </p>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Direct Sum:</b> \( V = W \oplus W^\perp \) (for finite dimensional spaces).</li>
 * <li><b>Unique Decomposition:</b> Any \( v \in V \) can be uniquely written as \( v = \hat{y} + z \)
 * where \( \hat{y} \in W \) and \( z \in W^\perp \).</li>
 * <li><b>Dimensions:</b> \( \dim(W) + \dim(W^\perp) = \dim(V) \).</li>
 * <li><b>Double Complement:</b> \( (W^\perp)^\perp = W \).</li>
 * </ul>
 *
 * <h2>Fundamental Subspaces:</h2>
 * <p>
 * In the context of a matrix \( A \):
 * </p>
 * <ul>
 * <li>(Row Space \( A \))\(^\perp\) = Null Space \( A \)</li>
 * <li>(Column Space \( A \))\(^\perp\) = Null Space \( A^T \) (Left Null Space)</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SubspaceBasis
 * @see Projection
 */
public final class OrthogonalComplement {

	/** Numerical threshold for treating a vector as (near) zero after orthogonalization. */
	private final double tol;

	/** Ambient dimension n. */
	private final int n;

	/** Orthonormal basis for W (rank r). */
	private final List<Vector> qW;

	/** Orthonormal basis for W^\perp (dimension n-r). */
	private final List<Vector> qPerp;

	private OrthogonalComplement(int n, double tol, List<Vector> qW, List<Vector> qPerp) {
		this.n = n;
		this.tol = tol;
		this.qW = Collections.unmodifiableList(qW);
		this.qPerp = Collections.unmodifiableList(qPerp);
	}

	/** Factory using a default tolerance. */
	public static OrthogonalComplement of(List<Vector> spanningSetForW) {
		return of(spanningSetForW, 1e-12);
	}

	/**
	 * Factory.
	 * @param spanningSetForW vectors spanning W (may be linearly dependent)
	 * @param tol numerical tolerance (e.g., 1e-12)
	 */
	public static OrthogonalComplement of(List<Vector> spanningSetForW, double tol) {
		Objects.requireNonNull(spanningSetForW, "spanningSetForW");
		if (spanningSetForW.isEmpty()) {
			throw new IllegalArgumentException("Spanning set is empty; ambient dimension is unknown.");
		}
		if (!(tol > 0.0)) {
			throw new IllegalArgumentException("tol must be > 0");
		}

		final int n = spanningSetForW.get(0).dimension();
		for (Vector v : spanningSetForW) {
			if (v.dimension() != n) {
				throw new IllegalArgumentException("All spanning vectors must have same dimension.");
			}
		}

		// 1) Orthonormalize spanning set -> qW (rank-revealing)
		List<Vector> qW = modifiedGramSchmidtOrthonormalize(spanningSetForW, tol);

		// 2) Build orthonormal basis for W^\perp by orthogonalizing standard basis e_i
		List<Vector> qPerp = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			Vector v = standardBasis(n, i);

			// Orthogonalize against W
			orthogonalizeInPlace(v, qW);
			reorthogonalizeInPlace(v, qW); // second pass improves stability

			// Orthogonalize against already found perp vectors (to keep them orthonormal)
			orthogonalizeInPlace(v, qPerp);
			reorthogonalizeInPlace(v, qPerp);

			double norm = v.norm2();
			if (norm > tol) {
				scaleInPlace(v, 1.0 / norm);
				qPerp.add(v);
			}
		}

		// qPerp size should be n - rank(W) (within numerical tolerance)
		return new OrthogonalComplement(n, tol, qW, qPerp);
	}

	/** Ambient dimension dim(V). */
	public int ambientDimension() {
		return n;
	}

	/** dim(W) (numerical rank after orthonormalization). */
	public int dimensionW() {
		return qW.size();
	}

	/** dim(W^\perp). */
	public int dimensionPerp() {
		return qPerp.size();
	}

	/** Orthonormal basis of W. */
	public List<Vector> basisW() {
		return qW;
	}

	/** Orthonormal basis of W^\perp. */
	public List<Vector> basisPerp() {
		return qPerp;
	}

	/** Returns true iff x is orthogonal to all of W (within tolerance). */
	public boolean contains(Vector x) {
		requireDim(x);
		for (Vector q : qW) {
			if (Math.abs(q.dot(x)) > tol) return false;
		}
		return true;
	}

	/**
	 * Orthogonal projection of v onto W:
	 * yHat = Σ (q_i^T v) q_i
	 */
	public Vector projectOntoW(Vector v) {
		requireDim(v);
		Vector yHat = zeros(n);
		for (Vector q : qW) {
			double alpha = q.dot(v);
			axpyInPlace(yHat, alpha, q);
		}
		return yHat;
	}

	/**
	 * Orthogonal projection of v onto W^\perp:
	 * z = v - proj_W(v)
	 */
	public Vector projectOntoPerp(Vector v) {
		requireDim(v);
		Vector yHat = projectOntoW(v);
		Vector z = v.copy();
		axpyInPlace(z, -1.0, yHat);
		return z;
	}

	/**
	 * Unique decomposition v = yHat + z with yHat ∈ W and z ∈ W^\perp.
	 */
	public Decomposition decompose(Vector v) {
		requireDim(v);
		Vector yHat = projectOntoW(v);
		Vector z = v.copy();
		axpyInPlace(z, -1.0, yHat);
		return new Decomposition(yHat, z);
	}

	public double tolerance() {
		return tol;
	}

	// -------------------------
	// Helpers / numerics
	// -------------------------

	public static final class Decomposition {
		public final Vector yHatInW;
		public final Vector zInPerp;

		public Decomposition(Vector yHatInW, Vector zInPerp) {
			this.yHatInW = yHatInW;
			this.zInPerp = zInPerp;
		}
	}

	private void requireDim(Vector v) {
		if (v == null) throw new NullPointerException("vector is null");
		if (v.dimension() != n) {
			throw new IllegalArgumentException("Expected dimension " + n + " but got " + v.dimension());
		}
	}

	/**
	 * Modified Gram-Schmidt with a second pass (reorth) and rank revealing drop.
	 * Returns an orthonormal list.
	 */
	private static List<Vector> modifiedGramSchmidtOrthonormalize(List<Vector> vecs, double tol) {
		List<Vector> Q = new ArrayList<>();
		for (Vector a : vecs) {
			Vector v = a.copy();

			// first pass
			orthogonalizeInPlace(v, Q);
			// second pass improves numerical orthogonality
			reorthogonalizeInPlace(v, Q);

			double norm = v.norm2();
			if (norm > tol) {
				scaleInPlace(v, 1.0 / norm);
				Q.add(v);
			}
		}
		return Q;
	}

	private static void orthogonalizeInPlace(Vector v, List<Vector> Q) {
		for (Vector q : Q) {
			double alpha = q.dot(v);
			axpyInPlace(v, -alpha, q); // v -= alpha q
		}
	}

	private static void reorthogonalizeInPlace(Vector v, List<Vector> Q) {
		// second pass for better numerical stability
		for (Vector q : Q) {
			double alpha = q.dot(v);
			axpyInPlace(v, -alpha, q);
		}
	}

	private static void axpyInPlace(Vector y, double a, Vector x) {
		// y += a*x
		int n = y.dimension();
		for (int i = 0; i < n; i++) {
			y.set(i, y.get(i) + a * x.get(i));
		}
	}

	private static void scaleInPlace(Vector v, double a) {
		int n = v.dimension();
		for (int i = 0; i < n; i++) {
			v.set(i, a * v.get(i));
		}
	}

	private static Vector standardBasis(int n, int i) {
		Vector e = zeros(n);
		e.set(i, 1.0);
		return e;
	}

	/**
	 * Creates a zero vector of dimension n.
	 */
	private static Vector zeros(int n) {
		return new Vector(new double[n]);
	}
}