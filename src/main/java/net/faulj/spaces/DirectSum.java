package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Models the direct sum decomposition of vector spaces.
 * <p>
 * A vector space \( V \) is the direct sum of subspaces \( U \) and \( W \), written
 * \( V = U \oplus W \), if every vector \( v \in V \) can be written uniquely as
 * \( v = u + w \) where \( u \in U \) and \( w \in W \).
 * </p>
 *
 * <h2>Conditions:</h2>
 * <ul>
 * <li>\( V = U + W \) (Sum of subspaces covers V)</li>
 * <li>\( U \cap W = \{0\} \) (Intersection is trivial)</li>
 * </ul>
 *
 * <h2>Matrix Representation:</h2>
 * <p>
 * If \( B_U \) is a basis for \( U \) and \( B_W \) is a basis for \( W \), then
 * \( B_U \cup B_W \) is a basis for \( V \). The concatenation of basis matrices
 * often yields a square, invertible matrix.
 * </p>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Decomposing spaces into eigenspaces.</li>
 * <li>Separating noise (kernel) from signal (row space).</li>
 * <li>Projection operators (projecting onto U along W).</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see Projection
 * @see OrthogonalComplement
 */
public final class DirectSum {

	private final Matrix BU;          // n x k
	private final Matrix BW;          // n x m
	private final Matrix BV;          // n x (k+m) = [BU BW]

	private final int n;
	private final int dimU;
	private final int dimW;
	private final int dimV;

	private final double tol;

	// Only available if BV is square and invertible (k+m == n and rank == n)
	private final boolean squareInvertible;
	private final Matrix BVinv;

	/**
	 * Constructs a DirectSum from bases for U and W with a default tolerance.
	 * @param basisU Basis matrix for subspace U (columns are basis vectors)
	 * @param basisW Basis matrix for subspace W (columns are basis vectors)
	 */
	public DirectSum(Matrix basisU, Matrix basisW) {
		this(basisU, basisW, 1e-12);
	}

	/**
	 * Constructs a DirectSum from bases for U and W with a specified tolerance.
	 * @param basisU Basis matrix for subspace U
	 * @param basisW Basis matrix for subspace W
	 * @param tolerance Numerical tolerance for rank and equality checks
	 * @throws IllegalArgumentException if dimensions mismatch or if U + W is not a direct sum
	 */
	public DirectSum(Matrix basisU, Matrix basisW, double tolerance) {
		if (basisU == null || basisW == null) {
			throw new IllegalArgumentException("basisU and basisW must be non-null.");
		}
		if (tolerance <= 0.0) {
			throw new IllegalArgumentException("tolerance must be > 0.");
		}

		this.tol = tolerance;

		this.n = basisU.getRowCount();
		if (basisW.getRowCount() != n) {
			throw new IllegalArgumentException("basisU and basisW must have the same ambient dimension (row count).");
		}

		this.dimU = basisU.getColumnCount();
		this.dimW = basisW.getColumnCount();
		if (dimU == 0 || dimW == 0) {
			throw new IllegalArgumentException("basisU and basisW must each have at least 1 column.");
		}

		this.BU = basisU.copy();
		this.BW = basisW.copy();

		// Use existing Matrix functionality to concat
		this.BV = this.BU.AppendMatrix(this.BW, "RIGHT");

		int r = rank(this.BV, this.tol);
		this.dimV = r;

		// Direct sum check: concatenated basis columns must be independent.
		if (r != dimU + dimW) {
			int intersectionDimEstimate = (dimU + dimW) - r;
			throw new IllegalArgumentException(
					"Not a direct sum with the provided bases: columns of [BU BW] are linearly dependent. " +
							"Estimated dim(U ∩ W) >= " + intersectionDimEstimate + ". " +
							"Fix by providing independent bases for U and W (or re-basis them)."
			);
		}

		this.squareInvertible = ((dimU + dimW) == n) && (r == n);
		if (squareInvertible) {
			this.BVinv = inverse(this.BV, this.tol);
		} else {
			this.BVinv = null;
		}
	}

	// -------------------------
	// Basic getters
	// -------------------------

	public int ambientDimension() { return n; }

	public int dimU() { return dimU; }

	public int dimW() { return dimW; }

	/** dim(V) where V := U + W */
	public int dimV() { return dimV; }

	public Matrix basisU() { return BU.copy(); }

	public Matrix basisW() { return BW.copy(); }

	/** Returns [BU BW] */
	public Matrix basisV() { return BV.copy(); }

	/** True iff [BU BW] is square and invertible (so V = R^n and projections are global linear maps). */
	public boolean isSquareInvertible() { return squareInvertible; }

	// -------------------------
	// Core operations
	// -------------------------

	/**
	 * Decompose v (n×1) into v = u + w with u in U, w in W.
	 * Unique if v lies in V := U + W. Throws if v is not in V.
	 */
	public Decomposition decompose(Vector v) {
		if (v == null) throw new IllegalArgumentException("Vector must not be null");
		if (v.dimension() != n) {
			throw new IllegalArgumentException("Vector dimension " + v.dimension() + " does not match ambient dimension " + n);
		}

		double[] b = v.getData(); // raw access for internal solver

		// Solve BV * y = b where y = [a; c], a in R^{dimU}, c in R^{dimW}
		double[] y;
		if (squareInvertible) {
			y = matVec(BVinv, b);
		} else {
			// Full column rank least-squares solve (unique because BV has independent columns):
			// y = argmin ||BV y - b||, and we reject if residual is too large (i.e., b not in span(BV)).
			y = solveNormalEquationsFullColumnRank(BV, b, tol);
		}

		double[] a = new double[dimU];
		double[] c = new double[dimW];
		System.arraycopy(y, 0, a, 0, dimU);
		System.arraycopy(y, dimU, c, 0, dimW);

		// Compute u = BU * a
		Vector u = matVecToVector(BU, a);
		// Compute w = BW * c
		Vector w = matVecToVector(BW, c);

		// Validate membership: ||(u+w)-v|| small
		Vector sum = u.add(w);
		double res = sum.subtract(v).norm2();

		double scale = 1.0 + v.norm2();
		if (res > tol * scale) {
			throw new IllegalArgumentException(
					"Vector is not in V = U + W (within tolerance). Residual=" + res + ", tol*scale=" + (tol * scale)
			);
		}

		return new Decomposition(u, w, new Vector(a), new Vector(c), res);
	}

	/** Projection onto U along W (defined on V = U ⊕ W). */
	public Vector projectOntoUAlongW(Vector v) {
		return decompose(v).u();
	}

	/** Projection onto W along U (defined on V = U ⊕ W). */
	public Vector projectOntoWAlongU(Vector v) {
		return decompose(v).w();
	}

	/**
	 * Returns the projection matrix P_U that maps v -> u (projection onto U along W),
	 * BUT only if BV is square/invertible (so it is a global linear map on R^n).
	 *
	 * Formula when BV is n×n invertible:
	 * coords = BV^{-1} v
	 * take first dimU coords via E_U = [I_dimU  0] (dimU×n)
	 * u = BU * (E_U * coords)
	 * => P_U = BU * E_U * BV^{-1}
	 */
	public Matrix projectionMatrixOntoUAlongW() {
		requireSquareInvertible("projectionMatrixOntoUAlongW");
		Matrix EU = selectorFirst(dimU, n); // dimU x n
		return BU.multiply(EU).multiply(BVinv);
	}

	/** Analogous projection matrix onto W along U: P_W = BW * E_W * BV^{-1}. */
	public Matrix projectionMatrixOntoWAlongU() {
		requireSquareInvertible("projectionMatrixOntoWAlongU");
		Matrix EW = selectorLast(dimW, n, dimU); // dimW x n, selects coords dimU..n-1
		return BW.multiply(EW).multiply(BVinv);
	}

	// -------------------------
	// Result type
	// -------------------------

	public static final class Decomposition {
		private final Vector u;       // n x 1
		private final Vector w;       // n x 1
		private final Vector coeffU;  // dimU x 1
		private final Vector coeffW;  // dimW x 1
		private final double residual;

		private Decomposition(Vector u, Vector w, Vector coeffU, Vector coeffW, double residual) {
			this.u = u;
			this.w = w;
			this.coeffU = coeffU;
			this.coeffW = coeffW;
			this.residual = residual;
		}

		public Vector u() { return u.copy(); }

		public Vector w() { return w.copy(); }

		public Vector coefficientsU() { return coeffU.copy(); }

		public Vector coefficientsW() { return coeffW.copy(); }

		public double residual() { return residual; }
	}

	// -------------------------
	// Internal helpers (Matrix ops)
	// -------------------------

	private void requireSquareInvertible(String method) {
		if (!squareInvertible) {
			throw new UnsupportedOperationException(
					method + " requires [BU BW] to be square & invertible (dimU+dimW == n and rank == n). " +
							"Otherwise the projection is only defined on V = span([BU BW]), not as a global n×n matrix."
			);
		}
	}

	/** Build selector E_U = [I_k 0] of size k×n */
	private static Matrix selectorFirst(int k, int n) {
		// Create k x n matrix
		// We can use Matrix.Identity(k) then append Matrix.zero(k, n-k)
		if (n == k) return Matrix.Identity(k);
		return Matrix.Identity(k).AppendMatrix(Matrix.zero(k, n - k), "RIGHT");
	}

	/** Build selector E_W that selects the last m coordinates starting at offset */
	private static Matrix selectorLast(int m, int n, int offset) {
		// [0 I_m] of size m x n
		if (offset == 0 && m == n) return Matrix.Identity(m);
		// Concatenate Zero(m, offset) and Identity(m)
		Matrix Z = Matrix.zero(m, offset);
		Matrix I = Matrix.Identity(m);
		// If there are more cols after (unlikely for [BU BW] context but for completeness)
		// In DirectSum context, offset + m == n usually
		return Z.AppendMatrix(I, "RIGHT");
	}

	private static Vector matVecToVector(Matrix A, double[] x) {
		return new Vector(matVec(A, x));
	}

	private static double[] matVec(Matrix A, double[] x) {
		int r = A.getRowCount();
		int c = A.getColumnCount();
		if (x.length != c) throw new IllegalArgumentException("Shape mismatch for matVec.");
		double[] y = new double[r];
		for (int i = 0; i < r; i++) {
			double sum = 0.0;
			for (int j = 0; j < c; j++) sum += A.get(i, j) * x[j];
			y[i] = sum;
		}
		return y;
	}

	// -------------------------
	// Linear algebra primitives (rank / inverse / solve)
	// -------------------------

	/**
	 * Rank via Gaussian elimination with partial pivoting (on a copied array).
	 */
	private static int rank(Matrix A, double tol) {
		double[][] M = toArray(A);
		int m = M.length;
		int n = M[0].length;

		int r = 0;
		int row = 0;

		for (int col = 0; col < n && row < m; col++) {
			int piv = row;
			double best = Math.abs(M[piv][col]);
			for (int i = row + 1; i < m; i++) {
				double v = Math.abs(M[i][col]);
				if (v > best) { best = v; piv = i; }
			}
			if (best <= tol) continue;

			swapRows(M, row, piv);

			double pivot = M[row][col];
			for (int i = row + 1; i < m; i++) {
				double f = M[i][col] / pivot;
				if (Math.abs(f) <= tol) continue;
				M[i][col] = 0.0;
				for (int j = col + 1; j < n; j++) {
					M[i][j] -= f * M[row][j];
				}
			}

			row++;
			r++;
		}

		return r;
	}

	/**
	 * Inverse via Gauss-Jordan with partial pivoting.
	 * Throws if singular under tol.
	 */
	private static Matrix inverse(Matrix A, double tol) {
		int n = A.getRowCount();
		if (A.getColumnCount() != n) throw new IllegalArgumentException("inverse requires a square matrix.");

		double[][] M = new double[n][2 * n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) M[i][j] = A.get(i, j);
			M[i][n + i] = 1.0;
		}

		for (int col = 0; col < n; col++) {
			int piv = col;
			double best = Math.abs(M[piv][col]);
			for (int i = col + 1; i < n; i++) {
				double v = Math.abs(M[i][col]);
				if (v > best) { best = v; piv = i; }
			}
			if (best <= tol) {
				throw new IllegalArgumentException("Matrix is singular (or near-singular) under tolerance: pivot=" + best);
			}
			swapRows(M, col, piv);

			double pivot = M[col][col];
			for (int j = 0; j < 2 * n; j++) M[col][j] /= pivot;

			for (int i = 0; i < n; i++) {
				if (i == col) continue;
				double f = M[i][col];
				if (Math.abs(f) <= tol) continue;
				M[i][col] = 0.0;
				for (int j = col + 1; j < 2 * n; j++) {
					M[i][j] -= f * M[col][j];
				}
			}
		}

		Vector[] cols = new Vector[n];
		for (int j = 0; j < n; j++) {
			double[] colData = new double[n];
			for (int i = 0; i < n; i++) {
				colData[i] = M[i][n + j];
			}
			cols[j] = new Vector(colData);
		}
		return new Matrix(cols);
	}

	/**
	 * Least-squares coefficients for BV y ≈ b when BV has full column rank.
	 * Uses normal equations: (BV^T BV) y = BV^T b, solved by Gaussian elimination.
	 *
	 * NOTE: Normal equations can be ill-conditioned; swap this for a QR solve when you have it.
	 */
	private static double[] solveNormalEquationsFullColumnRank(Matrix BV, double[] b, double tol) {
		int n = BV.getRowCount();
		int p = BV.getColumnCount();
		if (b.length != n) throw new IllegalArgumentException("b length mismatch.");

		double[][] G = new double[p][p];
		double[] rhs = new double[p];

		// G = BV^T BV, rhs = BV^T b
		for (int i = 0; i < p; i++) {
			for (int k = 0; k < n; k++) {
				double Bik = BV.get(k, i);
				rhs[i] += Bik * b[k];
			}
			for (int j = i; j < p; j++) {
				double sum = 0.0;
				for (int k = 0; k < n; k++) sum += BV.get(k, i) * BV.get(k, j);
				G[i][j] = sum;
				G[j][i] = sum;
			}
		}

		// Light Tikhonov regularization helps with near-dependence.
		double lambda = tol * tol;
		for (int i = 0; i < p; i++) G[i][i] += lambda;

		return solveLinearSystem(G, rhs, tol);
	}

	private static double[] solveLinearSystem(double[][] A, double[] b, double tol) {
		int n = A.length;
		if (A[0].length != n) throw new IllegalArgumentException("A must be square.");
		if (b.length != n) throw new IllegalArgumentException("b length mismatch.");

		double[][] M = new double[n][n + 1];
		for (int i = 0; i < n; i++) {
			System.arraycopy(A[i], 0, M[i], 0, n);
			M[i][n] = b[i];
		}

		// Forward elimination with partial pivoting
		for (int col = 0; col < n; col++) {
			int piv = col;
			double best = Math.abs(M[piv][col]);
			for (int i = col + 1; i < n; i++) {
				double v = Math.abs(M[i][col]);
				if (v > best) { best = v; piv = i; }
			}
			if (best <= tol) {
				throw new IllegalArgumentException("Singular/ill-conditioned system under tolerance. pivot=" + best);
			}
			swapRows(M, col, piv);

			double pivot = M[col][col];
			for (int i = col + 1; i < n; i++) {
				double f = M[i][col] / pivot;
				if (Math.abs(f) <= tol) continue;
				M[i][col] = 0.0;
				for (int j = col + 1; j <= n; j++) M[i][j] -= f * M[col][j];
			}
		}

		// Back substitution
		double[] x = new double[n];
		for (int i = n - 1; i >= 0; i--) {
			double sum = M[i][n];
			for (int j = i + 1; j < n; j++) sum -= M[i][j] * x[j];
			x[i] = sum / M[i][i];
		}
		return x;
	}

	private static double[][] toArray(Matrix A) {
		double[][] out = new double[A.getRowCount()][A.getColumnCount()];
		for (int i = 0; i < A.getRowCount(); i++) {
			for (int j = 0; j < A.getColumnCount(); j++) {
				out[i][j] = A.get(i, j);
			}
		}
		return out;
	}

	private static void swapRows(double[][] M, int i, int j) {
		if (i == j) return;
		double[] tmp = M[i];
		M[i] = M[j];
		M[j] = tmp;
	}
}