package net.faulj.solve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

public class SolverTest {

    private static final double TOL = 1e-9;
    private static final double TOL_LEAST_SQUARES = 1e-7;

    // ========== LUSolver Tests ==========

    @Test
    public void testSimpleSystem() {
        // 2x + y = 5
        // x + 3y = 7
        Matrix A = new Matrix(new double[][]{
                {2, 1},
                {1, 3}
        });

        Vector b = new Vector(new double[]{5, 7});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Expected solution: x = 1.6, y = 1.8
        assertEquals(1.6, x.get(0), TOL);
        assertEquals(1.8, x.get(1), TOL);
    }

    @Test
    public void testVerifySolutionSatisfiesEquation() {
        Matrix A = new Matrix(new double[][]{
                {3, 2, 1},
                {1, 4, 2},
                {2, 1, 5}
        });

        Vector b = new Vector(new double[]{10, 12, 15});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify Ax = b
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals("Component " + i + " should match",
                    b.get(i), Ax.get(i), TOL);
        }
    }

    @Test
    public void testUpperTriangularSystem() {
        Matrix A = new Matrix(new double[][]{
                {2, 1, 3},
                {0, 5, 2},
                {0, 0, 4}
        });

        Vector b = new Vector(new double[]{14, 17, 12});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify solution
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals(b.get(i), Ax.get(i), TOL);
        }
    }

    @Test
    public void testLowerTriangularSystem() {
        Matrix A = new Matrix(new double[][]{
                {3, 0, 0},
                {2, 4, 0},
                {1, 2, 5}
        });

        Vector b = new Vector(new double[]{6, 14, 20});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify solution
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals(b.get(i), Ax.get(i), TOL);
        }
    }

    @Test
    public void testDiagonalSystem() {
        Matrix A = new Matrix(new double[][]{
                {2, 0, 0},
                {0, 3, 0},
                {0, 0, 5}
        });

        Vector b = new Vector(new double[]{6, 9, 15});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Expected: x = [3, 3, 3]
        assertEquals(3.0, x.get(0), TOL);
        assertEquals(3.0, x.get(1), TOL);
        assertEquals(3.0, x.get(2), TOL);
    }

    @Test
    public void testIdentitySystem() {
        Matrix I = Matrix.Identity(3);
        Vector b = new Vector(new double[]{5, 7, 9});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(I, b);

        // For identity matrix, x = b
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals(b.get(i), x.get(i), TOL);
        }
    }

    @Test
    public void testLargerSystem() {
        Matrix A = new Matrix(new double[][]{
                {5, 2, 1, 3},
                {1, 4, 2, 1},
                {2, 1, 6, 2},
                {3, 1, 2, 5}
        });

        Vector b = new Vector(new double[]{20, 15, 25, 18});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify Ax = b
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals(b.get(i), Ax.get(i), TOL);
        }
    }

    @Test
    public void testMultipleSolutionsWithSameMatrix() {
        Matrix A = new Matrix(new double[][]{
                {4, 3, 2},
                {2, 5, 1},
                {1, 2, 3}
        });

        LUSolver solver = new LUSolver();

        // Solve for multiple right-hand sides
        Vector b1 = new Vector(new double[]{9, 8, 6});
        Vector x1 = solver.solve(A, b1);
        Vector Ax1 = A.multiply(x1.toMatrix()).getData()[0];
        for (int i = 0; i < b1.dimension(); i++) {
            assertEquals(b1.get(i), Ax1.get(i), TOL);
        }

        Vector b2 = new Vector(new double[]{15, 12, 10});
        Vector x2 = solver.solve(A, b2);
        Vector Ax2 = A.multiply(x2.toMatrix()).getData()[0];
        for (int i = 0; i < b2.dimension(); i++) {
            assertEquals(b2.get(i), Ax2.get(i), TOL);
        }
    }

    @Test
    public void testRandomSystem() {
        Matrix A = Matrix.randomMatrix(5, 5);
        Vector b = new Vector(new double[]{1, 2, 3, 4, 5});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify solution
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals(b.get(i), Ax.get(i), TOL);
        }
    }

    @Test
    public void testSymmetricSystem() {
        Matrix A = new Matrix(new double[][]{
                {4, 1, 2},
                {1, 3, 1},
                {2, 1, 5}
        });

        Vector b = new Vector(new double[]{7, 5, 8});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify solution
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals(b.get(i), Ax.get(i), TOL);
        }
    }

    @Test
    public void testWithNegativeCoefficients() {
        Matrix A = new Matrix(new double[][]{
                {-2, 1, 3},
                {1, -4, 2},
                {3, 2, -5}
        });

        Vector b = new Vector(new double[]{5, -7, 3});

        LUSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify solution
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        for (int i = 0; i < b.dimension(); i++) {
            assertEquals(b.get(i), Ax.get(i), TOL);
        }
    }

    // ========== LinearSolver Interface Tests ==========

    @Test
    public void testLUSolverInterface() {
        Matrix A = new Matrix(new double[][]{
                {3, 2},
                {1, 4}
        });

        Vector b = new Vector(new double[]{8, 9});

        LinearSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify Ax = b
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        assertEquals(b.get(0), Ax.get(0), TOL);
        assertEquals(b.get(1), Ax.get(1), TOL);
    }

    @Test
    public void testLeastSquaresSolverInterface() {
        // Overdetermined system
        Matrix A = new Matrix(new double[][]{
                {1, 1},
                {1, 2},
                {1, 3}
        });

        Vector b = new Vector(new double[]{2, 3, 4});

        LinearSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        assertNotNull(x);
        assertEquals(2, x.dimension());
    }

    @Test
    public void testComparisonBetweenSolvers() {
        // Small system that all solvers can handle
        Matrix A = new Matrix(new double[][]{
                {4, 1},
                {2, 3}
        });

        Vector b = new Vector(new double[]{9, 8});

        LUSolver luSolver = new LUSolver();
        CramerSolver cramerSolver = new CramerSolver();

        Vector xLU = luSolver.solve(A, b);
        Vector xCramer = cramerSolver.solve(A, b);

        // Both should give same result
        for (int i = 0; i < xLU.dimension(); i++) {
            assertEquals("Solutions should match at index " + i,
                    xLU.get(i), xCramer.get(i), TOL);
        }
    }

    @Test
    public void testPolymorphicUsage() {
        Matrix A = new Matrix(new double[][]{
                {5, 2, 1},
                {1, 4, 2},
                {2, 1, 6}
        });

        Vector b = new Vector(new double[]{10, 11, 13});

        // Test that all solvers implementing LinearSolver work
        LinearSolver[] solvers = {
                new LUSolver(),
                new CramerSolver()
        };

        Vector expectedResult = null;

        for (LinearSolver solver : solvers) {
            Vector x = solver.solve(A, b);

            if (expectedResult == null) {
                expectedResult = x;
            } else {
                // All solvers should produce same result
                for (int i = 0; i < x.dimension(); i++) {
                    assertEquals("All solvers should give same result",
                            expectedResult.get(i), x.get(i), TOL);
                }
            }
        }
    }

    @Test
    public void testDifferentSystemSizes() {
        // Test with different sized systems
        int[] sizes = {2, 3, 4};

        for (int n : sizes) {
            Matrix A = Matrix.Identity(n).multiplyScalar(2.0);
            double[] bData = new double[n];
            for (int i = 0; i < n; i++) {
                bData[i] = (i + 1) * 2.0;
            }
            Vector b = new Vector(bData);

            LinearSolver solver = new LUSolver();
            Vector x = solver.solve(A, b);

            // For 2I * x = b, solution is x = b/2
            for (int i = 0; i < n; i++) {
                assertEquals((i + 1), x.get(i), TOL);
            }
        }
    }

    @Test
    public void testConsistencyCheck() {
        Matrix A = new Matrix(new double[][]{
                {3, 1, 2},
                {1, 4, 1},
                {2, 1, 5}
        });

        Vector b = new Vector(new double[]{8, 10, 12});

        LinearSolver solver = new LUSolver();
        Vector x = solver.solve(A, b);

        // Verify solution satisfies the original equation
        Vector Ax = A.multiply(x.toMatrix()).getData()[0];
        Vector residual = Ax.subtract(b);
        double residualNorm = residual.norm2();

        assertTrue("Residual should be very small: " + residualNorm,
                residualNorm < TOL);
    }

    // ========== LeastSquaresSolver Tests ==========

    @Test
    public void testExactSolutionForSquareSystem() {
        // When system has exact solution, least squares should find it
        Matrix A = new Matrix(new double[][]{
                {2, 1},
                {1, 3}
        });

        Vector b = new Vector(new double[]{5, 7});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        // Verify Ax = b
        Matrix xMat = x.toMatrix();
        Vector Ax = A.multiply(xMat).getData()[0];
        assertEquals(b.get(0), Ax.get(0), TOL_LEAST_SQUARES);
        assertEquals(b.get(1), Ax.get(1), TOL_LEAST_SQUARES);
    }

    @Test
    public void testOverdeterminedSystem() {
        // More equations than unknowns
        Matrix A = new Matrix(new double[][]{
                {1, 1},
                {1, 2},
                {1, 3},
                {1, 4}
        });

        Vector b = new Vector(new double[]{2, 3, 4, 5});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        assertNotNull(x);
        assertEquals(2, x.dimension());

        // Verify residual is minimized (approximate fit)
        Matrix xMat = x.toMatrix();
        Vector Ax = A.multiply(xMat).getData()[0];
        Vector residual = Ax.subtract(b);
        double residualNorm = residual.norm2();

        // Residual should exist but be reasonable for a line fit
        assertTrue("Residual norm should be reasonable: " + residualNorm,
                residualNorm < 1.0);
    }

    @Test
    public void testLinearRegression() {
        // Fit y = mx + c to points: (0,1), (1,3), (2,4), (3,6)
        Matrix A = new Matrix(new double[][]{
                {0, 1},
                {1, 1},
                {2, 1},
                {3, 1}
        });

        Vector b = new Vector(new double[]{1, 3, 4, 6});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        // x[0] should be close to slope (approximately 5/3)
        // x[1] should be close to intercept (approximately 4/3)
        assertTrue("Solution should have 2 components", x.dimension() == 2);

        // Verify the fit is reasonable
        double slope = x.get(0);
        double intercept = x.get(1);

        assertTrue("Slope should be positive and reasonable", slope > 1.0 && slope < 2.0);
        assertTrue("Intercept should be positive and reasonable", intercept > 0.5 && intercept < 2.0);
    }

    @Test
    public void testTallMatrix() {
        // Many more equations than unknowns
        Matrix A = new Matrix(new double[][]{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
                {1, 1, 0},
                {1, 0, 1},
                {0, 1, 1}
        });

        Vector b = new Vector(new double[]{1, 2, 3, 3, 4, 5});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        assertNotNull(x);
        assertEquals(3, x.dimension());

        // Verify residual exists but solution is close
        Matrix xMat = x.toMatrix();
        Vector Ax = A.multiply(xMat).getData()[0];
        Vector residual = Ax.subtract(b);
        assertTrue("Residual should be reasonably small", residual.norm2() < 2.0);
    }

    @Test
    public void testNormalEquationsProperty() {
        // For least squares, A^T * A * x = A^T * b
        Matrix A = new Matrix(new double[][]{
                {2, 1},
                {3, 2},
                {4, 1}
        });

        Vector b = new Vector(new double[]{3, 5, 6});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        // Verify normal equations: A^T * A * x = A^T * b
        Matrix AtA = A.transpose().multiply(A);
        Vector Atb = A.transpose().multiply(b.toMatrix()).getData()[0];
        Vector AtAx = AtA.multiply(x.toMatrix()).getData()[0];

        for (int i = 0; i < x.dimension(); i++) {
            assertEquals("Normal equation component " + i,
                    Atb.get(i), AtAx.get(i), TOL_LEAST_SQUARES);
        }
    }

    @Test
    public void testQuadraticFit() {
        // Fit y = a*x^2 + b*x + c to data
        // Points: (0,1), (1,2), (2,5), (3,10)
        Matrix A = new Matrix(new double[][]{
                {0, 0, 1},  // x^2, x, 1 for x=0
                {1, 1, 1},  // x^2, x, 1 for x=1
                {4, 2, 1},  // x^2, x, 1 for x=2
                {9, 3, 1}   // x^2, x, 1 for x=3
        });

        Vector b = new Vector(new double[]{1, 2, 5, 10});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        assertEquals(3, x.dimension());

        // Verify the fit is reasonable (should be close to y = x^2 + 1)
        double a = x.get(0);  // coefficient of x^2
        double coeff_b = x.get(1);  // coefficient of x
        double c = x.get(2);  // constant

        assertTrue("Quadratic coefficient should be close to 1", Math.abs(a - 1.0) < 0.1);
        assertTrue("Linear coefficient should be close to 0", Math.abs(coeff_b) < 0.1);
        assertTrue("Constant should be close to 1", Math.abs(c - 1.0) < 0.1);
    }

    @Test
    public void testMinimizesResidualNorm() {
        Matrix A = new Matrix(new double[][]{
                {1, 2},
                {2, 3},
                {3, 4},
                {4, 5}
        });

        Vector b = new Vector(new double[]{1, 2, 3, 4});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        // Compute residual
        Matrix xMat = x.toMatrix();
        Vector Ax = A.multiply(xMat).getData()[0];
        Vector residual = Ax.subtract(b);
        double residualNorm = residual.norm2();

        // The residual should be small for this well-conditioned problem
        assertTrue("Residual norm should be small: " + residualNorm,
                residualNorm < 0.5);
    }

    @Test
    public void testIdentityMatrixCase() {
        // Tall identity-like matrix
        Matrix A = new Matrix(new double[][]{
                {1, 0},
                {0, 1},
                {0, 0}
        });

        Vector b = new Vector(new double[]{3, 4, 0});

        LeastSquaresSolver solver = new LeastSquaresSolver();
        Vector x = solver.solve(A, b);

        // Should give x = [3, 4]
        assertEquals(3.0, x.get(0), TOL_LEAST_SQUARES);
        assertEquals(4.0, x.get(1), TOL_LEAST_SQUARES);
    }
}
