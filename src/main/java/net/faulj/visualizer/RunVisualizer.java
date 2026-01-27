package net.faulj.visualizer;

import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.matrix.Matrix;

import java.io.File;

/**
 * CLI entry point for generating a LaTeX report from QR iteration output.
 */
public class RunVisualizer {
    /**
     * Entry point for generating a LaTeX report of QR iteration output.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        Matrix m = Matrix.randomMatrix(6,6);
        Matrix[] matrices = ExplicitQRIteration.decompose(m);

        // write LaTeX document
        String[] names = new String[matrices.length];
        for (int i = 0; i < names.length; i++) names[i] = "Matrix_" + i;
        String doc = MatrixLatexExporter.matricesToLatexDocument(matrices, names);
        File out = new File("build/matrix_output.tex");
        try {
            MatrixLatexExporter.writeLatexFile(out, doc);
            System.out.println("Wrote LaTeX document to " + out.getAbsolutePath());
        } catch (Exception e) {
            System.err.println("Failed to write LaTeX output: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
