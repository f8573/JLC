package net.faulj.test;

import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.matrix.Matrix;
import net.faulj.visualizer.MatrixLatexExporter;

import java.io.File;

public class Test {

    public static void main(String[] args) {
        Matrix m = Matrix.randomMatrix(6,6);
        Matrix[] matrices = ExplicitQRIteration.decompose(m);

        // Print human readable form
        for (int i = 0; i < matrices.length; i++) {
            System.out.println("--- Matrix " + i + " ---");
            System.out.println(matrices[i]);
        }

        // Create LaTeX document and write to build/matrix_output.tex
        String[] names = new String[matrices.length];
        for (int i = 0; i < names.length; i++) names[i] = "Matrix_" + i;
        String doc = MatrixLatexExporter.matricesToLatexDocument(matrices, names);
        File out = new File("build/matrix_output.tex");
        try {
            MatrixLatexExporter.writeLatexFile(out, doc);
            System.out.println("Wrote LaTeX document to " + out.getAbsolutePath());
        } catch (Exception e) {
            System.err.println("Failed to write LaTeX output: " + e.getMessage());
        }
    }
}
