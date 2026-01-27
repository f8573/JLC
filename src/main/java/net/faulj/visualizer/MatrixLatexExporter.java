package net.faulj.visualizer;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

/**
 * Utility class for exporting matrices and vectors to LaTeX.
 */
public class MatrixLatexExporter {

    private static final double TOL = 1e-12;

    /**
     * Format a number for LaTeX output.
     *
     * @param d value to format
     * @return formatted string
     */
    private static String fmt(double d) {
        if (Double.isNaN(d)) return "NaN";
        if (Double.isInfinite(d)) return d > 0 ? "\\infty" : "-\\infty";
        long rounded = Math.round(d);
        if (Math.abs(d - rounded) < TOL) return Long.toString(rounded);
        DecimalFormat df = new DecimalFormat("0.######");
        return df.format(d);
    }

    /**
     * Convert a vector to a LaTeX bmatrix string.
     *
     * @param v vector to convert
     * @return LaTeX string
     */
    public static String vectorToLatex(Vector v) {
        StringBuilder sb = new StringBuilder();
        sb.append("\\begin{bmatrix}");
        for (int i = 0; i < v.dimension(); i++) {
            sb.append(fmt(v.get(i)));
            if (i < v.dimension() - 1) sb.append("\\\\");
        }
        sb.append("\\end{bmatrix}");
        return sb.toString();
    }

    /**
     * Convert a matrix to a LaTeX bmatrix string.
     *
     * @param m matrix to convert
     * @return LaTeX string
     */
    public static String matrixToLatex(Matrix m) {
        int rows = m.getRowCount();
        int cols = m.getColumnCount();
        StringBuilder sb = new StringBuilder();
        sb.append("\\begin{bmatrix}");
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                sb.append(fmt(m.get(r, c)));
                if (c < cols - 1) sb.append("&");
            }
            if (r < rows - 1) sb.append("\\\\");
        }
        sb.append("\\end{bmatrix}");
        return sb.toString();
    }

    /**
     * Build a full LaTeX document containing the provided matrices.
     *
     * @param mats matrices to render
     * @param names optional section titles
     * @return LaTeX document string
     */
    public static String matricesToLatexDocument(Matrix[] mats, String[] names) {
        StringBuilder sb = new StringBuilder();
        sb.append("\\documentclass{article}\n");
        sb.append("\\usepackage{amsmath}\n");
        sb.append("\\begin{document}\n");
        for (int i = 0; i < mats.length; i++) {
            String title = (names != null && i < names.length && names[i] != null) ? names[i] : "Matrix " + i;
            sb.append("\\section*{" + title + "}\n");
            sb.append("\\[\n");
            sb.append(matrixToLatex(mats[i]));
            sb.append("\n\\]\n");
        }
        sb.append("\\end{document}\n");
        return sb.toString();
    }

    /**
     * Write a LaTeX document to disk.
     *
     * @param out output file
     * @param content LaTeX content
     * @throws IOException if write fails
     */
    public static void writeLatexFile(File out, String content) throws IOException {
        File parent = out.getParentFile();
        if (parent != null && !parent.exists()) parent.mkdirs();
        try (FileWriter fw = new FileWriter(out)) {
            fw.write(content);
        }
    }
}
