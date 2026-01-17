package net.faulj.web;

import net.faulj.matrix.Matrix;
import net.faulj.visualizer.MatrixLatexExporter;
import net.faulj.eigen.qr.ExplicitQRIteration;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.CrossOrigin;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
public class ApiController {

    @GetMapping("/api/ping")
    public Map<String, String> ping() {
        return Map.of("message", "pong from Java backend", "version", "1.0");
    }

    @GetMapping("/api/latex")
    public Map<String, Object> latex() {
        // generate a small random matrix and decompose it
        Matrix m = Matrix.randomMatrix(4, 4);
        Matrix[] mats = ExplicitQRIteration.decompose(m);

        List<String> schurLatex = new ArrayList<>();
        for (Matrix mat : mats) {
            schurLatex.add(MatrixLatexExporter.matrixToLatex(mat));
        }

        // Original matrix latex
        String originalLatex = MatrixLatexExporter.matrixToLatex(m);

        // Eigenvalues
        List<String> eigenStr = new ArrayList<>();
        for (net.faulj.scalar.Complex c : ExplicitQRIteration.getEigenvalues(m.copy())) {
            eigenStr.add(c.toString());
        }

        return Map.of(
                "original", originalLatex,
                "schur", schurLatex,
                "eigenvalues", eigenStr
        );
    }

}
