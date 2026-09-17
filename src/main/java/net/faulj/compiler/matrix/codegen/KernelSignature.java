package net.faulj.compiler.matrix.codegen;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Comparator;
import java.util.Objects;

import net.faulj.compiler.matrix.kernel.KernelAccess;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelLoop;
import net.faulj.compiler.matrix.kernel.KernelOp;
import net.faulj.compiler.matrix.kernel.KernelOpcode;
import net.faulj.compiler.matrix.kernel.KernelVerifier;
import net.faulj.compiler.matrix.affine.AffineExpr;

/**
 * Deterministic, shape-specialized identity for one verified R3 function.
 *
 * <p>The canonical text is intentionally inspectable. The digest is only used
 * for safe generated symbols; no user expression text is ever copied into a
 * C++ identifier.</p>
 */
public final class KernelSignature {
    private final String canonicalText;
    private final String sha256;
    private final int hashCode;

    private KernelSignature(String canonicalText) {
        this.canonicalText = Objects.requireNonNull(canonicalText, "Signature text");
        this.sha256 = sha256(canonicalText);
        this.hashCode = canonicalText.hashCode();
    }

    public static KernelSignature from(KernelFunction function) {
        if (function == null) {
            throw new IllegalArgumentException("Kernel function must not be null");
        }
        KernelVerifier.requireValid(function);
        StringBuilder text = new StringBuilder();
        text.append("jlc-kernel-signature-v1\n");
        text.append("shape-specialized=true\n");
        text.append("value-type=FP64\n");
        text.append("function-id=").append(function.id()).append('\n');
        text.append("leaf-count=").append(function.inputBuffers().size()).append('\n');
        text.append("output-count=").append(function.outputBuffers().size()).append('\n');
        text.append("loops=");
        for (KernelLoop loop : function.loops()) {
            text.append(loop.inductionVariable().name()).append(':')
                .append(loop.lowerBound()).append("..")
                .append(loop.upperBound()).append(':').append(loop.step()).append(';');
        }
        text.append('\n');
        text.append("buffers=\n");
        for (KernelBuffer buffer : function.buffers()) {
            text.append(buffer.id()).append('|').append(buffer.role()).append('|')
                .append(buffer.shape()).append('|').append(buffer.storageType()).append('|')
                .append(buffer.computeType()).append('|').append(buffer.accumulatorType())
                .append('|').append(buffer.layout()).append('|').append(buffer.memorySpace())
                .append('|').append(buffer.ownership()).append('\n');
        }
        text.append("aliases=\n");
        var aliases = function.aliasFacts().stream()
            .sorted(Comparator
                .comparingInt((net.faulj.compiler.matrix.kernel.KernelAliasFact fact)
                    -> Math.min(fact.first().id(), fact.second().id()))
                .thenComparingInt(fact -> Math.max(fact.first().id(), fact.second().id()))
                .thenComparing(fact -> fact.relation().name()))
            .toList();
        for (var fact : aliases) {
            int first = Math.min(fact.first().id(), fact.second().id());
            int second = Math.max(fact.first().id(), fact.second().id());
            text.append(first).append('<').append('>').append(second).append('=')
                .append(fact.relation()).append('\n');
        }
        text.append("ops=\n");
        for (KernelOp operation : function.body().operations()) {
            text.append(operation.opcode()).append('|')
                .append(operation.resultValueId()).append('|')
                .append(operation.operands() == null ? "-" : operation.operands()).append('|');
            if (operation.access() != null) {
                text.append(accessText(operation.access()));
            } else {
                text.append('-');
            }
            text.append('|');
            if (operation.opcode() == KernelOpcode.CONSTANT) {
                text.append(Double.doubleToRawLongBits(operation.immediate()));
            } else {
                text.append('-');
            }
            text.append('\n');
        }
        return new KernelSignature(text.toString());
    }

    /**
     * Reconstitute a semantic identity from its persisted canonical text.
     * The text is validated and hashed exactly as it was when emitted; it is
     * never interpreted as an execution/backend decision.
     */
    public static KernelSignature fromCanonicalText(String canonicalText) {
        if (canonicalText == null || canonicalText.isBlank()) {
            throw new IllegalArgumentException("Kernel signature text is empty");
        }
        if (!canonicalText.startsWith("jlc-kernel-signature-v1\n")) {
            throw new IllegalArgumentException("Unsupported kernel signature version");
        }
        return new KernelSignature(canonicalText);
    }

    /** Alias for callers that use parser terminology. */
    public static KernelSignature parse(String canonicalText) {
        return fromCanonicalText(canonicalText);
    }

    public String canonicalText() {
        return canonicalText;
    }

    public String text() {
        return canonicalText;
    }

    public String sha256() {
        return sha256;
    }

    public String shortHash() {
        return sha256.substring(0, 12);
    }

    public String generatedSymbol() {
        return "jlc_pk_" + shortHash();
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof KernelSignature signature
            && canonicalText.equals(signature.canonicalText);
    }

    @Override
    public int hashCode() {
        return hashCode;
    }

    @Override
    public String toString() {
        return canonicalText;
    }

    private static String accessText(KernelAccess access) {
        StringBuilder result = new StringBuilder("buffer=")
            .append(access.buffer().id()).append('[');
        for (int index = 0; index < access.indices().size(); index++) {
            if (index > 0) {
                result.append(',');
            }
            result.append(expressionText(access.index(index)));
        }
        return result.append(']').toString();
    }

    private static String expressionText(AffineExpr expression) {
        StringBuilder result = new StringBuilder(Long.toString(expression.constant()));
        for (var term : expression.coefficients().entrySet()) {
            result.append(';').append(term.getKey().name()).append('=')
                .append(term.getValue());
        }
        return result.toString();
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                .digest(value.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("JDK must provide SHA-256", impossible);
        }
    }
}
