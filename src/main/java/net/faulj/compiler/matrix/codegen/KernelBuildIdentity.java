package net.faulj.compiler.matrix.codegen;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.StringJoiner;

/** Build/codegen compatibility key for persisted generated-kernel decisions. */
public final class KernelBuildIdentity {
    public static final String CODEGEN_ABI_VERSION = "jlc-r5-codegen-abi-v1";
    public static final String STRICT_FLAGS = "-O2,-fno-fast-math,-ffp-contract=off";

    private final String jlcVersion;
    private final String gitSha;
    private final String kernelSignatureVersion;
    private final String variantSignatureVersion;
    private final String compilerIdentity;
    private final String codegenAbiVersion;
    private final String strictFlags;
    private final String key;

    public KernelBuildIdentity(String jlcVersion,
                               String gitSha,
                               String kernelSignatureVersion,
                               String variantSignatureVersion,
                               String compilerIdentity) {
        this(jlcVersion, gitSha, kernelSignatureVersion, variantSignatureVersion,
            compilerIdentity, CODEGEN_ABI_VERSION, STRICT_FLAGS);
    }

    public KernelBuildIdentity(String jlcVersion,
                               String gitSha,
                               String kernelSignatureVersion,
                               String variantSignatureVersion,
                               String compilerIdentity,
                               String codegenAbiVersion,
                               String strictFlags) {
        this.jlcVersion = text(jlcVersion);
        this.gitSha = text(gitSha);
        this.kernelSignatureVersion = text(kernelSignatureVersion);
        this.variantSignatureVersion = text(variantSignatureVersion);
        this.compilerIdentity = text(compilerIdentity);
        this.codegenAbiVersion = text(codegenAbiVersion);
        this.strictFlags = text(strictFlags);
        this.key = sha256(canonicalText());
    }

    public static KernelBuildIdentity current() {
        return new KernelBuildIdentity(
            System.getProperty("jlc.version", "1.0-SNAPSHOT"),
            firstNonBlank(System.getProperty("jlc.git.sha"),
                System.getProperty("jlc.compiler.buildSha"), "unknown"),
            "jlc-kernel-signature-v1", "jlc-kernel-variant-v" + KernelVariantSignature.CURRENT_VERSION,
            firstNonBlank(System.getProperty("jlc.compiler.nativeCompiler"), "unknown"));
    }

    public String jlcVersion() { return jlcVersion; }
    public String gitSha() { return gitSha; }
    public String kernelSignatureVersion() { return kernelSignatureVersion; }
    public String variantSignatureVersion() { return variantSignatureVersion; }
    public String compilerIdentity() { return compilerIdentity; }
    public String codegenAbiVersion() { return codegenAbiVersion; }
    public String strictFlags() { return strictFlags; }
    public String key() { return key; }

    public String canonicalText() {
        return new StringJoiner("\n")
            .add("jlc-version=" + jlcVersion)
            .add("git-sha=" + gitSha)
            .add("kernel-signature-version=" + kernelSignatureVersion)
            .add("variant-signature-version=" + variantSignatureVersion)
            .add("codegen-abi=" + codegenAbiVersion)
            .add("compiler=" + compilerIdentity)
            .add("strict-flags=" + strictFlags)
            .toString();
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof KernelBuildIdentity identity && key.equals(identity.key);
    }

    @Override
    public int hashCode() { return key.hashCode(); }

    private static String text(String value) {
        return value == null || value.isBlank() ? "unknown" : value.trim();
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) return value.trim();
        }
        return "unknown";
    }

    private static String sha256(String value) {
        try {
            return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }
}
