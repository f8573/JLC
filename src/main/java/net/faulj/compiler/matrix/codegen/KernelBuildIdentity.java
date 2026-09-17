package net.faulj.compiler.matrix.codegen;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Properties;
import java.util.StringJoiner;

/** Build/codegen compatibility key for persisted generated-kernel decisions. */
public final class KernelBuildIdentity {
    public static final String CODEGEN_ABI_VERSION = "jlc-r5-codegen-abi-v1";
    public static final String STRICT_FLAGS = "-O2,-fno-fast-math,-ffp-contract=off";
    public static final String KERNEL_SIGNATURE_VERSION = "jlc-kernel-signature-v1";

    private final String jlcVersion;
    private final String gitSha;
    private final String kernelSignatureVersion;
    private final String variantSignatureVersion;
    private final String compilerIdentity;
    private final String codegenAbiVersion;
    private final String strictFlags;
    private final boolean gitDirty;
    private final boolean identityVerified;
    private final String javaCompiler;
    private final String javaRuntime;
    private final String nativeCompilerVersion;
    private final String nativeVendor;
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
        this(jlcVersion, gitSha, kernelSignatureVersion, variantSignatureVersion,
            compilerIdentity, codegenAbiVersion, strictFlags, false, true,
            "explicit", "explicit", "explicit", "explicit");
    }

    public KernelBuildIdentity(String jlcVersion,
                               String gitSha,
                               String kernelSignatureVersion,
                               String variantSignatureVersion,
                               String compilerIdentity,
                               String codegenAbiVersion,
                               String strictFlags,
                               boolean gitDirty,
                               boolean identityVerified,
                               String javaCompiler,
                               String javaRuntime,
                               String nativeCompilerVersion,
                               String nativeVendor) {
        this.jlcVersion = text(jlcVersion);
        this.gitSha = text(gitSha);
        this.kernelSignatureVersion = text(kernelSignatureVersion);
        this.variantSignatureVersion = text(variantSignatureVersion);
        this.compilerIdentity = text(compilerIdentity);
        this.codegenAbiVersion = text(codegenAbiVersion);
        this.strictFlags = text(strictFlags);
        this.gitDirty = gitDirty;
        this.identityVerified = identityVerified;
        this.javaCompiler = text(javaCompiler);
        this.javaRuntime = text(javaRuntime);
        this.nativeCompilerVersion = text(nativeCompilerVersion);
        this.nativeVendor = text(nativeVendor);
        this.key = sha256(canonicalText());
    }

    public static KernelBuildIdentity current() {
        Properties fields = loadBuildFields();
        String gitSha = firstNonBlank(System.getProperty("jlc.git.sha"),
            fields.getProperty("git.sha"), "unknown");
        String dirtyText = firstNonBlank(System.getProperty("jlc.git.dirty"),
            fields.getProperty("git.dirty"), "unknown");
        boolean gitDirty = "true".equalsIgnoreCase(dirtyText);
        boolean verified = "true".equalsIgnoreCase(
            firstNonBlank(System.getProperty("jlc.build.identity.verified"),
                fields.getProperty("identity.verified"), "false"));
        if ("unknown".equalsIgnoreCase(dirtyText) || gitDirty) {
            verified = false;
        }
        String javaCompiler = firstNonBlank(fields.getProperty("java.compiler"), "unknown");
        String nativeCompilerVersion = firstNonBlank(
            fields.getProperty("native.compiler.version"), "unknown");
        if ("unknown".equalsIgnoreCase(gitSha)
            || "unknown".equalsIgnoreCase(javaCompiler)
            || "unknown".equalsIgnoreCase(nativeCompilerVersion)) {
            verified = false;
        }
        return new KernelBuildIdentity(
            firstNonBlank(System.getProperty("jlc.version"), fields.getProperty("jlc.version"),
                "1.0-SNAPSHOT"),
            gitSha,
            firstNonBlank(fields.getProperty("kernel.signature.version"),
                KERNEL_SIGNATURE_VERSION),
            firstNonBlank(fields.getProperty("kernel.variant.signature.version"),
                "jlc-kernel-variant-v" + KernelVariantSignature.CURRENT_VERSION),
            firstNonBlank(System.getProperty("jlc.compiler.nativeCompiler"),
                fields.getProperty("native.compiler"), "unknown"),
            firstNonBlank(fields.getProperty("codegen.abi"), CODEGEN_ABI_VERSION),
            firstNonBlank(fields.getProperty("strict.flags"), STRICT_FLAGS),
            gitDirty, verified,
            javaCompiler,
            firstNonBlank(fields.getProperty("java.runtime"),
                System.getProperty("java.version"), "unknown"),
            nativeCompilerVersion,
            firstNonBlank(fields.getProperty("native.vendor"), "unknown"));
    }

    public String jlcVersion() { return jlcVersion; }
    public String gitSha() { return gitSha; }
    public String kernelSignatureVersion() { return kernelSignatureVersion; }
    public String variantSignatureVersion() { return variantSignatureVersion; }
    public String compilerIdentity() { return compilerIdentity; }
    public String codegenAbiVersion() { return codegenAbiVersion; }
    public String strictFlags() { return strictFlags; }
    public boolean gitDirty() { return gitDirty; }
    public boolean identityVerified() { return identityVerified && !gitDirty; }
    public String javaCompiler() { return javaCompiler; }
    public String javaRuntime() { return javaRuntime; }
    public String nativeCompilerVersion() { return nativeCompilerVersion; }
    public String nativeVendor() { return nativeVendor; }
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
            .add("git-dirty=" + gitDirty)
            .add("identity-verified=" + identityVerified())
            .add("java-compiler=" + javaCompiler)
            .add("java-runtime=" + javaRuntime)
            .add("native-compiler-version=" + nativeCompilerVersion)
            .add("native-vendor=" + nativeVendor)
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

    private static Properties loadBuildFields() {
        Properties fields = new Properties();
        try (InputStream stream = KernelBuildIdentity.class
                .getResourceAsStream("/jlc-build.properties")) {
            if (stream != null) fields.load(stream);
        } catch (IOException ignored) {
            // Missing build metadata is intentionally represented as unverified.
        }
        return fields;
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
