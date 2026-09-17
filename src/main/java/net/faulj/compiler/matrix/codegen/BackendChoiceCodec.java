package net.faulj.compiler.matrix.codegen;

import java.util.LinkedHashMap;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;

/** JSON-boundary codec for the typed backend choice hierarchy. */
public final class BackendChoiceCodec {
    private BackendChoiceCodec() {
    }

    public static Map<String, Object> toMap(BackendChoice choice) {
        if (choice == null) {
            throw new IllegalArgumentException("Backend choice is required");
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("kind", choice.kind().tag());
        if (choice instanceof R2JavaBackend) {
            return result;
        }
        KernelVariantSignature variant = choice.variantOptional().orElseThrow(
            () -> new IllegalArgumentException("Native backend choice has no variant"));
        result.put("variantSignature", variant.canonicalText());
        result.put("variantSha256", variant.sha256());
        result.put("variantId", variant.variantId());
        return result;
    }

    public static BackendChoice fromNode(JsonNode node, KernelSignature signature) {
        if (node == null || !node.isObject()) {
            throw new IllegalArgumentException("Backend choice must be an object");
        }
        BackendChoice.BackendKind kind = BackendChoice.BackendKind.fromTag(
            text(node, "kind", ""));
        if (kind == BackendChoice.BackendKind.R2_JAVA) {
            if (nonNull(node, "variantSignature") || nonNull(node, "variantSha256")
                || nonNull(node, "variantId")) {
                throw new IllegalArgumentException("R2 Java choice must not carry a variant");
            }
            return new R2JavaBackend();
        }
        if (signature == null) {
            throw new IllegalArgumentException("Native backend choice requires a kernel signature");
        }
        String variantText = text(node, "variantSignature", "");
        KernelVariantSignature variant = KernelVariantSignature.fromCanonicalText(
            signature, variantText);
        String storedSha = nullableText(node, "variantSha256");
        if (storedSha != null && !storedSha.equals(variant.sha256())) {
            throw new IllegalArgumentException("Backend choice variant digest mismatch");
        }
        String storedId = nullableText(node, "variantId");
        if (storedId != null && !storedId.equals(variant.variantId())) {
            throw new IllegalArgumentException("Backend choice variant ID mismatch");
        }
        return switch (kind) {
            case SCALAR_NATIVE -> new ScalarNativeBackend(variant);
            case GENERATED_AVX2 -> new GeneratedAvx2Backend(variant);
            case R2_JAVA -> throw new AssertionError("handled above");
        };
    }

    private static String text(JsonNode node, String field, String fallback) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() ? fallback : value.asText(fallback);
    }

    private static String nullableText(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }

    private static boolean nonNull(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value != null && !value.isNull();
    }
}
