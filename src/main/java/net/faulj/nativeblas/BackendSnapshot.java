package net.faulj.nativeblas;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Diagnostics snapshot for backend selection.
 */
public record BackendSnapshot(
    BackendMode requestedBackend,
    String activeBackend,
    boolean fallbackToJava,
    NativeContext nativeContext
) {
    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("phase", "phase4");
        out.put("requested", requestedBackend.id());
        out.put("active", activeBackend);
        out.put("fallbackToJava", fallbackToJava);
        out.put("native", nativeContext.toMap());
        return out;
    }
}
