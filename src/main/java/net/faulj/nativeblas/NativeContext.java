package net.faulj.nativeblas;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Native runtime probe status exposed to diagnostics.
 */
public final class NativeContext {
    private final boolean libraryLoaded;
    private final NativeStatus status;
    private final String message;
    private final String runtimeDescription;
    private final String providerDescription;
    private final String libraryPath;
    private final NativeMatrixHandle workspaceHandle;

    public NativeContext(boolean libraryLoaded, NativeStatus status, String message,
                         String runtimeDescription, String providerDescription, String libraryPath,
                         NativeMatrixHandle workspaceHandle) {
        this.libraryLoaded = libraryLoaded;
        this.status = status;
        this.message = message;
        this.runtimeDescription = runtimeDescription;
        this.providerDescription = providerDescription;
        this.libraryPath = libraryPath;
        this.workspaceHandle = workspaceHandle == null ? NativeMatrixHandle.NULL : workspaceHandle;
    }

    public static NativeContext notRequested() {
        return new NativeContext(false, NativeStatus.NOT_REQUESTED, "Native backend not requested", null, null, null, NativeMatrixHandle.NULL);
    }

    public boolean isLibraryLoaded() {
        return libraryLoaded;
    }

    public NativeStatus getStatus() {
        return status;
    }

    public String getMessage() {
        return message;
    }

    public String getRuntimeDescription() {
        return runtimeDescription;
    }

    public String getLibraryPath() {
        return libraryPath;
    }

    public String getProviderDescription() {
        return providerDescription;
    }

    public NativeMatrixHandle getWorkspaceHandle() {
        return workspaceHandle;
    }

    public boolean isAvailable() {
        return status == NativeStatus.READY;
    }

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("library", "jlc_native");
        out.put("status", status.name());
        out.put("libraryLoaded", libraryLoaded);
        out.put("available", isAvailable());
        out.put("message", message);
        out.put("runtimeDescription", runtimeDescription);
        out.put("providerDescription", providerDescription);
        out.put("libraryPath", libraryPath);
        out.put("workspaceHandle", workspaceHandle.address());
        return out;
    }
}
