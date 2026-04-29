#include <jni.h>

#include <algorithm>
#include <limits>

#include "jlc_native.h"

namespace {
void throw_java_exception(JNIEnv* env, const char* class_name, const char* message) {
    jclass clazz = env->FindClass(class_name);
    if (clazz != nullptr) {
        env->ThrowNew(clazz, message);
    }
}

bool validate_array_length(JNIEnv* env, jdoubleArray array, jsize expected, const char* name) {
    if (array == nullptr) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Native GEMM requires non-null arrays");
        return false;
    }
    if (env->GetArrayLength(array) != expected) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", name);
        return false;
    }
    return true;
}

bool validate_int_array_length(JNIEnv* env, jintArray array, jsize expected, const char* name) {
    if (array == nullptr) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Native LAPACK requires non-null arrays");
        return false;
    }
    if (env->GetArrayLength(array) != expected) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", name);
        return false;
    }
    return true;
}

bool validate_non_negative(JNIEnv* env, jint value, const char* message) {
    if (value < 0) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", message);
        return false;
    }
    return true;
}

long long storage_span(int rows, int cols, int ld, int flags) {
    if (rows <= 0 || cols <= 0) {
        return 0LL;
    }
    const bool col_major = (flags & (JLC_GEMM_FLAG_A_COL_MAJOR | JLC_GEMM_FLAG_B_COL_MAJOR | JLC_GEMM_FLAG_C_COL_MAJOR)) != 0;
    if (col_major) {
        return static_cast<long long>(cols - 1) * ld + rows;
    }
    return static_cast<long long>(rows - 1) * ld + cols;
}

bool validate_strided_array(JNIEnv* env, jdoubleArray array,
                            jint offset, jint ld, jint rows, jint cols, jint flags,
                            const char* name) {
    if (array == nullptr) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Native GEMM requires non-null arrays");
        return false;
    }
    if (!validate_non_negative(env, offset, "Offsets must be non-negative")
        || !validate_non_negative(env, ld, "Leading dimensions must be non-negative")
        || !validate_non_negative(env, rows, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, cols, "Matrix dimensions must be non-negative")) {
        return false;
    }
    if (rows > 0 && cols > 0) {
        const bool col_major = (flags & (JLC_GEMM_FLAG_A_COL_MAJOR | JLC_GEMM_FLAG_B_COL_MAJOR | JLC_GEMM_FLAG_C_COL_MAJOR)) != 0;
        const int min_ld = col_major ? rows : cols;
        if (ld < min_ld) {
            throw_java_exception(env, "java/lang/IllegalArgumentException", "Leading dimension is too small for the requested layout");
            return false;
        }
    }
    const long long required = static_cast<long long>(offset) + storage_span(rows, cols, ld, flags);
    if (required > static_cast<long long>(env->GetArrayLength(array))) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", name);
        return false;
    }
    return true;
}

bool validate_batched_strided_array(JNIEnv* env, jdoubleArray array,
                                    jint offset, jint ld, jint rows, jint cols, jint flags,
                                    jint stride, jint batch_count,
                                    const char* name) {
    if (!validate_strided_array(env, array, offset, ld, rows, cols, flags, name)) {
        return false;
    }
    if (!validate_non_negative(env, stride, "Batch stride must be non-negative")
        || !validate_non_negative(env, batch_count, "Batch count must be non-negative")) {
        return false;
    }
    if (batch_count <= 1) {
        return true;
    }
    const long long span = storage_span(rows, cols, ld, flags);
    const long long required = static_cast<long long>(offset)
        + static_cast<long long>(batch_count - 1) * static_cast<long long>(stride)
        + span;
    if (required > static_cast<long long>(env->GetArrayLength(array))) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", name);
        return false;
    }
    return true;
}

bool validate_direct_buffer(JNIEnv* env, jobject buffer,
                            jlong byte_offset, jint ld, jint rows, jint cols, jint flags,
                            const char* name) {
    if (buffer == nullptr) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Native GEMM requires non-null direct buffers");
        return false;
    }
    if (byte_offset < 0 || (byte_offset % static_cast<jlong>(sizeof(double))) != 0) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Byte offsets must be non-negative and 8-byte aligned");
        return false;
    }
    if (!validate_non_negative(env, ld, "Leading dimensions must be non-negative")
        || !validate_non_negative(env, rows, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, cols, "Matrix dimensions must be non-negative")) {
        return false;
    }

    void* address = env->GetDirectBufferAddress(buffer);
    jlong capacity = env->GetDirectBufferCapacity(buffer);
    if (address == nullptr || capacity < 0) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Native GEMM requires direct ByteBuffer inputs");
        return false;
    }

    if (rows > 0 && cols > 0) {
        const bool col_major = (flags & (JLC_GEMM_FLAG_A_COL_MAJOR | JLC_GEMM_FLAG_B_COL_MAJOR | JLC_GEMM_FLAG_C_COL_MAJOR)) != 0;
        const int min_ld = col_major ? rows : cols;
        if (ld < min_ld) {
            throw_java_exception(env, "java/lang/IllegalArgumentException", "Leading dimension is too small for the requested layout");
            return false;
        }
    }

    const long long required_bytes = static_cast<long long>(byte_offset)
        + storage_span(rows, cols, ld, flags) * static_cast<long long>(sizeof(double));
    if (required_bytes > static_cast<long long>(capacity)) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", name);
        return false;
    }
    return true;
}

template <typename Fn>
bool with_critical_arrays(JNIEnv* env, jdoubleArray a, jdoubleArray b, jdoubleArray c, Fn&& fn) {
    jdouble* a_ptr = nullptr;
    jdouble* b_ptr = nullptr;
    jdouble* c_ptr = nullptr;

    a_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, nullptr));
    b_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(b, nullptr));
    c_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(c, nullptr));

    if (a_ptr == nullptr || b_ptr == nullptr || c_ptr == nullptr) {
        if (a_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
        }
        if (b_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(b, b_ptr, JNI_ABORT);
        }
        if (c_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(c, c_ptr, 0);
        }
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native GEMM");
        return false;
    }

    const jlc_status status = fn(a_ptr, b_ptr, c_ptr);

    env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(b, b_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(c, c_ptr, 0);

    if (status == JLC_STATUS_SUCCESS) {
        return true;
    }
    if (status == JLC_STATUS_INVALID_ARGUMENT) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "JLC_NATIVE_INVALID_ARGUMENT");
        return false;
    }
    if (status == JLC_STATUS_OUT_OF_MEMORY) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "JLC_NATIVE_OUT_OF_MEMORY");
        return false;
    }
    throw_java_exception(env, "java/lang/IllegalStateException", "JLC_NATIVE_INTERNAL_ERROR");
    return false;
}

void throw_status_exception(JNIEnv* env, jlc_status status) {
    if (status == JLC_STATUS_SUCCESS) {
        return;
    }
    if (status == JLC_STATUS_INVALID_ARGUMENT) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "JLC_NATIVE_INVALID_ARGUMENT");
        return;
    }
    if (status == JLC_STATUS_OUT_OF_MEMORY) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "JLC_NATIVE_OUT_OF_MEMORY");
        return;
    }
    throw_java_exception(env, "java/lang/IllegalStateException", "JLC_NATIVE_INTERNAL_ERROR");
}

}

extern "C" JNIEXPORT jboolean JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeIsAvailable(JNIEnv*, jclass) {
    return jlc_native_is_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeVendorLapackAvailable(JNIEnv*, jclass) {
    return jlc_native_vendor_lapack_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeRuntimeDescription(JNIEnv* env, jclass) {
    return env->NewStringUTF(jlc_native_runtime_description());
}

extern "C" JNIEXPORT jstring JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeProviderDescription(JNIEnv* env, jclass) {
    return env->NewStringUTF(jlc_native_provider_description());
}

extern "C" JNIEXPORT jlong JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeCreateContext(JNIEnv*, jclass,
                                                             jint preferred_threads,
                                                             jint alignment_bytes,
                                                             jint flags) {
    return static_cast<jlong>(jlc_native_context_create(preferred_threads, alignment_bytes, flags));
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeDestroyContext(JNIEnv*, jclass, jlong handle) {
    jlc_native_context_destroy(static_cast<jlc_context_handle>(handle));
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeProfileSetEnabled(JNIEnv*, jclass, jboolean enabled) {
    jlc_native_profile_set_enabled(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeProfileReset(JNIEnv*, jclass) {
    jlc_native_profile_reset();
}

extern "C" JNIEXPORT jlong JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeMatrixCreate(JNIEnv* env, jclass,
                                                            jint rows, jint cols, jint order,
                                                            jint alignment_bytes) {
    if (!validate_non_negative(env, rows, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, cols, "Matrix dimensions must be non-negative")) {
        return 0;
    }
    if (order != static_cast<jint>(JLC_MATRIX_ROW_MAJOR)
        && order != static_cast<jint>(JLC_MATRIX_COL_MAJOR)) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Invalid native matrix order");
        return 0;
    }

    const jlc_matrix_handle handle = jlc_native_matrix_create(rows, cols, order, alignment_bytes);
    if (handle == 0) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "Failed to allocate native matrix buffer");
        return 0;
    }
    return static_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeMatrixDestroy(JNIEnv*, jclass, jlong handle) {
    jlc_native_matrix_destroy(static_cast<jlc_matrix_handle>(handle));
}

extern "C" JNIEXPORT jobject JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeMatrixBuffer(JNIEnv* env, jclass, jlong handle) {
    if (handle == 0) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Native matrix handle must be non-zero");
        return nullptr;
    }
    auto* data = jlc_native_matrix_data(static_cast<jlc_matrix_handle>(handle));
    const std::uint64_t bytes = jlc_native_matrix_bytes(static_cast<jlc_matrix_handle>(handle));
    return env->NewDirectByteBuffer(data, static_cast<jlong>(bytes));
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeHessenbergReduceVendor(JNIEnv* env, jclass,
                                                                      jdoubleArray h, jint n) {
    if (!validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return;
    }
    const jsize h_expected = static_cast<jsize>(n * n);
    if (!validate_array_length(env, h, h_expected, "Array length mismatch for Hessenberg workspace")) {
        return;
    }

    jdouble* h_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(h, nullptr));
    if (h_ptr == nullptr) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java array for native Hessenberg");
        return;
    }

    const jlc_status status = jlc_native_hessenberg_reduce_vendor(h_ptr, n);
    env->ReleasePrimitiveArrayCritical(h, h_ptr, 0);

    if (status == JLC_STATUS_SUCCESS) {
        return;
    }
    if (status == JLC_STATUS_INVALID_ARGUMENT) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "JLC_NATIVE_INVALID_ARGUMENT");
        return;
    }
    if (status == JLC_STATUS_OUT_OF_MEMORY) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "JLC_NATIVE_OUT_OF_MEMORY");
        return;
    }
    throw_java_exception(env, "java/lang/IllegalStateException", "JLC_NATIVE_INTERNAL_ERROR");
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeHessenbergDecomposeVendor(JNIEnv* env, jclass,
                                                                         jdoubleArray h, jint n,
                                                                         jdoubleArray q) {
    if (!validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return;
    }
    const jsize expected = static_cast<jsize>(n * n);
    if (!validate_array_length(env, h, expected, "Array length mismatch for Hessenberg workspace")
        || !validate_array_length(env, q, expected, "Array length mismatch for Hessenberg Q")) {
        return;
    }

    jdouble* h_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(h, nullptr));
    jdouble* q_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(q, nullptr));
    if (h_ptr == nullptr || q_ptr == nullptr) {
        if (h_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(h, h_ptr, 0);
        }
        if (q_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(q, q_ptr, 0);
        }
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native Hessenberg");
        return;
    }

    const jlc_status status = jlc_native_hessenberg_decompose_vendor(h_ptr, n, q_ptr);
    env->ReleasePrimitiveArrayCritical(h, h_ptr, 0);
    env->ReleasePrimitiveArrayCritical(q, q_ptr, 0);

    if (status == JLC_STATUS_SUCCESS) {
        return;
    }
    if (status == JLC_STATUS_INVALID_ARGUMENT) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "JLC_NATIVE_INVALID_ARGUMENT");
        return;
    }
    if (status == JLC_STATUS_OUT_OF_MEMORY) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "JLC_NATIVE_OUT_OF_MEMORY");
        return;
    }
    throw_java_exception(env, "java/lang/IllegalStateException", "JLC_NATIVE_INTERNAL_ERROR");
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeLuFactor(JNIEnv* env, jclass,
                                                        jdoubleArray packed_lu, jint n,
                                                        jintArray pivots) {
    if (!validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return;
    }
    const jsize expected = static_cast<jsize>(n * n);
    if (!validate_array_length(env, packed_lu, expected, "Array length mismatch for LU workspace")
        || !validate_int_array_length(env, pivots, n, "Array length mismatch for LU pivots")) {
        return;
    }

    jdouble* lu_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(packed_lu, nullptr));
    jint* pivot_ptr = static_cast<jint*>(env->GetPrimitiveArrayCritical(pivots, nullptr));
    if (lu_ptr == nullptr || pivot_ptr == nullptr) {
        if (lu_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(packed_lu, lu_ptr, 0);
        }
        if (pivot_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(pivots, pivot_ptr, 0);
        }
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native LU");
        return;
    }

    int info = 0;
    const jlc_status status = jlc_native_lu_factor(
        lu_ptr, n,
        reinterpret_cast<int*>(pivot_ptr), n,
        &info
    );

    env->ReleasePrimitiveArrayCritical(packed_lu, lu_ptr, 0);
    env->ReleasePrimitiveArrayCritical(pivots, pivot_ptr, 0);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeLuFactorVendor(JNIEnv* env, jclass,
                                                              jdoubleArray packed_lu, jint n,
                                                              jintArray pivots) {
    if (!validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return;
    }
    const jsize expected = static_cast<jsize>(n * n);
    if (!validate_array_length(env, packed_lu, expected, "Array length mismatch for LU workspace")
        || !validate_int_array_length(env, pivots, n, "Array length mismatch for LU pivots")) {
        return;
    }

    jdouble* lu_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(packed_lu, nullptr));
    jint* pivot_ptr = static_cast<jint*>(env->GetPrimitiveArrayCritical(pivots, nullptr));
    if (lu_ptr == nullptr || pivot_ptr == nullptr) {
        if (lu_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(packed_lu, lu_ptr, 0);
        }
        if (pivot_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(pivots, pivot_ptr, 0);
        }
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native LU");
        return;
    }

    int info = 0;
    const jlc_status status = jlc_native_lu_factor_vendor(
        lu_ptr, n,
        reinterpret_cast<int*>(pivot_ptr), n,
        &info
    );

    env->ReleasePrimitiveArrayCritical(packed_lu, lu_ptr, 0);
    env->ReleasePrimitiveArrayCritical(pivots, pivot_ptr, 0);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeQrFactorizeOnly(JNIEnv* env, jclass,
                                                               jdoubleArray a, jint m, jint n) {
    if (!validate_non_negative(env, m, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return;
    }
    const jsize a_expected = static_cast<jsize>(m * n);
    if (!validate_array_length(env, a, a_expected, "Array length mismatch for QR input")) {
        return;
    }

    jdouble* a_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, nullptr));
    if (a_ptr == nullptr) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native QR");
        return;
    }

    const jlc_status status = jlc_native_qr_factorize_only(a_ptr, m, n);
    env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeQrFactorizeOnlyVendor(JNIEnv* env, jclass,
                                                                     jdoubleArray a, jint m, jint n) {
    if (!validate_non_negative(env, m, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return;
    }
    const jsize a_expected = static_cast<jsize>(m * n);
    if (!validate_array_length(env, a, a_expected, "Array length mismatch for QR input")) {
        return;
    }

    jdouble* a_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, nullptr));
    if (a_ptr == nullptr) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native QR");
        return;
    }

    const jlc_status status = jlc_native_qr_factorize_only_vendor(a_ptr, m, n);
    env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeQrDecompose(JNIEnv* env, jclass,
                                                           jdoubleArray a, jint m, jint n, jint q_cols,
                                                           jdoubleArray q, jdoubleArray r) {
    if (!validate_non_negative(env, m, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, n, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, q_cols, "Matrix dimensions must be non-negative")) {
        return;
    }
    const int k = std::min(m, n);
    if (!(q_cols == k || q_cols == m)) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Q column count must match thin or full QR");
        return;
    }
    const jsize a_expected = static_cast<jsize>(m * n);
    const jsize q_expected = static_cast<jsize>(m * q_cols);
    const jsize r_expected = static_cast<jsize>(q_cols * n);
    if (!validate_array_length(env, a, a_expected, "Array length mismatch for QR input")
        || !validate_array_length(env, q, q_expected, "Array length mismatch for QR Q")
        || !validate_array_length(env, r, r_expected, "Array length mismatch for QR R")) {
        return;
    }

    jdouble* a_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, nullptr));
    jdouble* q_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(q, nullptr));
    jdouble* r_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(r, nullptr));
    if (a_ptr == nullptr || q_ptr == nullptr || r_ptr == nullptr) {
        if (a_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
        }
        if (q_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(q, q_ptr, 0);
        }
        if (r_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(r, r_ptr, 0);
        }
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native QR");
        return;
    }

    const jlc_status status = jlc_native_qr_decompose(
        a_ptr, m, n, q_cols,
        q_ptr, r_ptr
    );

    env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(q, q_ptr, 0);
    env->ReleasePrimitiveArrayCritical(r, r_ptr, 0);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeQrDecomposeVendor(JNIEnv* env, jclass,
                                                                 jdoubleArray a, jint m, jint n, jint q_cols,
                                                                 jdoubleArray q, jdoubleArray r) {
    if (!validate_non_negative(env, m, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, n, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, q_cols, "Matrix dimensions must be non-negative")) {
        return;
    }
    const int k = std::min(m, n);
    if (!(q_cols == k || q_cols == m)) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Q column count must match thin or full QR");
        return;
    }
    const jsize a_expected = static_cast<jsize>(m * n);
    const jsize q_expected = static_cast<jsize>(m * q_cols);
    const jsize r_expected = static_cast<jsize>(q_cols * n);
    if (!validate_array_length(env, a, a_expected, "Array length mismatch for QR input")
        || !validate_array_length(env, q, q_expected, "Array length mismatch for QR Q")
        || !validate_array_length(env, r, r_expected, "Array length mismatch for QR R")) {
        return;
    }

    jdouble* a_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, nullptr));
    jdouble* q_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(q, nullptr));
    jdouble* r_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(r, nullptr));
    if (a_ptr == nullptr || q_ptr == nullptr || r_ptr == nullptr) {
        if (a_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
        }
        if (q_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(q, q_ptr, 0);
        }
        if (r_ptr != nullptr) {
            env->ReleasePrimitiveArrayCritical(r, r_ptr, 0);
        }
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java arrays for native QR");
        return;
    }

    const jlc_status status = jlc_native_qr_decompose_vendor(
        a_ptr, m, n, q_cols,
        q_ptr, r_ptr
    );

    env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(q, q_ptr, 0);
    env->ReleasePrimitiveArrayCritical(r, r_ptr, 0);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeCholeskyDecompose(JNIEnv* env, jclass,
                                                                 jdoubleArray packed_l, jint n) {
    if (!validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return 0;
    }
    const jsize expected = static_cast<jsize>(n * n);
    if (!validate_array_length(env, packed_l, expected, "Array length mismatch for Cholesky workspace")) {
        return 0;
    }

    jdouble* l_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(packed_l, nullptr));
    if (l_ptr == nullptr) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java array for native Cholesky");
        return 0;
    }

    int info = 0;
    const jlc_status status = jlc_native_cholesky_decompose(l_ptr, n, &info);
    env->ReleasePrimitiveArrayCritical(packed_l, l_ptr, 0);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
        return 0;
    }
    return static_cast<jint>(info);
}

extern "C" JNIEXPORT jint JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeCholeskyDecomposeVendor(JNIEnv* env, jclass,
                                                                       jdoubleArray packed_l, jint n) {
    if (!validate_non_negative(env, n, "Matrix dimensions must be non-negative")) {
        return 0;
    }
    const jsize expected = static_cast<jsize>(n * n);
    if (!validate_array_length(env, packed_l, expected, "Array length mismatch for Cholesky workspace")) {
        return 0;
    }

    jdouble* l_ptr = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(packed_l, nullptr));
    if (l_ptr == nullptr) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to pin Java array for native Cholesky");
        return 0;
    }

    int info = 0;
    const jlc_status status = jlc_native_cholesky_decompose_vendor(l_ptr, n, &info);
    env->ReleasePrimitiveArrayCritical(packed_l, l_ptr, 0);

    if (status != JLC_STATUS_SUCCESS) {
        throw_status_exception(env, status);
        return 0;
    }
    return static_cast<jint>(info);
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeProfileSnapshot(JNIEnv* env, jclass) {
    constexpr jsize field_count = 23;
    jlong values[field_count];
    jlc_gemm_profile profile{};
    jlc_native_profile_snapshot(&profile);

    values[0] = static_cast<jlong>(profile.calls);
    values[1] = static_cast<jlong>(profile.wall_ns);
    values[2] = static_cast<jlong>(profile.vendor_calls);
    values[3] = static_cast<jlong>(profile.vendor_ns);
    values[4] = static_cast<jlong>(profile.scale_c_ns);
    values[5] = static_cast<jlong>(profile.pack_a_ns);
    values[6] = static_cast<jlong>(profile.pack_b_ns);
    values[7] = static_cast<jlong>(profile.kernel_ns);
    values[8] = static_cast<jlong>(profile.thread_launch_ns);
    values[9] = static_cast<jlong>(profile.thread_join_ns);
    values[10] = static_cast<jlong>(profile.pack_a_calls);
    values[11] = static_cast<jlong>(profile.pack_b_calls);
    values[12] = static_cast<jlong>(profile.microtile_calls);
    values[13] = static_cast<jlong>(profile.pack_a_bytes);
    values[14] = static_cast<jlong>(profile.pack_b_bytes);
    values[15] = static_cast<jlong>(profile.last_requested_threads);
    values[16] = static_cast<jlong>(profile.last_actual_threads);
    values[17] = static_cast<jlong>(profile.last_panel_count);
    values[18] = static_cast<jlong>(profile.last_mc);
    values[19] = static_cast<jlong>(profile.last_kc);
    values[20] = static_cast<jlong>(profile.last_nc);
    values[21] = static_cast<jlong>(profile.last_mr);
    values[22] = static_cast<jlong>(profile.last_nr);

    jlongArray out = env->NewLongArray(field_count);
    if (out == nullptr) {
        return nullptr;
    }
    env->SetLongArrayRegion(out, 0, field_count, values);
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeGemm(JNIEnv* env, jclass,
                                                    jdoubleArray a, jint a_rows, jint a_cols,
                                                    jdoubleArray b, jint b_rows, jint b_cols,
                                                    jdoubleArray c, jint c_rows, jint c_cols,
                                                    jdouble alpha, jdouble beta,
                                                    jint threads, jint flags) {
    if (!validate_non_negative(env, a_rows, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, a_cols, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, b_rows, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, b_cols, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, c_rows, "Matrix dimensions must be non-negative")
        || !validate_non_negative(env, c_cols, "Matrix dimensions must be non-negative")) {
        return;
    }

    const jsize a_expected = static_cast<jsize>(a_rows * a_cols);
    const jsize b_expected = static_cast<jsize>(b_rows * b_cols);
    const jsize c_expected = static_cast<jsize>(c_rows * c_cols);
    if (!validate_array_length(env, a, a_expected, "Array length mismatch for A")
        || !validate_array_length(env, b, b_expected, "Array length mismatch for B")
        || !validate_array_length(env, c, c_expected, "Array length mismatch for C")) {
        return;
    }

    with_critical_arrays(env, a, b, c, [&](jdouble* a_ptr, jdouble* b_ptr, jdouble* c_ptr) {
        return jlc_native_gemm(
            a_ptr, a_rows, a_cols,
            b_ptr, b_rows, b_cols,
            c_ptr, c_rows, c_cols,
            alpha, beta, threads, flags
        );
    });
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeGemmStrided(JNIEnv* env, jclass,
                                                           jdoubleArray a, jint a_offset, jint a_ld, jint a_rows, jint a_cols, jint a_flags,
                                                           jdoubleArray b, jint b_offset, jint b_ld, jint b_rows, jint b_cols, jint b_flags,
                                                           jdoubleArray c, jint c_offset, jint c_ld, jint c_rows, jint c_cols, jint c_flags,
                                                           jdouble alpha, jdouble beta,
                                                           jint threads, jint flags) {
    if (!validate_strided_array(env, a, a_offset, a_ld, a_rows, a_cols, a_flags, "Array bounds mismatch for A")
        || !validate_strided_array(env, b, b_offset, b_ld, b_rows, b_cols, b_flags, "Array bounds mismatch for B")
        || !validate_strided_array(env, c, c_offset, c_ld, c_rows, c_cols, c_flags, "Array bounds mismatch for C")) {
        return;
    }

    with_critical_arrays(env, a, b, c, [&](jdouble* a_ptr, jdouble* b_ptr, jdouble* c_ptr) {
        return jlc_native_gemm_strided(
            a_ptr, a_offset, a_ld, a_rows, a_cols, a_flags,
            b_ptr, b_offset, b_ld, b_rows, b_cols, b_flags,
            c_ptr, c_offset, c_ld, c_rows, c_cols, c_flags,
            alpha, beta, threads, flags
        );
    });
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeGemmStridedBatched(JNIEnv* env, jclass,
                                                                  jdoubleArray a, jint a_offset, jint a_ld, jint a_rows, jint a_cols, jint a_flags, jint a_stride,
                                                                  jdoubleArray b, jint b_offset, jint b_ld, jint b_rows, jint b_cols, jint b_flags, jint b_stride,
                                                                  jdoubleArray c, jint c_offset, jint c_ld, jint c_rows, jint c_cols, jint c_flags, jint c_stride,
                                                                  jdouble alpha, jdouble beta,
                                                                  jint batch_count,
                                                                  jint threads, jint flags) {
    if (!validate_batched_strided_array(env, a, a_offset, a_ld, a_rows, a_cols, a_flags, a_stride, batch_count,
                                        "Array bounds mismatch for A")
        || !validate_batched_strided_array(env, b, b_offset, b_ld, b_rows, b_cols, b_flags, b_stride, batch_count,
                                           "Array bounds mismatch for B")
        || !validate_batched_strided_array(env, c, c_offset, c_ld, c_rows, c_cols, c_flags, c_stride, batch_count,
                                           "Array bounds mismatch for C")) {
        return;
    }

    with_critical_arrays(env, a, b, c, [&](jdouble* a_ptr, jdouble* b_ptr, jdouble* c_ptr) {
        return jlc_native_gemm_strided_batched(
            a_ptr, a_offset, a_ld, a_rows, a_cols, a_flags, a_stride,
            b_ptr, b_offset, b_ld, b_rows, b_cols, b_flags, b_stride,
            c_ptr, c_offset, c_ld, c_rows, c_cols, c_flags, c_stride,
            alpha, beta,
            batch_count,
            threads, flags
        );
    });
}

extern "C" JNIEXPORT void JNICALL
Java_net_faulj_nativeblas_NativeBindings_nativeGemmDirect(JNIEnv* env, jclass,
                                                          jobject a_buffer, jlong a_byte_offset, jint a_ld, jint a_rows, jint a_cols, jint a_flags,
                                                          jobject b_buffer, jlong b_byte_offset, jint b_ld, jint b_rows, jint b_cols, jint b_flags,
                                                          jobject c_buffer, jlong c_byte_offset, jint c_ld, jint c_rows, jint c_cols, jint c_flags,
                                                          jdouble alpha, jdouble beta,
                                                          jint threads, jint flags) {
    if (!validate_direct_buffer(env, a_buffer, a_byte_offset, a_ld, a_rows, a_cols, a_flags, "Buffer bounds mismatch for A")
        || !validate_direct_buffer(env, b_buffer, b_byte_offset, b_ld, b_rows, b_cols, b_flags, "Buffer bounds mismatch for B")
        || !validate_direct_buffer(env, c_buffer, c_byte_offset, c_ld, c_rows, c_cols, c_flags, "Buffer bounds mismatch for C")) {
        return;
    }

    auto* a_ptr = static_cast<jdouble*>(env->GetDirectBufferAddress(a_buffer));
    auto* b_ptr = static_cast<jdouble*>(env->GetDirectBufferAddress(b_buffer));
    auto* c_ptr = static_cast<jdouble*>(env->GetDirectBufferAddress(c_buffer));
    if (a_ptr == nullptr || b_ptr == nullptr || c_ptr == nullptr) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Failed to resolve direct buffer addresses for native GEMM");
        return;
    }

    const jint a_offset = static_cast<jint>(a_byte_offset / static_cast<jlong>(sizeof(double)));
    const jint b_offset = static_cast<jint>(b_byte_offset / static_cast<jlong>(sizeof(double)));
    const jint c_offset = static_cast<jint>(c_byte_offset / static_cast<jlong>(sizeof(double)));
    const jlc_status status = jlc_native_gemm_strided(
        a_ptr, a_offset, a_ld, a_rows, a_cols, a_flags,
        b_ptr, b_offset, b_ld, b_rows, b_cols, b_flags,
        c_ptr, c_offset, c_ld, c_rows, c_cols, c_flags,
        alpha, beta, threads, flags
    );
    if (status == JLC_STATUS_SUCCESS) {
        return;
    }
    if (status == JLC_STATUS_INVALID_ARGUMENT) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "JLC_NATIVE_INVALID_ARGUMENT");
        return;
    }
    if (status == JLC_STATUS_OUT_OF_MEMORY) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "JLC_NATIVE_OUT_OF_MEMORY");
        return;
    }
    throw_java_exception(env, "java/lang/IllegalStateException", "JLC_NATIVE_INTERNAL_ERROR");
}
