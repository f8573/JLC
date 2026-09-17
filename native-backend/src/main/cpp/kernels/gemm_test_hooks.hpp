#pragma once

#include "gemm_internal.hpp"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

#if defined(JLC_NATIVE_TEST_POOL_FAULT_INJECTION)
int pool_creation_failure_index() {
    const char* value = std::getenv("JLC_NATIVE_TEST_FAIL_POOL_CREATE_AT");
    if (value == nullptr || *value == '\0') return -1;
    char* end = nullptr;
    const long index = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || index < 0 || index > INT_MAX) {
        std::fprintf(stderr,
                     "JLC pool fault injection error: JLC_NATIVE_TEST_FAIL_POOL_CREATE_AT must be a non-negative integer (got '%s')\n",
                     value);
        throw std::runtime_error("invalid persistent-pool creation failure index");
    }
    return static_cast<int>(index);
}

bool pool_test_diagnostics_requested() {
    return parse_env_bool("JLC_NATIVE_TEST_POOL_DIAGNOSTICS", false);
}
#endif

} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
