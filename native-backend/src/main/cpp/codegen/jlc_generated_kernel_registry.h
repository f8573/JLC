#pragma once

#include <cstddef>
#include <cstdint>

/** Backend tag for the generated-kernel registry. */
enum class jlc_generated_backend : std::uint32_t {
    SCALAR_CPP = 0,
    AVX2 = 1
};

/** Common whole-region ABI used by generated registry adapters. */
using jlc_generated_kernel_entry = void (*)(
    const double* const* inputs,
    std::size_t input_count,
    double* output,
    std::size_t rows,
    std::size_t cols);

struct jlc_generated_kernel_descriptor {
    const char* signature;
    const char* variant;
    const char* symbol;
    jlc_generated_backend backend;
    std::uint32_t vector_width;
    const char* value_type;
    std::size_t rows;
    std::size_t cols;
    std::uint32_t unroll;
    const char* loop_form;
    const char* tail_policy;
};

bool jlc_generated_register(const jlc_generated_kernel_descriptor& descriptor,
                            jlc_generated_kernel_entry entry);

const jlc_generated_kernel_descriptor* jlc_generated_lookup(const char* signature,
                                                           jlc_generated_kernel_entry* entry);

const jlc_generated_kernel_descriptor* jlc_generated_lookup_variant(
    const char* signature, const char* variant, jlc_generated_kernel_entry* entry);

std::size_t jlc_generated_registry_size();

/** Runtime, rather than compile-machine, AVX2 capability check. */
bool jlc_generated_avx2_supported();

/** Execute one complete registered region; false means safe registry fallback. */
bool jlc_generated_execute(const char* signature,
                           const double* const* inputs,
                           std::size_t input_count,
                           double* output,
                           std::size_t rows,
                           std::size_t cols);

/** Execute one exact R5 variant; false means safe fallback. */
bool jlc_generated_execute_variant(const char* signature,
                                   const char* variant,
                                   const double* const* inputs,
                                   std::size_t input_count,
                                   double* output,
                                   std::size_t rows,
                                   std::size_t cols);
