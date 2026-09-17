#include "jlc_generated_kernel_registry.h"

#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {
struct RegisteredKernel {
    jlc_generated_kernel_descriptor descriptor;
    jlc_generated_kernel_entry entry;
};

std::mutex& registry_mutex() {
    static std::mutex value;
    return value;
}

std::unordered_map<std::string, RegisteredKernel>& registry() {
    static std::unordered_map<std::string, RegisteredKernel> value;
    return value;
}
}

bool jlc_generated_register(const jlc_generated_kernel_descriptor& descriptor,
                            jlc_generated_kernel_entry entry) {
    if (descriptor.signature == nullptr || descriptor.symbol == nullptr
        || descriptor.value_type == nullptr || entry == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(registry_mutex());
    const std::string key(descriptor.signature);
    auto& table = registry();
    const auto existing = table.find(key);
    if (existing != table.end()) {
        return existing->second.entry == entry
            && std::string(existing->second.descriptor.symbol) == descriptor.symbol;
    }
    table.emplace(key, RegisteredKernel{descriptor, entry});
    return true;
}

const jlc_generated_kernel_descriptor* jlc_generated_lookup(const char* signature,
                                                           jlc_generated_kernel_entry* entry) {
    if (entry != nullptr) {
        *entry = nullptr;
    }
    if (signature == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(registry_mutex());
    const auto& table = registry();
    const auto found = table.find(signature);
    if (found == table.end()) {
        return nullptr;
    }
    if (entry != nullptr) {
        *entry = found->second.entry;
    }
    return &found->second.descriptor;
}

std::size_t jlc_generated_registry_size() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    return registry().size();
}

bool jlc_generated_avx2_supported() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#elif defined(_MSC_VER)
    int registers[4] = {0, 0, 0, 0};
    __cpuid(registers, 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#else
    return false;
#endif
#else
    return false;
#endif
}

bool jlc_generated_execute(const char* signature,
                           const double* const* inputs,
                           std::size_t input_count,
                           double* output,
                           std::size_t rows,
                           std::size_t cols) {
    const bool has_elements = rows != 0 && cols != 0;
    if (signature == nullptr || (has_elements && output == nullptr)
        || (input_count > 0 && inputs == nullptr)) {
        return false;
    }
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
        return false;
    }
    jlc_generated_kernel_entry entry = nullptr;
    const jlc_generated_kernel_descriptor* descriptor =
        jlc_generated_lookup(signature, &entry);
    if (descriptor == nullptr || entry == nullptr) {
        return false;
    }
    if (rows != descriptor->rows || cols != descriptor->cols) {
        return false;
    }
    if (descriptor->backend == jlc_generated_backend::AVX2
        && !jlc_generated_avx2_supported()) {
        return false;
    }
    entry(inputs, input_count, output, rows, cols);
    return true;
}
