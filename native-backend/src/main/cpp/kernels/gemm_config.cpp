#include "gemm_internal.hpp"
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
#include "gemm_experimental.hpp"
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

constexpr int L1_CACHE = 32 * 1024;
constexpr int L2_CACHE = 256 * 1024;
constexpr int L3_CACHE = 8 * 1024 * 1024;

enum class IntegerPolicy { LegacyPrefix, Exact };

int parse_env_integer(const char* name, int fallback, IntegerPolicy policy,
                      int minimum, int maximum, int multiple = 1) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    const bool valid = end != value && (policy == IntegerPolicy::LegacyPrefix || *end == '\0')
        && parsed >= minimum && parsed <= maximum && parsed % multiple == 0;
    if (valid) return static_cast<int>(parsed);
    if (policy == IntegerPolicy::LegacyPrefix) return fallback;
    std::fprintf(stderr,
        "JLC GEMM block-size error: %s must be an integer in [%d, %d]%s (got '%s')\n",
        name, minimum, maximum, multiple > 1 ? " and a multiple of the kernel tile width" : "", value);
    throw std::runtime_error("invalid JLC GEMM block size");
}

#if defined(__AVX2__) && !defined(__AVX512F__)
inline GemmKernelOption selected_avx2_microkernel() {
    const char* value = std::getenv("JLC_NATIVE_AVX2_MICROKERNEL");
    if (value != nullptr && std::strcmp(value, "5x4") == 0) {
        return GemmKernelOption::Kernel5x4;
    }
    if (value != nullptr && std::strcmp(value, "5x8") == 0) {
        return GemmKernelOption::Kernel5x8;
    }
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    if (value != nullptr && std::strcmp(value, "6x8-u2") == 0) {
        return GemmKernelOption::Kernel6x8U2;
    }
    if (value != nullptr && std::strcmp(value, "6x8-u4-asm") == 0) {
        return GemmKernelOption::Kernel6x8U4Asm;
    }
    if (value != nullptr && std::strcmp(value, "6x8-u4") == 0) {
        return GemmKernelOption::Kernel6x8U4;
    }
    if (value != nullptr && std::strcmp(value, "6x8-noinline") == 0) {
        return GemmKernelOption::Kernel6x8NoInline;
    }
#else
    if (value != nullptr && (std::strcmp(value, "6x8-u2") == 0
        || std::strcmp(value, "6x8-u4") == 0 || std::strcmp(value, "6x8-u4-asm") == 0
        || std::strcmp(value, "6x8-noinline") == 0)) {
        throw std::runtime_error("GEMM consumer requires JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM=ON");
    }
#endif
    return GemmKernelOption::Kernel6x8;
}
#endif

bool parse_env_bool(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    if (value[0] == '0' || value[0] == 'f' || value[0] == 'F'
        || value[0] == 'n' || value[0] == 'N') {
        return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T'
        || value[0] == 'y' || value[0] == 'Y';
}

// Compatibility policy: automatic -> legacy prefix/round/clamp -> exact override.
// Legacy callers still exist; new benchmarks should use the strict GEMM_* names.
void resolve_block_overrides(int& mc, int& kc, int& nc, int& mr, int& nr) {
    const int mr_override = parse_env_integer("JLC_NATIVE_MR", 0, IntegerPolicy::LegacyPrefix, 1, 1'000'000);
    const int nr_override = parse_env_integer("JLC_NATIVE_NR", 0, IntegerPolicy::LegacyPrefix, 1, 1'000'000);
    const int kc_override = parse_env_integer("JLC_NATIVE_KC", 0, IntegerPolicy::LegacyPrefix, 1, 1'000'000);
    const int nc_override = parse_env_integer("JLC_NATIVE_NC", 0, IntegerPolicy::LegacyPrefix, 1, 1'000'000);
    const int mc_override = parse_env_integer("JLC_NATIVE_MC", 0, IntegerPolicy::LegacyPrefix, 1, 1'000'000);

    if (mr_override > 0) {
        mr = mr_override;
    }
    if (nr_override > 0) {
        nr = nr_override;
    }
    if (kc_override > 0) {
        kc = round_up(kc_override, 4);
    }
    if (nc_override > 0) {
        nc = round_up(nc_override, std::max(1, nr));
    }
    if (mc_override > 0) {
        mc = round_up(mc_override, std::max(1, mr));
    }
    mr = std::clamp(mr, 1, 64);
    nr = std::clamp(nr, 1, 64);
    kc = std::clamp(kc, 8, 4096);
    nc = std::clamp(nc, nr, 4096);
    mc = std::clamp(mc, mr, 4096);

    // Exact optional blocking controls. Unlike the legacy overrides above,
    // these reject illegal values instead of rounding or clamping them.
    mc = parse_env_integer("JLC_NATIVE_GEMM_MC", mc, IntegerPolicy::Exact, mr, 4096, 1);
    kc = parse_env_integer("JLC_NATIVE_GEMM_KC", kc, IntegerPolicy::Exact, 8, 4096, 1);
    nc = parse_env_integer("JLC_NATIVE_GEMM_NC", nc, IntegerPolicy::Exact, nr, 4096, nr);

}

ResolvedGemmPlan resolve_gemm_plan(int m, int n, int k) {
    int nr = 1;
    int mr = 4;
    GemmKernelOption requested_kernel = GemmKernelOption::Kernel6x8;
#if defined(__AVX512F__)
    nr = 8;
    mr = 6;
#elif defined(__AVX2__)
    requested_kernel = selected_avx2_microkernel();
    nr = requested_kernel == GemmKernelOption::Kernel5x4 ? 4 : 8;
    mr = requested_kernel == GemmKernelOption::Kernel5x4
        || requested_kernel == GemmKernelOption::Kernel5x8 ? 5 : 6;
#endif

    const int l1_constraint = std::max(64, L1_CACHE / (8 * mr * 2));
    const int l2_constraint = static_cast<int>(std::sqrt(L2_CACHE / (8.0 * 2.0)));

    int kc = round_up(std::min(l1_constraint, l2_constraint), 8);
    kc = std::clamp(kc, 64, 512);

    int nc = round_up((L2_CACHE / 2) / (kc * 8), nr);
    nc = std::clamp(nc, nr * 4, 4096);

    int mc = round_up(L3_CACHE / (8 * std::max(1, kc + nc)), mr);
    mc = std::clamp(mc, mr * 4, 2048);

#if defined(__AVX2__) && !defined(__AVX512F__)
    const int max_dim = std::max(m, std::max(n, k));
    if (max_dim >= 4096) {
        kc = std::max(kc, 384);
    } else if (max_dim >= 2048) {
        kc = std::max(kc, 256);
    }
#endif

    resolve_block_overrides(mc, kc, nc, mr, nr);

    ResolvedGemmPlan plan{mc, kc, nc, mr, nr};
    // ISA/layout compatibility is invariant for the call; only actual tails vary.
#if defined(__AVX512F__)
    if (mr == 6 && nr == 8) plan.tile = ResolvedGemmPlan::Tile::M6N8;
#endif
#if defined(__AVX2__)
    if (mr == 6 && nr == 8) plan.tile = ResolvedGemmPlan::Tile::M6N8;
    else if (mr == 5 && nr == 8) plan.tile = ResolvedGemmPlan::Tile::M5N8;
    else if (mr == 5 && nr == 4) plan.tile = ResolvedGemmPlan::Tile::M5N4;
#endif
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    resolve_experimental_layout(plan, requested_kernel);
#endif
    return plan;
}

GemmSchedulerMode parse_scheduler_mode() {
    const char* value = std::getenv("JLC_NATIVE_GEMM_SCHEDULER");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "panel") == 0) {
        return GemmSchedulerMode::Panel;
    }
    if (std::strcmp(value, "shared-tail") == 0) {
        return GemmSchedulerMode::SharedTail;
    }
    std::fprintf(stderr, "JLC GEMM scheduler must be 'panel' or 'shared-tail' (got '%s')\n", value);
    throw std::runtime_error("invalid JLC GEMM scheduler");
}

bool scheduler_diagnostics_requested() {
    const char* value = std::getenv("JLC_NATIVE_GEMM_SCHEDULER_DIAGNOSTICS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
GemmAPackingMode parse_a_packing_mode() {
    const char* value = std::getenv("JLC_NATIVE_A_PACKING");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "private") == 0) {
        return GemmAPackingMode::Private;
    }
    if (std::strcmp(value, "l3") == 0) {
        return GemmAPackingMode::SharedL3;
    }
    std::fprintf(stderr, "JLC A packing must be 'private' or 'l3' (got '%s')\n", value);
    throw std::runtime_error("invalid JLC A packing mode");
}

bool a_packing_diagnostics_requested() {
    return parse_env_bool("JLC_NATIVE_A_PACKING_DIAGNOSTICS", false);
}

#endif

GemmWorkerAffinityMode parse_worker_affinity_mode() {
    const char* value = std::getenv("JLC_NATIVE_WORKER_AFFINITY");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "none") == 0) {
        return GemmWorkerAffinityMode::None;
    }
    if (std::strcmp(value, "physical") == 0) {
        return GemmWorkerAffinityMode::Physical;
    }
    std::fprintf(stderr,
                 "JLC worker affinity error: JLC_NATIVE_WORKER_AFFINITY must be 'none' or 'physical' (got '%s')\n",
                 value);
    throw std::runtime_error("invalid JLC native worker affinity mode");
}

bool worker_affinity_diagnostics_requested() {
    return parse_env_bool("JLC_NATIVE_WORKER_AFFINITY_DIAGNOSTICS", false);
}

GemmRuntimeConfig resolve_gemm_runtime_config(bool parallel) {
    GemmRuntimeConfig config;
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    config.a_packing = parse_a_packing_mode();
    config.a_packing_diagnostics = a_packing_diagnostics_requested();
#else
    const char* a_packing = std::getenv("JLC_NATIVE_A_PACKING");
    if (a_packing != nullptr && *a_packing != '\0' && std::strcmp(a_packing, "private") != 0)
        throw std::runtime_error("A packing requires experimental GEMM build (supported here: private)");
#endif
    if (parallel) {
        config.scheduler = parse_scheduler_mode();
        config.scheduler_diagnostics = scheduler_diagnostics_requested();
        config.worker_affinity = parse_worker_affinity_mode();
        config.worker_affinity_diagnostics = worker_affinity_diagnostics_requested();
    }
    return config;
}

} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
