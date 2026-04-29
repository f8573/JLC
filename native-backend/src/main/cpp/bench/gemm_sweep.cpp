#include "jlc_native.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
struct RunSummary {
    int size = 0;
    int requested_threads = 0;
    double best_seconds = 0.0;
    double median_seconds = 0.0;
    double mean_seconds = 0.0;
    double best_gflops = 0.0;
    double median_gflops = 0.0;
    double mean_gflops = 0.0;
    jlc_gemm_profile best_profile{};
};

std::vector<int> default_sizes() {
    return {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072};
}

std::vector<int> default_thread_sweep() {
    const unsigned int hardware = std::max(1u, std::thread::hardware_concurrency());
    std::vector<int> values = {1, 2, 4, 8, 12, 16, 24, static_cast<int>(hardware)};
    std::sort(values.begin(), values.end());
    values.erase(std::remove_if(values.begin(), values.end(), [](int value) { return value <= 0; }), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int parse_positive_int(const std::string& value, int fallback) {
    try {
        const int parsed = std::stoi(value);
        return parsed > 0 ? parsed : fallback;
    } catch (...) {
        return fallback;
    }
}

std::vector<int> parse_int_list(const std::string& value) {
    std::vector<int> values;
    std::size_t start = 0;
    while (start < value.size()) {
        const std::size_t end = value.find(',', start);
        const std::string token = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        const int parsed = parse_positive_int(token, -1);
        if (parsed <= 0) {
            return {};
        }
        values.push_back(parsed);
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int provider_flags(const std::string& provider) {
    if (provider == "auto") {
        return JLC_GEMM_FLAG_PREFER_VENDOR;
    }
    if (provider == "vendor" || provider == "blas" || provider == "mkl" || provider == "openblas") {
        return JLC_GEMM_FLAG_FORCE_VENDOR;
    }
    if (provider == "builtin" || provider == "native" || provider == "kernel") {
        return JLC_GEMM_FLAG_FORCE_BUILTIN;
    }
    return 0;
}

void fill_random(std::vector<double>& values, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (double& value : values) {
        value = dist(rng);
    }
}

double total_profile_ns(const jlc_gemm_profile& profile) {
    return static_cast<double>(
        profile.vendor_ns
        + profile.scale_c_ns
        + profile.pack_a_ns
        + profile.pack_b_ns
        + profile.kernel_ns
        + profile.thread_launch_ns
        + profile.thread_join_ns
    );
}

double share_pct(std::uint64_t value, double total) {
    return total > 0.0 ? static_cast<double>(value) * 100.0 / total : 0.0;
}

double gib_per_second(std::uint64_t bytes, std::uint64_t ns) {
    if (ns == 0) {
        return 0.0;
    }
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (static_cast<double>(ns) / 1.0e9);
}

RunSummary run_case(int size, int requested_threads, int warmup_runs, int measured_runs, int flags, bool profile_enabled) {
    const std::size_t elements = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    std::vector<double> a(elements);
    std::vector<double> b(elements);
    std::vector<double> c(elements, 0.0);
    fill_random(a, 17'000u + static_cast<std::uint64_t>(size));
    fill_random(b, 29'000u + static_cast<std::uint64_t>(size));

    for (int run = 0; run < warmup_runs; ++run) {
        const jlc_status status = jlc_native_gemm(
            a.data(), size, size,
            b.data(), size, size,
            c.data(), size, size,
            1.0, 0.0,
            requested_threads, flags
        );
        if (status != JLC_STATUS_SUCCESS) {
            std::fprintf(stderr, "warmup failure at size %d threads=%d status=%d\n",
                         size, requested_threads, static_cast<int>(status));
            std::exit(1);
        }
    }

    std::vector<double> seconds(static_cast<std::size_t>(measured_runs), 0.0);
    jlc_gemm_profile best_profile{};
    double best_seconds = 1.0e300;
    for (int run = 0; run < measured_runs; ++run) {
        jlc_native_profile_reset();
        const auto start = std::chrono::steady_clock::now();
        const jlc_status status = jlc_native_gemm(
            a.data(), size, size,
            b.data(), size, size,
            c.data(), size, size,
            1.0, 0.0,
            requested_threads, flags
        );
        const auto end = std::chrono::steady_clock::now();
        if (status != JLC_STATUS_SUCCESS) {
            std::fprintf(stderr, "measured failure at size %d threads=%d status=%d\n",
                         size, requested_threads, static_cast<int>(status));
            std::exit(1);
        }
        jlc_gemm_profile run_profile{};
        jlc_native_profile_snapshot(&run_profile);
        const double elapsed = std::chrono::duration<double>(end - start).count();
        if (elapsed < best_seconds) {
            best_profile = run_profile;
        }
        seconds[static_cast<std::size_t>(run)] = elapsed;
        best_seconds = std::min(best_seconds, elapsed);
    }

    double total_seconds = std::accumulate(seconds.begin(), seconds.end(), 0.0);
    std::sort(seconds.begin(), seconds.end());
    const double mean_seconds = total_seconds / measured_runs;
    const double median_seconds = seconds[seconds.size() / 2];
    const double flops = 2.0 * size * static_cast<double>(size) * size;
    return RunSummary{
        size,
        requested_threads,
        best_seconds,
        median_seconds,
        mean_seconds,
        flops / best_seconds / 1.0e9,
        flops / median_seconds / 1.0e9,
        flops / mean_seconds / 1.0e9,
        best_profile
    };
}
}

int main(int argc, char** argv) {
    std::vector<int> sizes = default_sizes();
    std::vector<int> thread_values = {static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))};
    int warmup_runs = 6;
    int measured_runs = 6;
    std::string provider = "auto";
    bool profile_enabled = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] == nullptr ? "" : argv[i];
        if (arg == "--profile") {
            profile_enabled = true;
        } else if (arg == "--thread-sweep") {
            thread_values = default_thread_sweep();
        } else if (arg.rfind("--sizes=", 0) == 0) {
            std::vector<int> parsed = parse_int_list(arg.substr(8));
            if (!parsed.empty()) {
                sizes = std::move(parsed);
            }
        } else if (arg.rfind("--threads=", 0) == 0) {
            const int requested = parse_positive_int(arg.substr(10), thread_values.front());
            thread_values = {requested};
        } else if (arg.rfind("--thread-sweep=", 0) == 0) {
            std::vector<int> parsed = parse_int_list(arg.substr(15));
            if (!parsed.empty()) {
                thread_values = std::move(parsed);
            }
        } else if (arg.rfind("--warmup=", 0) == 0) {
            warmup_runs = parse_positive_int(arg.substr(9), warmup_runs);
        } else if (arg.rfind("--runs=", 0) == 0) {
            measured_runs = parse_positive_int(arg.substr(7), measured_runs);
        } else if (arg.rfind("--provider=", 0) == 0) {
            provider = arg.substr(11);
        }
    }

    const int flags = provider_flags(provider);
    const int workspace_threads = *std::max_element(thread_values.begin(), thread_values.end());
    const jlc_context_handle workspace = jlc_native_context_create(workspace_threads, 64, 0);
    jlc_native_profile_set_enabled(profile_enabled);

    std::printf("PURE_CPP_GEMM_SWEEP\n");
    std::printf("warmupRuns=%d\n", warmup_runs);
    std::printf("measuredRuns=%d\n", measured_runs);
    std::printf("threadValues=");
    for (std::size_t i = 0; i < thread_values.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", thread_values[i]);
    }
    std::printf("\n");
    std::printf("providerMode=%s\n", provider.c_str());
    std::printf("nativeProvider=%s\n", jlc_native_provider_description());
    std::printf("workspaceHandle=%llu\n", static_cast<unsigned long long>(workspace));
    std::printf("profileEnabled=%s\n", profile_enabled ? "true" : "false");
    std::printf("csv=size,requested_threads,best_ms,median_ms,mean_ms,best_gflops,median_gflops,mean_gflops,actual_threads,panel_count,mc,kc,nc,mr,nr,pack_a_pct,pack_b_pct,kernel_pct,thread_pct,scale_pct,vendor_pct,pack_a_gib_s,pack_b_gib_s\n");

    std::vector<RunSummary> summaries;
    summaries.reserve(sizes.size() * thread_values.size());
    for (int requested_threads : thread_values) {
        for (int size : sizes) {
            RunSummary summary = run_case(size, requested_threads, warmup_runs, measured_runs, flags, profile_enabled);
            const jlc_gemm_profile& profile = summary.best_profile;
            const double profile_total = total_profile_ns(profile);
            const double pack_a_pct = share_pct(profile.pack_a_ns, profile_total);
            const double pack_b_pct = share_pct(profile.pack_b_ns, profile_total);
            const double kernel_pct = share_pct(profile.kernel_ns, profile_total);
            const double thread_pct = share_pct(profile.thread_launch_ns + profile.thread_join_ns, profile_total);
            const double scale_pct = share_pct(profile.scale_c_ns, profile_total);
            const double vendor_pct = share_pct(profile.vendor_ns, profile_total);
            const double pack_a_gib_s = gib_per_second(profile.pack_a_bytes, profile.pack_a_ns);
            const double pack_b_gib_s = gib_per_second(profile.pack_b_bytes, profile.pack_b_ns);

            std::printf(
                "size=%d requested_threads=%d best_ms=%.6f median_ms=%.6f mean_ms=%.6f best_gflops=%.6f actual_threads=%llu profile_kernel_pct=%.2f profile_pack_a_pct=%.2f profile_pack_b_pct=%.2f\n",
                summary.size,
                summary.requested_threads,
                summary.best_seconds * 1000.0,
                summary.median_seconds * 1000.0,
                summary.mean_seconds * 1000.0,
                summary.best_gflops,
                static_cast<unsigned long long>(profile.last_actual_threads),
                kernel_pct,
                pack_a_pct,
                pack_b_pct
            );
            std::printf(
                "csv_row=%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                summary.size,
                summary.requested_threads,
                summary.best_seconds * 1000.0,
                summary.median_seconds * 1000.0,
                summary.mean_seconds * 1000.0,
                summary.best_gflops,
                summary.median_gflops,
                summary.mean_gflops,
                static_cast<unsigned long long>(profile.last_actual_threads),
                static_cast<unsigned long long>(profile.last_panel_count),
                static_cast<unsigned long long>(profile.last_mc),
                static_cast<unsigned long long>(profile.last_kc),
                static_cast<unsigned long long>(profile.last_nc),
                static_cast<unsigned long long>(profile.last_mr),
                static_cast<unsigned long long>(profile.last_nr),
                pack_a_pct,
                pack_b_pct,
                kernel_pct,
                thread_pct,
                scale_pct,
                vendor_pct,
                pack_a_gib_s,
                pack_b_gib_s
            );
            summaries.push_back(summary);
        }
    }

    if (thread_values.size() > 1) {
        std::printf("BEST_BY_SIZE\n");
        std::printf("best_csv=size,best_requested_threads,best_ms,best_gflops,actual_threads,kernel_pct,pack_a_pct,pack_b_pct,thread_pct\n");
        for (int size : sizes) {
            const RunSummary* best = nullptr;
            for (const RunSummary& candidate : summaries) {
                if (candidate.size != size) {
                    continue;
                }
                if (best == nullptr || candidate.best_gflops > best->best_gflops) {
                    best = &candidate;
                }
            }
            if (best == nullptr) {
                continue;
            }
            const jlc_gemm_profile& profile = best->best_profile;
            const double profile_total = total_profile_ns(profile);
            std::printf(
                "best_csv_row=%d,%d,%.6f,%.6f,%llu,%.2f,%.2f,%.2f,%.2f\n",
                size,
                best->requested_threads,
                best->best_seconds * 1000.0,
                best->best_gflops,
                static_cast<unsigned long long>(profile.last_actual_threads),
                share_pct(profile.kernel_ns, profile_total),
                share_pct(profile.pack_a_ns, profile_total),
                share_pct(profile.pack_b_ns, profile_total),
                share_pct(profile.thread_launch_ns + profile.thread_join_ns, profile_total)
            );
        }
    }

    jlc_native_profile_set_enabled(false);
    jlc_native_context_destroy(workspace);
    return 0;
}
