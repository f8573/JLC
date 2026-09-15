// Handwritten kmajor6 consumer regression cases; retains the existing independent long-double oracle.
#include "jlc_native.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Case {
    int m, k, n, threads;
    bool ta = false, tb = false, ca = false, cb = false, cc = false;
    double alpha = -0.625, beta = 0.375;
    const char* label = "";
};

struct Matrix {
    int logical_rows, logical_cols, rows, cols, ld, offset;
    bool transpose, col_major;
    std::vector<double> data;

    Matrix(int r, int c, bool t, bool col, int extra = 5)
        : logical_rows(r), logical_cols(c), rows(t ? c : r), cols(t ? r : c),
          ld((col ? rows : cols) + extra), offset(13), transpose(t), col_major(col),
          data(offset + static_cast<std::size_t>(ld) * (col ? cols : rows) + 23,
               -991.125) {}

    std::size_t index(int i, int j) const {
        if (transpose) std::swap(i, j);
        return offset + (col_major ? static_cast<std::size_t>(j) * ld + i
                                   : static_cast<std::size_t>(i) * ld + j);
    }
    double get(int i, int j) const { return data[index(i, j)]; }
};

struct Totals {
    std::uint64_t cases = 0, bitwise = 0, oracle = 0, calls = 0;
    double worst_abs = 0.0, worst_rel = 0.0;
};

static int a_flags(const Case& x) {
    return (x.ta ? JLC_GEMM_FLAG_A_TRANSPOSE : 0) |
           (x.ca ? JLC_GEMM_FLAG_A_COL_MAJOR : 0);
}
static int b_flags(const Case& x) {
    return (x.tb ? JLC_GEMM_FLAG_B_TRANSPOSE : 0) |
           (x.cb ? JLC_GEMM_FLAG_B_COL_MAJOR : 0);
}
static int c_flags(const Case& x) { return x.cc ? JLC_GEMM_FLAG_C_COL_MAJOR : 0; }

static jlc_status call(const Case& x, Matrix& a, Matrix& b, Matrix& c) {
    return jlc_native_gemm_strided(
        a.data.data(), a.offset, a.ld, a.rows, a.cols, a_flags(x),
        b.data.data(), b.offset, b.ld, b.rows, b.cols, b_flags(x),
        c.data.data(), c.offset, c.ld, c.rows, c.cols, c_flags(x),
        x.alpha, x.beta, x.threads, JLC_GEMM_FLAG_FORCE_BUILTIN);
}

static bool one_case(const Case& x, bool pool, std::uint64_t seed, Totals& total,
                     bool profile_counts = false) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-0.75, 0.75);
    Matrix a(x.m, x.k, x.ta, x.ca), b(x.k, x.n, x.tb, x.cb),
           c(x.m, x.n, false, x.cc, 9);
    for (int i = 0; i < x.m; ++i)
        for (int p = 0; p < x.k; ++p) a.data[a.index(i, p)] = dist(rng);
    for (int p = 0; p < x.k; ++p)
        for (int j = 0; j < x.n; ++j) b.data[b.index(p, j)] = dist(rng);
    for (int i = 0; i < x.m; ++i)
        for (int j = 0; j < x.n; ++j) c.data[c.index(i, j)] = dist(rng);
    const auto initial = c.data;
    std::vector<double> panel;
    jlc_gemm_profile p_panel{}, p_shared{};

    for (int pass = 0; pass < 4; ++pass) {
        const char* mode = "kmajor6";
        setenv("JLC_NATIVE_A_LAYOUT", mode, 1);
        setenv("JLC_NATIVE_AVX2_MICROKERNEL", pass == 0 ? "6x8-u4" : "6x8-u4-asm", 1);
        c.data = initial;
        if (profile_counts) {
            jlc_native_profile_reset();
            jlc_native_profile_set_enabled(true);
        }
        const auto status = call(x, a, b, c);
        ++total.calls;
        if (profile_counts) {
            jlc_native_profile_snapshot(pass == 0 ? &p_panel : &p_shared);
            jlc_native_profile_set_enabled(false);
        }
        if (status != JLC_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL status label=%s pool=%d mode=%s status=%d\n",
                         x.label, pool, mode, static_cast<int>(status));
            return false;
        }
        if (pass == 0) panel = c.data;
        else if (std::memcmp(panel.data(), c.data.data(), c.data.size() * sizeof(double)) != 0) {
            std::fprintf(stderr, "FAIL bitwise label=%s pool=%d\n", x.label, pool);
            return false;
        }
    }

    total.bitwise += static_cast<std::uint64_t>(x.m) * x.n;
    const std::uint64_t elements = static_cast<std::uint64_t>(x.m) * x.n;
    const std::uint64_t target = elements < 25000 ? elements : 257;
    for (std::uint64_t s = 0; s < target; ++s) {
        const std::uint64_t q = target == elements ? s : (s * (elements - 1)) / (target - 1);
        const int i = static_cast<int>(q / x.n), j = static_cast<int>(q % x.n);
        long double sum = 0.0L;
        for (int p = 0; p < x.k; ++p)
            sum += static_cast<long double>(a.get(i, p)) * static_cast<long double>(b.get(p, j));
        const long double expected_ld = static_cast<long double>(x.alpha) * sum +
                                        static_cast<long double>(x.beta) * initial[c.index(i, j)];
        const double expected = static_cast<double>(expected_ld);
        const double got = c.get(i, j);
        const double abs_err = std::abs(got - expected);
        const double rel_err = abs_err / std::max(1.0, std::abs(expected));
        total.worst_abs = std::max(total.worst_abs, abs_err);
        total.worst_rel = std::max(total.worst_rel, rel_err);
        ++total.oracle;
        if (!std::isfinite(got) || rel_err > 3e-12) {
            std::fprintf(stderr, "FAIL oracle label=%s pool=%d i=%d j=%d got=%.17g expected=%.17g rel=%.3g\n",
                         x.label, pool, i, j, got, expected, rel_err);
            return false;
        }
    }

    auto guards = c.data;
    for (int i = 0; i < x.m; ++i)
        for (int j = 0; j < x.n; ++j) guards[c.index(i, j)] = initial[c.index(i, j)];
    if (std::memcmp(guards.data(), initial.data(), initial.size() * sizeof(double)) != 0) {
        std::fprintf(stderr, "FAIL guard label=%s pool=%d\n", x.label, pool);
        return false;
    }

    if (profile_counts) {
        const bool equal = p_panel.pack_a_calls == p_shared.pack_a_calls &&
            p_panel.pack_a_bytes == p_shared.pack_a_bytes &&
            p_panel.pack_b_calls == p_shared.pack_b_calls &&
            p_panel.pack_b_bytes == p_shared.pack_b_bytes &&
            p_panel.microtile_calls == p_shared.microtile_calls;
        for (int q = 0; q < 2; ++q) {
            const auto& p = q ? p_shared : p_panel;
            std::printf("COUNTS label=%s pool=%d mode=%s A=%llu A_bytes=%llu B=%llu B_bytes=%llu micro=%llu actual=%llu panels=%llu mc=%llu kc=%llu nc=%llu mr=%llu nr=%llu vendor=%llu\n",
                x.label, pool, q ? "asm" : "intrinsic",
                (unsigned long long)p.pack_a_calls, (unsigned long long)p.pack_a_bytes,
                (unsigned long long)p.pack_b_calls, (unsigned long long)p.pack_b_bytes,
                (unsigned long long)p.microtile_calls, (unsigned long long)p.last_actual_threads,
                (unsigned long long)p.last_panel_count, (unsigned long long)p.last_mc,
                (unsigned long long)p.last_kc, (unsigned long long)p.last_nc,
                (unsigned long long)p.last_mr, (unsigned long long)p.last_nr,
                (unsigned long long)p.vendor_calls);
        }
        if (!equal) return false;

    }
    ++total.cases;
    std::printf("PASS label=%s pool=%d m=%d k=%d n=%d threads=%d\n",
                x.label, pool, x.m, x.k, x.n, x.threads);
    return true;
}

int main() {
    std::vector<Case> cases;
    for (int k : {1,2,3,4,5,7,8,9,15,16,17,31,32,33,127,128,129,255,256,257,263,513}) {
        for (int tail = 0; tail < 6; ++tail) {
            cases.push_back({6+tail,k,8,1,false,false,false,false,false,1.,0.,"full-n"});
            cases.push_back({6+tail,k,11,1,false,false,false,false,false,-.625,.375,"n-tail"});
        }
    }
    for (bool ta : {false,true}) for (bool ca : {false,true})
        for (bool tb : {false,true}) for (bool cb : {false,true})
            cases.push_back({13,259,17,1,ta,tb,ca,cb,false,-.9,-.2,"storage"});
    cases.push_back({2053,513,17,1,false,false,false,false,false,.875,-.25,"multiple-mc-kc"});
    cases.push_back({73,257,385,4,false,false,false,false,true,.8,.3,"column-c-bypass"});
    cases.push_back({67,513,512,4,false,false,false,false,false,-.625,.375,"parallel-divisible"});
    cases.push_back({71,259,641,4,true,false,false,false,false,1.,-.25,"parallel-tail"});
    for(double alpha : {0.,1.,-.625}) for(double beta : {0.,1.,-.5})
        cases.push_back({13,513,17,1,false,false,false,false,false,alpha,beta,"alpha-beta"});
    Totals total;
    setenv("JLC_NATIVE_WORKER_AFFINITY","physical",1);
    for (bool pool : {false,true}) {
        auto ctx=pool?jlc_native_context_create(4,64,0):0;
        if(pool&&!ctx)return 20;
        setenv("JLC_NATIVE_A_PACKING","private",1);
        setenv("JLC_NATIVE_GEMM_SCHEDULER","panel",1);
        for(std::size_t i=0;i<cases.size();++i)
            if(!one_case(cases[i],pool,0x5a17ULL+i*7919,total,true))return 21;
        // Exercise both other existing packing call sites without changing them.
        for(const char* scheduler : {"panel","shared-tail"}) {
            setenv("JLC_NATIVE_GEMM_SCHEDULER",scheduler,1);
            for(const char* packing : {"private","l3"}) {
                setenv("JLC_NATIVE_A_PACKING",packing,1);
                for(int repeat=0;repeat<3;++repeat)
                    if(!one_case(cases[cases.size()-10],pool,7001+repeat,total,true))return 22;
            }
        }
        if(ctx)jlc_native_context_destroy(ctx);
    }
    std::printf("SUMMARY cases=%llu calls=%llu bitwise_outputs=%llu oracle_outputs=%llu worst_abs=%.17g worst_rel=%.17g tolerance=3e-12\n",
        (unsigned long long)total.cases,(unsigned long long)total.calls,
        (unsigned long long)total.bitwise,(unsigned long long)total.oracle,total.worst_abs,total.worst_rel);
    return 0;
}
