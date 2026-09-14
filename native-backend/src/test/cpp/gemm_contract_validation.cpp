#include "jlc_native.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
static void clear_blocks() {
    for (const char* name : {"JLC_NATIVE_MR", "JLC_NATIVE_NR", "JLC_NATIVE_MC", "JLC_NATIVE_KC", "JLC_NATIVE_NC",
                            "JLC_NATIVE_GEMM_MC", "JLC_NATIVE_GEMM_KC", "JLC_NATIVE_GEMM_NC"}) env(name, nullptr);
}
static jlc_status call() {
    std::vector<double> a(2048, 0.25), b(2048, -0.5);
    double c = 1.0;
    return jlc_native_gemm(a.data(), 1, 2048, b.data(), 2048, 1, &c, 1, 1,
                           1.0, 0.0, 1, JLC_GEMM_FLAG_FORCE_BUILTIN);
}
static bool shape(int mc, int kc, int nc, int mr, int nr) {
    if (call() != JLC_STATUS_SUCCESS) return false;
    jlc_gemm_profile p{}; jlc_native_profile_snapshot(&p);
    const bool ok = p.last_mc == mc && p.last_kc == kc && p.last_nc == nc && p.last_mr == mr && p.last_nr == nr;
    if (!ok) std::printf("FAIL shape got=%llu/%llu/%llu/%llu/%llu expected=%d/%d/%d/%d/%d\n",
        (unsigned long long)p.last_mc, (unsigned long long)p.last_kc, (unsigned long long)p.last_nc,
        (unsigned long long)p.last_mr, (unsigned long long)p.last_nr, mc,kc,nc,mr,nr);
    return ok;
}
int main(int argc, char**) {
    clear_blocks(); env("JLC_NATIVE_AVX2_MICROKERNEL", "6x8"); env("JLC_NATIVE_A_PACKING", "private");
    jlc_gemm_profile baseline{};
    if (call() != JLC_STATUS_SUCCESS) return 1;
    jlc_native_profile_snapshot(&baseline);
    env("JLC_NATIVE_KC", "257suffix"); env("JLC_NATIVE_MC", "7"); env("JLC_NATIVE_NC", "129");
    const int mr = (int)baseline.last_mr, nr = (int)baseline.last_nr;
    if (!shape(((7+mr-1)/mr)*mr, 260, ((129+nr-1)/nr)*nr, mr, nr)) return 2;
    env("JLC_NATIVE_GEMM_MC", "7"); env("JLC_NATIVE_GEMM_KC", "257"); env("JLC_NATIVE_GEMM_NC", "128");
    if (!shape(7,257,128,mr,nr)) return 3;
    for (const char* invalid : {"0", "-1", "257x", "9999999999999999999999999", "7", "4097"}) {
        env("JLC_NATIVE_GEMM_KC", invalid);
        if (call() != JLC_STATUS_INTERNAL_ERROR) return 4;
    }
    clear_blocks();
    for (const char* invalid : {"0", "-1", "garbage", "9999999999999999999999999"}) {
        env("JLC_NATIVE_KC", invalid);
        if (!shape((int)baseline.last_mc,(int)baseline.last_kc,(int)baseline.last_nc,mr,nr)) return 5;
    }
    clear_blocks(); env("JLC_NATIVE_MR","100");env("JLC_NATIVE_NR","67");
    env("JLC_NATIVE_MC","1");env("JLC_NATIVE_KC","1");env("JLC_NATIVE_NC","129");
    if (!shape(100,8,134,64,64)) return 6;
    clear_blocks();
    // Serial calls intentionally do not parse parallel-only controls.
    env("JLC_NATIVE_GEMM_SCHEDULER","invalid");env("JLC_NATIVE_WORKER_AFFINITY","invalid");
    if (call()!=JLC_STATUS_SUCCESS) return 7;
    env("JLC_NATIVE_GEMM_SCHEDULER",nullptr);env("JLC_NATIVE_WORKER_AFFINITY",nullptr);
    if (std::strstr(jlc_native_runtime_description(),"AVX2")) {
        env("JLC_NATIVE_AVX2_MICROKERNEL","6x8-u2");
        if (call() != (argc>1 ? JLC_STATUS_SUCCESS : JLC_STATUS_INTERNAL_ERROR)) return 8;
    }
    clear_blocks();
    std::puts("PASS exact/legacy precedence, unchanged rounding, invalid controls, serial parsing, experimental build boundary");
}
