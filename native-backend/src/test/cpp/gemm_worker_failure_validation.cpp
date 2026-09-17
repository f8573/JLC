#include "jlc_native.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

static bool call(bool failure) {
    const int m=97,k=263,n=641;
    std::vector<double> a(m*k,0.25),b(k*n,-0.5),c(m*n,1.0);
    const auto status=jlc_native_gemm(a.data(),m,k,b.data(),k,n,c.data(),m,n,1,0,4,JLC_GEMM_FLAG_FORCE_BUILTIN);
    if (failure) return status==JLC_STATUS_INTERNAL_ERROR;
    if (status!=JLC_STATUS_SUCCESS) return false;
    return std::all_of(c.begin(),c.end(),[](double v){return v == -32.875;});
}
int main() {
    int cases=0;
    setenv("JLC_NATIVE_A_PACKING","private",1);
    for (const char* scheduler : {"panel","shared-tail"})
    for (const char* affinity : {"none","physical"})
    for (bool pool : {false,true})
    for (const char* fault : {"startup","task","create"}) {
        if (pool && std::string(fault)=="create") continue; // Exhaustive pool construction matrix is separate.
        setenv("JLC_NATIVE_GEMM_SCHEDULER",scheduler,1);
        setenv("JLC_NATIVE_WORKER_AFFINITY",affinity,1);
        auto ctx=pool?jlc_native_context_create(4,64,0):0;
        setenv("JLC_NATIVE_TEST_WORKER_FAIL",fault,1);
        if (!call(true)) {std::printf("FAIL injected %s/%s pool=%d fault=%s\n",scheduler,affinity,pool,fault);return 1;}
        unsetenv("JLC_NATIVE_TEST_WORKER_FAIL");
        if (!call(false)) return 2;
        if (ctx) jlc_native_context_destroy(ctx);
        if (!call(false)) return 3;
        ++cases;
    }
    std::printf("PASS worker failure/recovery cases=%d: startup, task allocation, partial fallback launch, pool reuse, fallback reuse\n",cases);
}
