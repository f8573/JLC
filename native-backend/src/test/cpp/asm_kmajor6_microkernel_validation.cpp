// Link with a test-only wrapper around the unchanged intrinsic and the assembly.
// Guard pages are essential: ASan does not instrument handwritten assembly.
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
extern "C" void intrinsic_test(int,const double*,const double*,int,double*,int);
extern "C" void jlc_kmajor6_6x8_u4_asm(int,const double*,const double*,int,double*,int);
extern "C" int selector_test(const char*,const char*,int,int);
struct Guard {
    size_t bytes, total; void* mapping; double* p;
    Guard(size_t n,int slack,bool end) {
        size_t page=sysconf(_SC_PAGESIZE);bytes=((n+slack)*8+page-1)/page*page;
        if(!bytes)bytes=page;
        total=bytes+2*page;mapping=mmap(nullptr,total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(mapping==MAP_FAILED)std::abort();
        char* start=static_cast<char*>(mapping)+page;
        if(mprotect(start,bytes,PROT_READ|PROT_WRITE))std::abort();
        p=reinterpret_cast<double*>(end?start+bytes-(n+slack)*8:start+slack*8);
    }
    ~Guard(){munmap(mapping,total);}
};
int main(){
    struct Select{const char*k;const char*l;int mr,nr,want;};
    for(auto s:std::vector<Select>{{"6x8-u4-asm","kmajor6",0,0,7},{"6x8-u4","kmajor6",0,0,5},
        {"6x8-u4-asm",nullptr,0,0,4},{"6x8-u4-asm","row",0,0,4},{"6x8-u4-asm","bad",0,0,4},
        {"6x8-u4-asm","kmajor6",5,8,4},{"6x8-u4-asm","kmajor6",6,4,4},
        {"bad","kmajor6",0,0,0},{"6x8-noinline","kmajor6",0,0,0}}) {
        int got=selector_test(s.k,s.l,s.mr,s.nr);
        if(got!=s.want){printf("FAIL selector %s got=%d want=%d\n",s.k,got,s.want);return 1;}
    }
    std::mt19937_64 rng(20260913);std::uniform_real_distribution<double>d(-.75,.75);
    size_t cases=0,outputs=0;double worst=0;
    for(int k:{0,1,2,3,4,5,7,8,9,15,16,17,31,32,33,127,128,129,255,256,257,263})
    for(int pn:{8,16,128})for(int ldc:{8,11,17})for(int slack=0;slack<4;++slack)
    for(bool end:{false,true})for(bool zero:{false,true}) {
        size_t an=6*k,bn=k?(k-1)*pn+8:0,cn=5*ldc+8;
        Guard a(an,slack,end),b(bn,(slack+1)%4,end),c(cn,(slack+2)%4,end);
        for(size_t i=0;i<an;++i)a.p[i]=d(rng);
        for(size_t i=0;i<bn;++i)b.p[i]=d(rng);
        std::fill(c.p,c.p+cn,-991.125);
        for(int i=0;i<6;++i)for(int j=0;j<8;++j)c.p[i*ldc+j]=zero?0:d(rng);
        std::vector<double> initial(c.p,c.p+cn),ref=initial,acopy(a.p,a.p+an),bcopy(b.p,b.p+bn);
        intrinsic_test(k,a.p,b.p,pn,ref.data(),ldc);
        jlc_kmajor6_6x8_u4_asm(k,a.p,b.p,pn,c.p,ldc);
        if(memcmp(c.p,ref.data(),cn*8)||(an&&memcmp(a.p,acopy.data(),an*8))||(bn&&memcmp(b.p,bcopy.data(),bn*8))){
            printf("FAIL bitwise/guard/input K=%d pn=%d ldc=%d slack=%d end=%d zero=%d\n",k,pn,ldc,slack,end,zero);return 2;
        }
        for(int i=0;i<6;++i)for(int j=0;j<8;++j){
            long double sum=initial[i*ldc+j];
            for(int p=0;p<k;++p)sum+=(long double)a.p[6*p+i]*(long double)b.p[p*pn+j];
            double error=std::abs(c.p[i*ldc+j]-(double)sum)/std::max(1.,std::abs((double)sum));
            worst=std::max(worst,error);
            if(!std::isfinite(c.p[i*ldc+j])||error>3e-12){printf("FAIL oracle\n");return 3;}
            ++outputs;
        }
        ++cases;
    }
    printf("PASS selectors=9 cases=%zu outputs=%zu full-allocation-bitwise=true input-unchanged=true guard-pages=true worst_rel=%.17g tolerance=3e-12\n",cases,outputs,worst);
}
