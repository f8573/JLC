/* JLC full 6x8 kmajor6/NR8 consumers, Linux x86-64 SysV.
 * Each logical K consumes one 64-byte B cache line. The prefetch variants use
 * temporal read prefetches at exact future-K distances and suppress any group
 * whose targets would extend beyond K.
 */
#define JLC_NR8_FUNCTION jlc_kmajor6_6x8_u4_nr8_asm
#ifdef JLC_NR8_BENCHMARK_GROUP4
#define JLC_NR8_PREFETCH_DISTANCE 0
#define JLC_NR8_GROUP4 1
#elif defined(JLC_NR8_BENCHMARK_DISTANCE)
#define JLC_NR8_PREFETCH_DISTANCE JLC_NR8_BENCHMARK_DISTANCE
#else
#define JLC_NR8_PREFETCH_DISTANCE 0
#endif
#include "kmajor6_6x8_u4_nr8_template.S"
#undef JLC_NR8_PREFETCH_DISTANCE
#undef JLC_NR8_FUNCTION

#if !defined(JLC_NR8_BENCHMARK_DISTANCE) && !defined(JLC_NR8_BENCHMARK_GROUP4)
#define JLC_NR8_FUNCTION jlc_kmajor6_6x8_u4_nr8_pf1_asm
#define JLC_NR8_PREFETCH_DISTANCE 1
#include "kmajor6_6x8_u4_nr8_template.S"
#undef JLC_NR8_PREFETCH_DISTANCE
#undef JLC_NR8_FUNCTION

#define JLC_NR8_FUNCTION jlc_kmajor6_6x8_u4_nr8_pf2_asm
#define JLC_NR8_PREFETCH_DISTANCE 2
#include "kmajor6_6x8_u4_nr8_template.S"
#undef JLC_NR8_PREFETCH_DISTANCE
#undef JLC_NR8_FUNCTION

#define JLC_NR8_FUNCTION jlc_kmajor6_6x8_u4_nr8_pf4_asm
#define JLC_NR8_PREFETCH_DISTANCE 4
#include "kmajor6_6x8_u4_nr8_template.S"
#undef JLC_NR8_PREFETCH_DISTANCE
#undef JLC_NR8_FUNCTION

#define JLC_NR8_FUNCTION jlc_kmajor6_6x8_u4_nr8_group4_asm
#define JLC_NR8_PREFETCH_DISTANCE 0
#define JLC_NR8_GROUP4 1
#include "kmajor6_6x8_u4_nr8_template.S"
#undef JLC_NR8_PREFETCH_DISTANCE
#undef JLC_NR8_FUNCTION
#endif

.section .note.GNU-stack,"",@progbits
