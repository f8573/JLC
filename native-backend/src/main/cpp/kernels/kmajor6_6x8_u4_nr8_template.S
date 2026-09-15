/* Instantiated by kmajor6_6x8_u4_nr8.S.
 * JLC_NR8_FUNCTION names the entry point. JLC_NR8_PREFETCH_DISTANCE selects
 * the prior four-prefetch distance experiment; JLC_NR8_GROUP4 selects one
 * prefetch for the first cache line of the next U4 group.
 */
#ifndef JLC_NR8_GROUP4
#define JLC_NR8_GROUP4 0
#endif
.text
.p2align 5
.globl JLC_NR8_FUNCTION
.hidden JLC_NR8_FUNCTION
.type JLC_NR8_FUNCTION, @function
JLC_NR8_FUNCTION:
.cfi_startproc
    mov %r8d, %r9d
    mov %rcx, %r8
    movslq %r9d, %r9
    shl $3, %r9
    lea (%r9,%r9,2), %rax
    add %r8, %rax
    vmovupd 0(%r8), %ymm4
    vmovupd 32(%r8), %ymm5
    vmovupd 0(%r8,%r9,1), %ymm6
    vmovupd 32(%r8,%r9,1), %ymm7
    vmovupd 0(%r8,%r9,2), %ymm8
    vmovupd 32(%r8,%r9,2), %ymm9
    vmovupd 0(%rax), %ymm10
    vmovupd 32(%rax), %ymm11
    vmovupd 0(%rax,%r9,1), %ymm12
    vmovupd 32(%rax,%r9,1), %ymm13
    vmovupd 0(%rax,%r9,2), %ymm14
    vmovupd 32(%rax,%r9,2), %ymm15
    add $128, %rsi
    mov %edi, %r11d
    shr $2, %r11d
    and $3, %edi

.macro JLC_NR8_U4_BODY
    vmovupd (%rdx), %ymm0
    vmovupd 32(%rdx), %ymm1
    vbroadcastsd -128(%rsi), %ymm2
    vbroadcastsd -120(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm4
    vfmadd231pd %ymm1, %ymm2, %ymm5
    vbroadcastsd -112(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm6
    vfmadd231pd %ymm1, %ymm3, %ymm7
    vbroadcastsd -104(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm8
    vfmadd231pd %ymm1, %ymm2, %ymm9
    vbroadcastsd -96(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm10
    vfmadd231pd %ymm1, %ymm3, %ymm11
    vbroadcastsd -88(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm12
    vfmadd231pd %ymm1, %ymm2, %ymm13
    vfmadd231pd %ymm0, %ymm3, %ymm14
    vmovupd 64(%rdx), %ymm0
    vfmadd231pd %ymm1, %ymm3, %ymm15
    vmovupd 96(%rdx), %ymm1
    vbroadcastsd -80(%rsi), %ymm2
    vbroadcastsd -72(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm4
    vfmadd231pd %ymm1, %ymm2, %ymm5
    vbroadcastsd -64(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm6
    vfmadd231pd %ymm1, %ymm3, %ymm7
    vbroadcastsd -56(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm8
    vfmadd231pd %ymm1, %ymm2, %ymm9
    vbroadcastsd -48(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm10
    vfmadd231pd %ymm1, %ymm3, %ymm11
    vbroadcastsd -40(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm12
    vfmadd231pd %ymm1, %ymm2, %ymm13
    vfmadd231pd %ymm0, %ymm3, %ymm14
    vmovupd 128(%rdx), %ymm0
    vfmadd231pd %ymm1, %ymm3, %ymm15
    vmovupd 160(%rdx), %ymm1
    vbroadcastsd -32(%rsi), %ymm2
    vbroadcastsd -24(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm4
    vfmadd231pd %ymm1, %ymm2, %ymm5
    vbroadcastsd -16(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm6
    vfmadd231pd %ymm1, %ymm3, %ymm7
    vbroadcastsd -8(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm8
    vfmadd231pd %ymm1, %ymm2, %ymm9
    vbroadcastsd 0(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm10
    vfmadd231pd %ymm1, %ymm3, %ymm11
    vbroadcastsd 8(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm12
    vfmadd231pd %ymm1, %ymm2, %ymm13
    vfmadd231pd %ymm0, %ymm3, %ymm14
    vmovupd 192(%rdx), %ymm0
    vfmadd231pd %ymm1, %ymm3, %ymm15
    vmovupd 224(%rdx), %ymm1
    vbroadcastsd 16(%rsi), %ymm2
    vbroadcastsd 24(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm4
    vfmadd231pd %ymm1, %ymm2, %ymm5
    vbroadcastsd 32(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm6
    vfmadd231pd %ymm1, %ymm3, %ymm7
    vbroadcastsd 40(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm8
    vfmadd231pd %ymm1, %ymm2, %ymm9
    vbroadcastsd 48(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm10
    vfmadd231pd %ymm1, %ymm3, %ymm11
    vbroadcastsd 56(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm12
    vfmadd231pd %ymm1, %ymm2, %ymm13
    vfmadd231pd %ymm0, %ymm3, %ymm14
    vfmadd231pd %ymm1, %ymm3, %ymm15
    add $192, %rsi
    add $256, %rdx
.endm

    test %r11d, %r11d
    jz 4f
#if JLC_NR8_GROUP4
    /* One target, K+4: the address is valid iff another logical K exists. */
    mov %r11d, %r10d
    test %edi, %edi
    jnz 7f
    dec %r10d
7:
    test %r10d, %r10d
    jz 3f
    sub %r10d, %r11d
    .p2align 5
1:
    prefetcht0 256(%rdx)
    JLC_NR8_U4_BODY
    dec %r10d
    jnz 1b
3:
    test %r11d, %r11d
    jz 4f
    .p2align 5
2:
    JLC_NR8_U4_BODY
    dec %r11d
    jnz 2b
#elif JLC_NR8_PREFETCH_DISTANCE > 0
    /* floor((K-D)/4): peel the final group when K%4 is smaller than D. */
    mov %r11d, %r10d
    cmp $JLC_NR8_PREFETCH_DISTANCE, %edi
    jae 7f
    dec %r10d
7:
    test %r10d, %r10d
    jz 3f
    sub %r10d, %r11d
    .p2align 5
1:
    prefetcht0 (JLC_NR8_PREFETCH_DISTANCE * 64)(%rdx)
    prefetcht0 (JLC_NR8_PREFETCH_DISTANCE * 64 + 64)(%rdx)
    prefetcht0 (JLC_NR8_PREFETCH_DISTANCE * 64 + 128)(%rdx)
    prefetcht0 (JLC_NR8_PREFETCH_DISTANCE * 64 + 192)(%rdx)
    JLC_NR8_U4_BODY
    dec %r10d
    jnz 1b
3:
    test %r11d, %r11d
    jz 4f
    .p2align 5
2:
    JLC_NR8_U4_BODY
    dec %r11d
    jnz 2b
#else
    .p2align 5
1:
    JLC_NR8_U4_BODY
    dec %r11d
    jnz 1b
#endif
4:
    test %edi, %edi
    jz 6f
5:
    vmovupd (%rdx), %ymm0
    vmovupd 32(%rdx), %ymm1
    vbroadcastsd -128(%rsi), %ymm2
    vbroadcastsd -120(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm4
    vfmadd231pd %ymm1, %ymm2, %ymm5
    vbroadcastsd -112(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm6
    vfmadd231pd %ymm1, %ymm3, %ymm7
    vbroadcastsd -104(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm8
    vfmadd231pd %ymm1, %ymm2, %ymm9
    vbroadcastsd -96(%rsi), %ymm2
    vfmadd231pd %ymm0, %ymm3, %ymm10
    vfmadd231pd %ymm1, %ymm3, %ymm11
    vbroadcastsd -88(%rsi), %ymm3
    vfmadd231pd %ymm0, %ymm2, %ymm12
    vfmadd231pd %ymm1, %ymm2, %ymm13
    vfmadd231pd %ymm0, %ymm3, %ymm14
    vfmadd231pd %ymm1, %ymm3, %ymm15
    add $48, %rsi
    add $64, %rdx
    dec %edi
    jnz 5b
6:
    vmovupd %ymm4, 0(%r8)
    vmovupd %ymm5, 32(%r8)
    vmovupd %ymm6, 0(%r8,%r9,1)
    vmovupd %ymm7, 32(%r8,%r9,1)
    vmovupd %ymm8, 0(%r8,%r9,2)
    vmovupd %ymm9, 32(%r8,%r9,2)
    vmovupd %ymm10, 0(%rax)
    vmovupd %ymm11, 32(%rax)
    vmovupd %ymm12, 0(%rax,%r9,1)
    vmovupd %ymm13, 32(%rax,%r9,1)
    vmovupd %ymm14, 0(%rax,%r9,2)
    vmovupd %ymm15, 32(%rax,%r9,2)
    vzeroupper
    ret
.cfi_endproc
.size JLC_NR8_FUNCTION, .-JLC_NR8_FUNCTION
.purgem JLC_NR8_U4_BODY
#undef JLC_NR8_GROUP4
