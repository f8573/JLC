/* JLC full 6x8 kmajor6 consumer, Linux x86-64 SysV.
 * void f(int K, const double* A, const double* B, int packed_n,
 *        double* C, int ldc);
 * A[6*p+r] already includes alpha. C already includes beta and prior KC blocks.
 * Only natural double alignment is required; no readable padding is assumed.
 * ymm4..15 = row low/high accumulators, ymm0/1 = B, ymm2/3 = A broadcasts.
 * No prefetch, stack scratch, callee-saved register use, or reassociation.
 */
.text
.p2align 5
.globl jlc_kmajor6_6x8_u4_asm
.hidden jlc_kmajor6_6x8_u4_asm
.type jlc_kmajor6_6x8_u4_asm, @function
jlc_kmajor6_6x8_u4_asm:
.cfi_startproc
    movslq %ecx, %rcx
    shl $3, %rcx
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
    lea (%rcx,%rcx,2), %r10
    add $128, %rsi
    mov %edi, %r11d
    shr $2, %r11d
    and $3, %edi
    test %r11d, %r11d
    jz .Lremainder_check
    .p2align 5
.Lu4:
    vmovupd (%rdx), %ymm0
    vmovupd 32(%rdx), %ymm1
    # K0: interleave upcoming A broadcasts with current FMAs.
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
    vmovupd 0(%rdx,%rcx), %ymm0
    vfmadd231pd %ymm1, %ymm3, %ymm15
    vmovupd 32(%rdx,%rcx), %ymm1
    # K1: interleave upcoming A broadcasts with current FMAs.
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
    vmovupd 0(%rdx,%rcx,2), %ymm0
    vfmadd231pd %ymm1, %ymm3, %ymm15
    vmovupd 32(%rdx,%rcx,2), %ymm1
    # K2: interleave upcoming A broadcasts with current FMAs.
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
    vmovupd 0(%rdx,%r10), %ymm0
    vfmadd231pd %ymm1, %ymm3, %ymm15
    vmovupd 32(%rdx,%r10), %ymm1
    # K3: interleave upcoming A broadcasts with current FMAs.
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
    lea (%rdx,%rcx,4), %rdx
    dec %r11d
    jnz .Lu4
.Lremainder_check:
    test %edi, %edi
    jz .Lstore
.Lremainder:
    vmovupd (%rdx), %ymm0
    vmovupd 32(%rdx), %ymm1
    # K0: interleave upcoming A broadcasts with current FMAs.
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
    add %rcx, %rdx
    dec %edi
    jnz .Lremainder
.Lstore:
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
.size jlc_kmajor6_6x8_u4_asm, .-jlc_kmajor6_6x8_u4_asm
.section .note.GNU-stack,"",@progbits
