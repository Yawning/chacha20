#!/usr/bin/env python3
#
# To the extent possible under law, Yawning Angel has waived all copyright
# and related or neighboring rights to chacha20, using the Creative
# Commons "CC0" public domain dedication. See LICENSE or
# <http://creativecommons.org/publicdomain/zero/1.0/> for full details.

#
# cgo sucks.  Plan 9 assembly sucks.  Real languages have SIMD intrinsics.
# The least terrible/retarded option is to use a Python code generator, so
# that's what I did.
#
# Code based on Ted Krovetz's vec128 C implementation, with corrections
# to use a 64 bit counter instead of 32 bit, and to allow unaligned input and
# output pointers.
#
# Dependencies: https://github.com/Maratyszcza/PeachPy
#
# python3 -m peachpy.x86_64 -mabi=goasm -S -o chacha20_amd64.s chacha20_amd64.py
#

from peachpy import *
from peachpy.x86_64 import *

x = Argument(ptr(uint32_t))
inp = Argument(ptr(const_uint8_t))
outp = Argument(ptr(uint8_t))
nrBlocks = Argument(ptr(size_t))

def DQRoundVectors_sse2(tmp, a, b, c, d):
    # a += b; d ^= a; d = ROTW16(d);
    PADDD(a, b)
    PXOR(d, a)
    MOVDQA(tmp, d)
    PSLLD(tmp, 16)
    PSRLD(d, 16)
    PXOR(d, tmp)

    # c += d; b ^= c; b = ROTW12(b);
    PADDD(c, d)
    PXOR(b, c)
    MOVDQA(tmp, b)
    PSLLD(tmp, 12)
    PSRLD(b, 20)
    PXOR(b, tmp)

    # a += b; d ^= a; d = ROTW8(d);
    PADDD(a, b)
    PXOR(d, a)
    MOVDQA(tmp, d)
    PSLLD(tmp, 8)
    PSRLD(d, 24)
    PXOR(d, tmp)

    # c += d; b ^= c; b = ROTW7(b)
    PADDD(c, d)
    PXOR(b, c)
    MOVDQA(tmp, b)
    PSLLD(tmp, 7)
    PSRLD(b, 25)
    PXOR(b, tmp)

    # b = ROTV1(b); c = ROTV2(c);  d = ROTV3(d);
    PSHUFD(b, b, 0x39)
    PSHUFD(c, c, 0x4e)
    PSHUFD(d, d, 0x93)

    # a += b; d ^= a; d = ROTW16(d);
    PADDD(a, b)
    PXOR(d, a)
    MOVDQA(tmp, d)
    PSLLD(tmp, 16)
    PSRLD(d, 16)
    PXOR(d, tmp)

    # c += d; b ^= c; b = ROTW12(b);
    PADDD(c, d)
    PXOR(b, c)
    MOVDQA(tmp, b)
    PSLLD(tmp, 12)
    PSRLD(b, 20)
    PXOR(b, tmp)

    # a += b; d ^= a; d = ROTW8(d);
    PADDD(a, b)
    PXOR(d, a)
    MOVDQA(tmp, d)
    PSLLD(tmp, 8)
    PSRLD(d, 24)
    PXOR(d, tmp)

    # c += d; b ^= c; b = ROTW7(b);
    PADDD(c, d)
    PXOR(b, c)
    MOVDQA(tmp, b)
    PSLLD(tmp, 7)
    PSRLD(b, 25)
    PXOR(b, tmp)

    # b = ROTV3(b); c = ROTV2(c); d = ROTV1(d);
    PSHUFD(b, b, 0x93)
    PSHUFD(c, c, 0x4e)
    PSHUFD(d, d, 0x39)

def WriteXor_sse2(tmp, inp, outp, d, v0, v1, v2, v3):
    MOVDQU(tmp, [inp+d])
    PXOR(tmp, v0)
    MOVDQU([outp+d], tmp)
    MOVDQU(tmp, [inp+d+16])
    PXOR(tmp, v1)
    MOVDQU([outp+d+16], tmp)
    MOVDQU(tmp, [inp+d+32])
    PXOR(tmp, v2)
    MOVDQU([outp+d+32], tmp)
    MOVDQU(tmp, [inp+d+48])
    PXOR(tmp, v3)
    MOVDQU([outp+d+48], tmp)

# SSE2 ChaCha20 (aka vec128).  Does not handle partial blocks, and will
# process 3 blocks at a time.  x (the ChaCha20 state) must be 16 byte aligned.
with Function("blocksAmd64SSE2", (x, inp, outp, nrBlocks)):
    reg_x = GeneralPurposeRegister64()
    reg_inp = GeneralPurposeRegister64()
    reg_outp = GeneralPurposeRegister64()
    reg_blocks = GeneralPurposeRegister64()

    LOAD.ARGUMENT(reg_x, x)
    LOAD.ARGUMENT(reg_inp, inp)
    LOAD.ARGUMENT(reg_outp, outp)
    LOAD.ARGUMENT(reg_blocks, nrBlocks)

    # Align the stack to a 16 byte boundary.
    reg_align_tmp = GeneralPurposeRegister64()
    MOV(reg_align_tmp, registers.rsp)
    AND(reg_align_tmp, 0x0f)
    reg_align = GeneralPurposeRegister64()
    MOV(reg_align, 0x10)
    SUB(reg_align, reg_align_tmp)
    SUB(registers.rsp, reg_align)

    # Build the counter increment vector on the stack, and allocate the scratch
    # space
    SUB(registers.rsp, 16+16)
    reg_tmp = GeneralPurposeRegister32()
    MOV(reg_tmp, 0x00000001)
    MOV([registers.rsp], reg_tmp)
    MOV(reg_tmp, 0x00000000)
    MOV([registers.rsp+4], reg_tmp)
    MOV([registers.rsp+8], reg_tmp)
    MOV([registers.rsp+12], reg_tmp)
    mem_one = [registers.rsp]     # (Stack) Counter increment vector
    mem_tmp0 = [registers.rsp+16] # (Stack) Scratch space.

    mem_s0 = [reg_x]           # (Memory) Cipher state [0..3]
    mem_s1 = [reg_x+16]        # (Memory) Cipher state [4..7]
    mem_s2 = [reg_x+32]        # (Memory) Cipher state [8..11]
    mem_s3 = [reg_x+48]        # (Memory) Cipher state [12..15]

    xmm_v0 = XMMRegister()
    xmm_v1 = XMMRegister()
    xmm_v2 = XMMRegister()
    xmm_v3 = XMMRegister()

    xmm_v4 = XMMRegister()
    xmm_v5 = XMMRegister()
    xmm_v6 = XMMRegister()
    xmm_v7 = XMMRegister()

    xmm_v8 = XMMRegister()
    xmm_v9 = XMMRegister()
    xmm_v10 = XMMRegister()
    xmm_v11 = XMMRegister()

    xmm_v12 = XMMRegister()
    xmm_v13 = XMMRegister()
    xmm_v14 = XMMRegister()
    xmm_v15 = XMMRegister()

    xmm_tmp = xmm_v12

    vector_loop = Loop()
    SUB(reg_blocks, 4)
    JB(vector_loop.end)
    with vector_loop:

        MOVDQA(xmm_v0, mem_s0)
        MOVDQA(xmm_v1, mem_s1)
        MOVDQA(xmm_v2, mem_s2)
        MOVDQA(xmm_v3, mem_s3)

        MOVDQA(xmm_v4, xmm_v0)
        MOVDQA(xmm_v5, xmm_v1)
        MOVDQA(xmm_v6, xmm_v2)
        MOVDQA(xmm_v7, xmm_v3)
        PADDQ(xmm_v7, mem_one)

        MOVDQA(xmm_v8, xmm_v0)
        MOVDQA(xmm_v9, xmm_v1)
        MOVDQA(xmm_v10, xmm_v2)
        MOVDQA(xmm_v11, xmm_v7)
        PADDQ(xmm_v11, mem_one)

        MOVDQA(xmm_v12, xmm_v0)
        MOVDQA(xmm_v13, xmm_v1)
        MOVDQA(xmm_v14, xmm_v2)
        MOVDQA(xmm_v15, xmm_v11)
        PADDQ(xmm_v15, mem_one)

        reg_rounds = GeneralPurposeRegister64()
        MOV(reg_rounds, 20)
        rounds_loop = Loop()
        with rounds_loop:
            # What was a nice set of macros is now a gigantic inlined blob
            # of code because the C code can have the compiler handle spilling
            # registers onto the stack and instruction scheduling, while
            # I can't.

            # a += b; d ^= a; d = ROTW16(d);
            PADDD(xmm_v0, xmm_v1)
            PADDD(xmm_v4, xmm_v5)
            PADDD(xmm_v8, xmm_v9)
            PADDD(xmm_v12, xmm_v13)
            PXOR(xmm_v3, xmm_v0)
            PXOR(xmm_v7, xmm_v4)
            PXOR(xmm_v11, xmm_v8)
            PXOR(xmm_v15, xmm_v12)

            MOVDQA(mem_tmp0, xmm_tmp) # Save

            MOVDQA(xmm_tmp, xmm_v3)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v3, 16)
            PXOR(xmm_v3, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v7)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v7, 16)
            PXOR(xmm_v7, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v11)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v11, 16)
            PXOR(xmm_v11, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v15)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v15, 16)
            PXOR(xmm_v15, xmm_tmp)

            # c += d; b ^= c; b = ROTW12(b);
            PADDD(xmm_v2, xmm_v3)
            PADDD(xmm_v6, xmm_v7)
            PADDD(xmm_v10, xmm_v11)
            PADDD(xmm_v14, xmm_v15)
            PXOR(xmm_v1, xmm_v2)
            PXOR(xmm_v5, xmm_v6)
            PXOR(xmm_v9, xmm_v10)
            PXOR(xmm_v13, xmm_v14)

            MOVDQA(xmm_tmp, xmm_v1)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v1, 20)
            PXOR(xmm_v1, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v5)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v5, 20)
            PXOR(xmm_v5, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v9)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v9, 20)
            PXOR(xmm_v9, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v13)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v13, 20)
            PXOR(xmm_v13, xmm_tmp)

            # a += b; d ^= a; d = ROTW8(d);
            MOVDQA(xmm_tmp, mem_tmp0) # Restore

            PADDD(xmm_v0, xmm_v1)
            PADDD(xmm_v4, xmm_v5)
            PADDD(xmm_v8, xmm_v9)
            PADDD(xmm_v12, xmm_v13)
            PXOR(xmm_v3, xmm_v0)
            PXOR(xmm_v7, xmm_v4)
            PXOR(xmm_v11, xmm_v8)
            PXOR(xmm_v15, xmm_v12)

            MOVDQA(mem_tmp0, xmm_tmp) # Save

            MOVDQA(xmm_tmp, xmm_v3)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v3, 24)
            PXOR(xmm_v3, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v7)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v7, 24)
            PXOR(xmm_v7, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v11)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v11, 24)
            PXOR(xmm_v11, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v15)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v15, 24)
            PXOR(xmm_v15, xmm_tmp)

            # c += d; b ^= c; b = ROTW7(b)
            PADDD(xmm_v2, xmm_v3)
            PADDD(xmm_v6, xmm_v7)
            PADDD(xmm_v10, xmm_v11)
            PADDD(xmm_v14, xmm_v15)
            PXOR(xmm_v1, xmm_v2)
            PXOR(xmm_v5, xmm_v6)
            PXOR(xmm_v9, xmm_v10)
            PXOR(xmm_v13, xmm_v14)

            MOVDQA(xmm_tmp, xmm_v1)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v1, 25)
            PXOR(xmm_v1, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v5)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v5, 25)
            PXOR(xmm_v5, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v9)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v9, 25)
            PXOR(xmm_v9, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v13)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v13, 25)
            PXOR(xmm_v13, xmm_tmp)

            # b = ROTV1(b); c = ROTV2(c);  d = ROTV3(d);
            PSHUFD(xmm_v1, xmm_v1, 0x39)
            PSHUFD(xmm_v5, xmm_v5, 0x39)
            PSHUFD(xmm_v9, xmm_v9, 0x39)
            PSHUFD(xmm_v13, xmm_v13, 0x39)
            PSHUFD(xmm_v2, xmm_v2, 0x4e)
            PSHUFD(xmm_v6, xmm_v6, 0x4e)
            PSHUFD(xmm_v10, xmm_v10, 0x4e)
            PSHUFD(xmm_v14, xmm_v14, 0x4e)
            PSHUFD(xmm_v3, xmm_v3, 0x93)
            PSHUFD(xmm_v7, xmm_v7, 0x93)
            PSHUFD(xmm_v11, xmm_v11, 0x93)
            PSHUFD(xmm_v15, xmm_v15, 0x93)

            MOVDQA(xmm_tmp, mem_tmp0) # Restore

            # a += b; d ^= a; d = ROTW16(d);
            PADDD(xmm_v0, xmm_v1)
            PADDD(xmm_v4, xmm_v5)
            PADDD(xmm_v8, xmm_v9)
            PADDD(xmm_v12, xmm_v13)
            PXOR(xmm_v3, xmm_v0)
            PXOR(xmm_v7, xmm_v4)
            PXOR(xmm_v11, xmm_v8)
            PXOR(xmm_v15, xmm_v12)

            MOVDQA(mem_tmp0, xmm_tmp) # Save

            MOVDQA(xmm_tmp, xmm_v3)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v3, 16)
            PXOR(xmm_v3, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v7)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v7, 16)
            PXOR(xmm_v7, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v11)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v11, 16)
            PXOR(xmm_v11, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v15)
            PSLLD(xmm_tmp, 16)
            PSRLD(xmm_v15, 16)
            PXOR(xmm_v15, xmm_tmp)

            # c += d; b ^= c; b = ROTW12(b);
            PADDD(xmm_v2, xmm_v3)
            PADDD(xmm_v6, xmm_v7)
            PADDD(xmm_v10, xmm_v11)
            PADDD(xmm_v14, xmm_v15)
            PXOR(xmm_v1, xmm_v2)
            PXOR(xmm_v5, xmm_v6)
            PXOR(xmm_v9, xmm_v10)
            PXOR(xmm_v13, xmm_v14)

            MOVDQA(xmm_tmp, xmm_v1)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v1, 20)
            PXOR(xmm_v1, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v5)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v5, 20)
            PXOR(xmm_v5, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v9)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v9, 20)
            PXOR(xmm_v9, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v13)
            PSLLD(xmm_tmp, 12)
            PSRLD(xmm_v13, 20)
            PXOR(xmm_v13, xmm_tmp)

            # a += b; d ^= a; d = ROTW8(d);
            MOVDQA(xmm_tmp, mem_tmp0) # Restore

            PADDD(xmm_v0, xmm_v1)
            PADDD(xmm_v4, xmm_v5)
            PADDD(xmm_v8, xmm_v9)
            PADDD(xmm_v12, xmm_v13)
            PXOR(xmm_v3, xmm_v0)
            PXOR(xmm_v7, xmm_v4)
            PXOR(xmm_v11, xmm_v8)
            PXOR(xmm_v15, xmm_v12)

            MOVDQA(mem_tmp0, xmm_tmp) # Save

            MOVDQA(xmm_tmp, xmm_v3)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v3, 24)
            PXOR(xmm_v3, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v7)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v7, 24)
            PXOR(xmm_v7, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v11)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v11, 24)
            PXOR(xmm_v11, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v15)
            PSLLD(xmm_tmp, 8)
            PSRLD(xmm_v15, 24)
            PXOR(xmm_v15, xmm_tmp)

            # c += d; b ^= c; b = ROTW7(b)
            PADDD(xmm_v2, xmm_v3)
            PADDD(xmm_v6, xmm_v7)
            PADDD(xmm_v10, xmm_v11)
            PADDD(xmm_v14, xmm_v15)
            PXOR(xmm_v1, xmm_v2)
            PXOR(xmm_v5, xmm_v6)
            PXOR(xmm_v9, xmm_v10)
            PXOR(xmm_v13, xmm_v14)

            MOVDQA(xmm_tmp, xmm_v1)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v1, 25)
            PXOR(xmm_v1, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v5)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v5, 25)
            PXOR(xmm_v5, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v9)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v9, 25)
            PXOR(xmm_v9, xmm_tmp)

            MOVDQA(xmm_tmp, xmm_v13)
            PSLLD(xmm_tmp, 7)
            PSRLD(xmm_v13, 25)
            PXOR(xmm_v13, xmm_tmp)

            # b = ROTV1(b); c = ROTV2(c);  d = ROTV3(d);
            PSHUFD(xmm_v1, xmm_v1, 0x93)
            PSHUFD(xmm_v5, xmm_v5, 0x93)
            PSHUFD(xmm_v9, xmm_v9, 0x93)
            PSHUFD(xmm_v13, xmm_v13, 0x93)
            PSHUFD(xmm_v2, xmm_v2, 0x4e)
            PSHUFD(xmm_v6, xmm_v6, 0x4e)
            PSHUFD(xmm_v10, xmm_v10, 0x4e)
            PSHUFD(xmm_v14, xmm_v14, 0x4e)
            PSHUFD(xmm_v3, xmm_v3, 0x39)
            PSHUFD(xmm_v7, xmm_v7, 0x39)
            PSHUFD(xmm_v11, xmm_v11, 0x39)
            PSHUFD(xmm_v15, xmm_v15, 0x39)

            MOVDQA(xmm_tmp, mem_tmp0) # Restore

            SUB(reg_rounds, 2)
            JNZ(rounds_loop.begin)

        MOVDQA(mem_tmp0, xmm_tmp)

        PADDD(xmm_v0, mem_s0)
        PADDD(xmm_v1, mem_s1)
        PADDD(xmm_v2, mem_s2)
        PADDD(xmm_v3, mem_s3)
        WriteXor_sse2(xmm_tmp, reg_inp, reg_outp, 0, xmm_v0, xmm_v1, xmm_v2, xmm_v3)
        MOVDQA(xmm_v3, mem_s3)
        PADDQ(xmm_v3, mem_one)

        PADDD(xmm_v4, mem_s0)
        PADDD(xmm_v5, mem_s1)
        PADDD(xmm_v6, mem_s2)
        PADDD(xmm_v7, xmm_v3)
        WriteXor_sse2(xmm_tmp, reg_inp, reg_outp, 64, xmm_v4, xmm_v5, xmm_v6, xmm_v7)
        PADDQ(xmm_v3, mem_one)

        PADDD(xmm_v8, mem_s0)
        PADDD(xmm_v9, mem_s1)
        PADDD(xmm_v10, mem_s2)
        PADDD(xmm_v11, xmm_v3)
        WriteXor_sse2(xmm_tmp, reg_inp, reg_outp, 128, xmm_v8, xmm_v9, xmm_v10, xmm_v11)
        PADDQ(xmm_v3, mem_one)

        MOVDQA(xmm_tmp, mem_tmp0)

        PADDD(xmm_v12, mem_s0)
        PADDD(xmm_v13, mem_s1)
        PADDD(xmm_v14, mem_s2)
        PADDD(xmm_v15, xmm_v3)
        WriteXor_sse2(xmm_v0, reg_inp, reg_outp, 192, xmm_v12, xmm_v13, xmm_v14, xmm_v15)
        PADDQ(xmm_v3, mem_one)

        MOVDQA(mem_s3, xmm_v3)

        ADD(reg_inp, 4 * 64)
        ADD(reg_outp, 4 * 64)

        SUB(reg_blocks, 4)
        JAE(vector_loop.begin)

    ADD(reg_blocks, 4)
    out = Label()
    JZ(out)

    # Since we're only doing 1 block at  a time, we can use registers for s0
    # and the counter vector now.
    xmm_s0 = xmm_v4
    xmm_s1 = xmm_v5
    xmm_s2 = xmm_v6
    xmm_s3 = xmm_v7
    xmm_one = xmm_v8
    MOVDQA(xmm_s0, mem_s0)
    MOVDQA(xmm_s1, mem_s1)
    MOVDQA(xmm_s2, mem_s2)
    MOVDQA(xmm_s3, mem_s3)
    MOVDQA(xmm_one, mem_one) # counter increment

    serial_loop = Loop()
    with serial_loop:
        MOVDQA(xmm_v0, xmm_s0)
        MOVDQA(xmm_v1, xmm_s1)
        MOVDQA(xmm_v2, xmm_s2)
        MOVDQA(xmm_v3, xmm_s3)

        reg_rounds = GeneralPurposeRegister64()
        MOV(reg_rounds, 20)
        rounds_loop = Loop()
        with rounds_loop:
            DQRoundVectors_sse2(xmm_tmp, xmm_v0, xmm_v1, xmm_v2, xmm_v3)
            SUB(reg_rounds, 2)
            JNZ(rounds_loop.begin)

        PADDD(xmm_v0, xmm_s0)
        PADDD(xmm_v1, xmm_s1)
        PADDD(xmm_v2, xmm_s2)
        PADDD(xmm_v3, xmm_s3)
        WriteXor_sse2(xmm_tmp, reg_inp, reg_outp, 0, xmm_v0, xmm_v1, xmm_v2, xmm_v3)
        PADDQ(xmm_s3, xmm_one)

        ADD(reg_inp, 64)
        ADD(reg_outp, 64)

        SUB(reg_blocks, 1)
        JNZ(serial_loop.begin)

    # Write back the updated counter.  Stoping at 2^70 bytes is the user's
    # problem, not mine.
    MOVDQA(mem_s3, xmm_s3)

    LABEL(out)

    # Paranoia, cleanse the scratch space.
    PXOR(xmm_v0, xmm_v0)
    MOVDQA(mem_tmp0, xmm_v0)

    ADD(registers.rsp, 16+16)
    ADD(registers.rsp, reg_align)

    RETURN()
