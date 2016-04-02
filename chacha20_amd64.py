#!/usr/bin/env python3
#
# To the extent possible under law, Yawning Angel has waived all copyright
# and related or neighboring rights to chacha20, using the Creative
# Commons "CC0" public domain dedication. See LICENSE or
# <http://creativecommons.org/publicdomain/zero/1.0/> for full details.

#
# Ok.  The first revision of this code started off as a cgo version of Ted
# Krovetz's vec128 ChaCha20 implementation, but cgo sucks because it carves
# off a separate stack (needed, but expensive), and worse, can allocate an OS
# thread because it treats all cgo invocations as system calls.
#
# For something like a low level cryptography routine, both of these behaviors
# are just unneccecary overhead, and the latter is totally fucking retarded.
#
# Since Golang doesn't have SIMD intrinsics, this means, that it's either
# "learn plan 9 assembly", or resort to more extreme measures like using a
# python code generator.  This obviously goes for the latter.
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

# Helper routines for the actual ChaCha round function.
#
# Note:
#   It's been pointed out by the PeachPy author that the tmp variable for a
#   scratch register is kind of silly, but I only have one XMM register that
#   can be used for scratch (everything else is used to store the cipher state
#   or output).

def RotV1(x):
    PSHUFD(x, x, 0x39)

def RotV2(x):
    PSHUFD(x, x, 0x4e)

def RotV3(x):
    PSHUFD(x, x, 0x93)

def RotW7(tmp, x):
    MOVDQA(tmp, x)
    PSLLD(tmp, 7)
    PSRLD(x, 25)
    PXOR(x, tmp)

def RotW8(tmp, x):
    MOVDQA(tmp, x)
    PSLLD(tmp, 8)
    PSRLD(x, 24)
    PXOR(x, tmp)

def RotW12(tmp, x):
    MOVDQA(tmp, x)
    PSLLD(tmp, 12)
    PSRLD(x, 20)
    PXOR(x, tmp)

def RotW16(tmp, x):
    MOVDQA(tmp, x)
    PSLLD(tmp, 16)
    PSRLD(x, 16)
    PXOR(x, tmp)

def DQRoundVectors(tmp, a, b, c, d):
    # a += b; d ^= a; d = ROTW16(d);
    PADDD(a, b)
    PXOR(d, a)
    RotW16(tmp, d)

    # c += d; b ^= c; b = ROTW12(b);
    PADDD(c, d)
    PXOR(b, c)
    RotW12(tmp, b)

    # a += b; d ^= a; d = ROTW8(d);
    PADDD(a, b)
    PXOR(d, a)
    RotW8(tmp, d)

    # c += d; b ^= c; b = ROTW7(b)
    PADDD(c, d)
    PXOR(b, c)
    RotW7(tmp, b)

    # b = ROTV1(b); c = ROTV2(c);  d = ROTV3(d);
    RotV1(b)
    RotV2(c)
    RotV3(d)

    # a += b; d ^= a; d = ROTW16(d);
    PADDD(a, b)
    PXOR(d, a)
    RotW16(tmp, d)

    # c += d; b ^= c; b = ROTW12(b);
    PADDD(c, d)
    PXOR(b, c)
    RotW12(tmp, b)

    # a += b; d ^= a; d = ROTW8(d);
    PADDD(a, b)
    PXOR(d, a)
    RotW8(tmp, d)

    # c += d; b ^= c; b = ROTW7(b);
    PADDD(c, d)
    PXOR(b, c)
    RotW7(tmp, b)

    # b = ROTV3(b); c = ROTV2(c); d = ROTV1(d);
    RotV3(b)
    RotV2(c)
    RotV1(d)

def WriteXor(tmp, inp, outp, d, v0, v1, v2, v3):
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

# SSE2 ChaCha20.  Does not handle partial blocks, and will process 3 blocks at
# a time.  x (the ChaCha20 state) must be 16 byte aligned.
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

    # Build the counter increment vector on the stack.
    SUB(registers.rsp, 16)
    reg_tmp = GeneralPurposeRegister32()
    MOV(reg_tmp, 0x00000001)
    MOV([registers.rsp], reg_tmp)
    MOV(reg_tmp, 0x00000000)
    MOV([registers.rsp+4], reg_tmp)
    MOV([registers.rsp+8], reg_tmp)
    MOV([registers.rsp+12], reg_tmp)
    mem_one = [registers.rsp]  # (Stack) Counter increment vector

    xmm_tmp = XMMRegister()    # The single scratch register
    mem_s0 = [reg_x]           # (Memory) Cipher state [0..3]
    xmm_s1 = XMMRegister()     # (Fixed Reg) Cipher state [4..7]
    MOVDQA(xmm_s1, [reg_x+16])
    xmm_s2 = XMMRegister()     # (Fixed Reg) Cipher state [8..11]
    MOVDQA(xmm_s2, [reg_x+32])
    xmm_s3 = XMMRegister()     # (Fixed Reg) Cipher state [12..15]
    MOVDQA(xmm_s3, [reg_x+48])

    vector_loop = Loop()
    serial_loop = Loop()

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

    SUB(reg_blocks, 3)
    JB(vector_loop.end)
    with vector_loop:
        MOVDQA(xmm_v0, mem_s0)
        MOVDQA(xmm_v1, xmm_s1)
        MOVDQA(xmm_v2, xmm_s2)
        MOVDQA(xmm_v3, xmm_s3)

        MOVDQA(xmm_v4, mem_s0)
        MOVDQA(xmm_v5, xmm_s1)
        MOVDQA(xmm_v6, xmm_s2)
        MOVDQA(xmm_v7, xmm_s3)
        PADDQ(xmm_v7, mem_one)

        MOVDQA(xmm_v8, mem_s0)
        MOVDQA(xmm_v9, xmm_s1)
        MOVDQA(xmm_v10, xmm_s2)
        MOVDQA(xmm_v11, xmm_v7)
        PADDQ(xmm_v11, mem_one)

        reg_rounds = GeneralPurposeRegister64()
        MOV(reg_rounds, 20)
        rounds_loop = Loop()
        with rounds_loop:
            DQRoundVectors(xmm_tmp, xmm_v0, xmm_v1, xmm_v2, xmm_v3)
            DQRoundVectors(xmm_tmp, xmm_v4, xmm_v5, xmm_v6, xmm_v7)
            DQRoundVectors(xmm_tmp, xmm_v8, xmm_v9, xmm_v10, xmm_v11)
            SUB(reg_rounds, 2)
            JNZ(rounds_loop.begin)

        PADDD(xmm_v0, mem_s0)
        PADDD(xmm_v1, xmm_s1)
        PADDD(xmm_v2, xmm_s2)
        PADDD(xmm_v3, xmm_s3)
        WriteXor(xmm_tmp, reg_inp, reg_outp, 0, xmm_v0, xmm_v1, xmm_v2, xmm_v3)
        PADDQ(xmm_s3, mem_one)

        PADDD(xmm_v4, mem_s0)
        PADDD(xmm_v5, xmm_s1)
        PADDD(xmm_v6, xmm_s2)
        PADDD(xmm_v7, xmm_s3)
        WriteXor(xmm_tmp, reg_inp, reg_outp, 64, xmm_v4, xmm_v5, xmm_v6, xmm_v7)
        PADDQ(xmm_s3, mem_one)

        PADDD(xmm_v8, mem_s0)
        PADDD(xmm_v9, xmm_s1)
        PADDD(xmm_v10, xmm_s2)
        PADDD(xmm_v11, xmm_s3)
        WriteXor(xmm_tmp, reg_inp, reg_outp, 128, xmm_v8, xmm_v9, xmm_v10, xmm_v11)
        PADDQ(xmm_s3, mem_one)

        ADD(reg_inp, 192)
        ADD(reg_outp, 192)

        SUB(reg_blocks, 3)
        JAE(vector_loop.begin)

    ADD(reg_blocks, 3)
    JZ(serial_loop.end)

    # Since we're only doing 1 block at  a time, we can use registers for s0
    # and the counter vector now.
    xmm_s0 = xmm_v4
    xmm_one = xmm_v5
    MOVDQA(xmm_s0, mem_s0)   # sigma
    MOVDQA(xmm_one, mem_one) # counter increment
    with serial_loop:
        MOVDQA(xmm_v0, xmm_s0)
        MOVDQA(xmm_v1, xmm_s1)
        MOVDQA(xmm_v2, xmm_s2)
        MOVDQA(xmm_v3, xmm_s3)

        reg_rounds = GeneralPurposeRegister64()
        MOV(reg_rounds, 20)
        rounds_loop = Loop()
        with rounds_loop:
            DQRoundVectors(xmm_tmp, xmm_v0, xmm_v1, xmm_v2, xmm_v3)
            SUB(reg_rounds, 2)
            JNZ(rounds_loop.begin)

        PADDD(xmm_v0, xmm_s0)
        PADDD(xmm_v1, xmm_s1)
        PADDD(xmm_v2, xmm_s2)
        PADDD(xmm_v3, xmm_s3)
        WriteXor(xmm_tmp, reg_inp, reg_outp, 0, xmm_v0, xmm_v1, xmm_v2, xmm_v3)
        PADDQ(xmm_s3, xmm_one)

        ADD(reg_inp, 64)
        ADD(reg_outp, 64)

        SUB(reg_blocks, 1)
        JNZ(serial_loop.begin)

    # Write back the updated counter.  Stoping at 2^70 bytes is the user's
    # problem, not mine.
    MOVDQA([reg_x+48], xmm_s3)

    ADD(registers.rsp, 16)
    ADD(registers.rsp, reg_align)

    RETURN()
