// chacha20_amd64.go - AMD64 optimized chacha20.
//
// To the extent possible under law, Yawning Angel has waived all copyright
// and related or neighboring rights to chacha20, using the Creative
// Commons "CC0" public domain dedication. See LICENSE or
// <http://creativecommons.org/publicdomain/zero/1.0/> for full details.

// +build amd64,!gccgo,!appengine

package chacha20

import (
	"math"
)

func blocksAmd64SSE2(x *uint32, in, out *byte, nrBlocks uint)

func blocksAmd64(x *[stateSize]uint32, in []byte, out []byte, nrBlocks int, isIetf bool) {
	if isIetf {
		var totalBlocks uint64
		totalBlocks = uint64(x[8]) + uint64(nrBlocks)
		if totalBlocks > math.MaxUint32 {
			panic("chacha20: Exceeded keystream per nonce limit")
		}
	}

	if in == nil {
		for i := range out {
			out[i] = 0
		}
		in = out
	}

	blocksAmd64SSE2(&x[0], &in[0], &out[0], uint(nrBlocks))
}

func init() {
	blocksFn = blocksAmd64
	usingVectors = true
}
