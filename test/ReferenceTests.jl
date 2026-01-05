# Copyright (c) 2026 by the FastAC.jl contributors
# Copyright (c) 2019 by Amir Said (said@ieee.org) &
#                       William A. Pearlman (pearlw@ecse.rpi.edu)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

module ReferenceTests

export reference_tests

using FastAC, Test

const SIMUL_TESTS::UInt32 = 1_000_000
const MIN_PROBABILITY::Float64 = 1e-4

function reference_tests(alphabet_symbols, test_cycles = 10)
  2 <= alphabet_symbols <= 500 || error("invalid number of data symbols")
  1 <= test_cycles <= 999 || error("invalid number of simulations")
  if alphabet_symbols == 2
    binary_benchmark(test_cycles)
  else
    general_benchmark(alphabet_symbols, test_cycles)
  end
end

mutable struct RandomGenerator
  s1::UInt32
  s2::UInt32
  s3::UInt32

  RandomGenerator(seed::UInt32 = zero(UInt32)) = set_seed!(new(UInt32.((0, 0, 0))...), seed)
end

function set_seed!(this::RandomGenerator, seed::UInt32)
  this.s1 = iszero(seed) ? 0x147AE11 : seed & 0xFFFFFFF
  this.s2 = xor(this.s1, 0xFFFFF07)
  this.s3 = xor(this.s1, 0xF03CD2F)
  this
end

@inline function get_word!(this::RandomGenerator)
  # "Taus88" generator with period (2^31 - 1) * (2^29 - 1) * (2^28 - 1)
  b  = xor((this.s1 << 13), this.s1)   >> 19
  this.s1 = xor((this.s1 & 0xFFFFFFFE) << 12, b)
  b  = xor((this.s2 <<  2), this.s2)   >> 25
  this.s2 = xor((this.s2 & 0xFFFFFFF8) <<  4, b)
  b  = xor((this.s3 <<  3), this.s3)   >> 11
  this.s3 = xor((this.s3 & 0xFFFFFFF0) << 17, b)
  xor(xor(this.s1, this.s2), this.s3)
end

mutable struct RandomBitSource
  generator::RandomGenerator
  threshold::UInt32
  ent::Float64
  prob_0::Float64

  RandomBitSource() = new(RandomGenerator(), zero(UInt32), 1.0, 0.5)
end

set_seed!(this::RandomBitSource, seed) = (set_seed!(this.generator, seed); this)
entropy(this::RandomBitSource) = this.ent
symbol_0_probability(this::RandomBitSource) = this.prob_0
symbol_1_probability(this::RandomBitSource) = 1 - this.prob_0

function set_probability_0!(this::RandomBitSource, p0::Float64)::Float64
  MIN_PROBABILITY <= p0 <= 1 - MIN_PROBABILITY || error("invalid random bit probability")
  this.prob_0 = p0
  this.threshold = floor(UInt32, p0 * 0xFFFFFFFF)
  this.ent = ((p0 - 1.0) * log(1 - p0) - p0 * log(p0)) / log(2)
end

function set_entropy!(this::RandomBitSource, entropy::Float64)
  0.0001 <= entropy <= 1.0 || error("invalid random bit entropy")
  h = entropy * log(2.0)
  p = 0.5 * entropy * entropy
  for _ in 1:8
    lp1 = log(1.0 - p)
    lp2 = lp1 - log(p)
    d = h + lp1 - p * lp2
    abs(d) < 1e-12 && break
    p += d / lp2
  end
  set_probability_0!(this, p)
  this
end

shuffle_probabilities!(this::RandomBitSource) =
  (get_word!(this.generator) > 0x80000000 && set_probability_0!(this, 1 - this.prob_0); this)

get_bit!(this::RandomBitSource) = UInt32(get_word!(this.generator) > this.threshold)

function binary_benchmark(num_cycles)

  # set simulation parameters
  num_simulations = 10
  entropy = 0.1
  entropy_increment = 0.1

  # variables
  src = RandomBitSource()
  codec = ArithmeticCodec(SIMUL_TESTS >> 2)
  static_model = StaticBitModel()
  adaptive_model = AdaptiveBitModel()

  source_bits = zeros(UInt8, SIMUL_TESTS)
  decoded_bits = zeros(UInt8, SIMUL_TESTS)

  for simul in 1:num_simulations
    for pass in 1:2

      set_entropy!(src, entropy)
      set_seed!(src, UInt32(1839304 + 2017 * (simul - 1)))

      for _ in 1:num_cycles

        # fill bit buffer
        shuffle_probabilities!(src)
        foreach(ind -> source_bits[ind] = get_bit!(src), eachindex(source_bits))
        pass == 1 && @info "random data" entropy symbol_1_probability(src) (sum(source_bits)/length(source_bits))

        if pass == 1
          FastAC.set_probability_0!(static_model, symbol_0_probability(src))

          # encode bit buffer
          start_encoder!(codec)
          foreach(bit -> encode!(codec, UInt32(bit), static_model), source_bits)
          code_bits = 8 * stop_encoder!(codec)

          # decode bit buffer
          start_decoder!(codec)
          foreach(ind -> decoded_bits[ind] = decode!(codec, static_model), eachindex(decoded_bits))
          stop_decoder!(codec)
        else
          @warn "skipping adaptive bit model test"
          continue
        end
      end
      @test source_bits == decoded_bits
    end
    entropy += entropy_increment
  end
end

end # module ReferenceTests
