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

include("RandomData.jl")

using FastAC, Test
using .RandomData: RandomData

const SIMUL_TESTS::UInt32 = 1_000_000

function reference_benchmark(data_symbols, test_cycles = 10; verbose = false)
  2 <= data_symbols <= 500 || error("invalid number of data symbols")
  1 <= test_cycles <= 999 || error("invalid number of simulations")
  isbits = data_symbols == 2

  entropies = let
    hmax = log2(data_symbols) # max entropy (equal probability)
    isbits ? (0.1:0.1:1) : hmax <= 3 ? (0.2:0.2:hmax) : hmax <= 5 ? (0.5:0.25:hmax) : (1.0:0.5:hmax)
  end

  if isbits
    T = UInt8
    src = RandomData.RandomBitSource()
    static_model = StaticBitModel
    adaptive_model = AdaptiveBitModel
  else
    # could use UInt8 when <=256, but keeping it the same as the C++ code for now
    T = UInt16
    src = RandomData.RandomDataSource()
    static_model = StaticDataModel{data_symbols}
    adaptive_model = AdaptiveDataModel{data_symbols}
  end

  # assign memory for random test data
  source_data = zeros(T, SIMUL_TESTS)
  decoded_data = zeros(T, SIMUL_TESTS)

  set = "Static & adaptive " * (isbits ? "bit" : "data") * " models"
  @testset "$set ($data_symbols symbols)" begin
    @testset "entropy = $entropy" for (simul, entropy) in enumerate(entropies)
      for model_type in (static_model, adaptive_model)

        if isbits
          RandomData.set_entropy!(src, entropy)
          RandomData.set_seed!(src, UInt32(1839304 + 2017 * (simul - 1)))
        else
          RandomData.set_truncated_geometric!(src, UInt32(data_symbols), entropy)
          RandomData.set_seed!(src, UInt32(8315739 + 1031 * (simul - 1) + 11 * data_symbols))
        end

        code_bits = 0
        encode_time = 0.0
        decode_time = 0.0

        for _ in 1:test_cycles

          # fill data buffer
          RandomData.shuffle_probabilities!(src)
          get_data = isbits ? RandomData.get_bit! : RandomData.get_data!
          foreach(ind -> source_data[ind] = get_data(src), eachindex(source_data))

          # set up distribution of model
          model = if model_type <: StaticBitModel
            model_type(probability_0 = RandomData.symbol_0_probability(src))
            #continue
          elseif model_type <: StaticDataModel
            model_type(RandomData.probability(src))
          else
            model_type()
          end

          # encode data buffer
          enc = Encoder(IOBuffer(sizehint = SIMUL_TESTS << (isbits ? -2 : 1)))
          start_time = time()
          encode!(enc, source_data, model)
          test_cycles > 1 && (encode_time += time() - start_time)
          encoded_data = finalize!(enc)
          code_bits += 8 * length(encoded_data)

          # start with fresh distribution for decoding
          model_type <: Union{AdaptiveBitModel,AdaptiveDataModel} && reset!(model)

          # decode data buffer
          dec = Decoder(encoded_data)
          start_time = time()
          decode!(dec, decoded_data, model)
          test_cycles > 1 && (decode_time += time() - start_time)

          # check for decoding errors
          @test source_data == decoded_data
        end

        # reporting & check that the redundancy of the compressed data is <1%
        bit_rate = code_bits / (SIMUL_TESTS * test_cycles)
        redundancy = 100 * (bit_rate - entropy) / entropy
        encode_speed = SIMUL_TESTS * (test_cycles - 1) * 1e-6 / encode_time # M symbols / sec
        decode_speed = SIMUL_TESTS * (test_cycles - 1) * 1e-6 / decode_time # M symbols / sec
        probabilities = Tuple(floor(1000 * p) / 1000 for p in RandomData.probability(src))
        verbose && @info "$model_type with entropy = $entropy" probabilities bit_rate (
          redundancy) encode_speed decode_speed
        @test abs(redundancy) < 1
      end
    end
  end
end

function test_raw(n)
  bits = rand(Bool, n)
  bytes = rand(UInt8, n)
  shorts = rand(UInt16, n)
  vals = [rand(0:2047) for _ in 1:n] # 11-bit values

  enc = Encoder()
  for ind in 1:n
    encode!(enc, bits[ind])
    encode!(enc, bytes[ind])
    encode!(enc, shorts[ind])
    encode!(enc, vals[ind], 11)
  end
  encoded = finalize!(enc)

  decoded_bits = Bool[]
  decoded_bytes = UInt8[]
  decoded_shorts = UInt16[]
  decoded_vals = []

  dec = Decoder(encoded)
  for _ in 1:n
    push!(decoded_bits, decode!(dec, Bool))
    push!(decoded_bytes, decode!(dec, UInt8))
    push!(decoded_shorts, decode!(dec, UInt16))
    push!(decoded_vals, decode!(dec, 11))
  end

  @test bits == decoded_bits
  @test bytes == decoded_bytes
  @test shorts == decoded_shorts
  @test vals == decoded_vals

  # redundancy should be very small since there is no randomness
  expected_bits = n * (1 + 8 + 16 + 11)
  actual_bits = length(encoded) * 8
  redundancy = (actual_bits - expected_bits) / expected_bits * 100
  @test redundancy < 0.1
end

@testset "FastAC.jl Tests" verbose = true begin
  @testset "Tests from C++ reference code" verbose=true begin

    cycles = 2

    # test binary compression (two symbols)
    reference_benchmark(2, cycles; verbose = true)

    # test data compression with <=16 symbols (table is not used)
    reference_benchmark(10, cycles; verbose = true)

    # test data compression with >16 symbols (table is used)
    reference_benchmark(20, cycles; verbose = true)
  end

  @testset "Encoding raw bits without model" begin
    test_raw(100_000)
  end
end
