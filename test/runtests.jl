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

  entropies = let
    hmax = log2(data_symbols) # max entropy (equal probability)
    hmax == 2 ? (0.1:0.1:1) : hmax <= 3 ? (0.2:0.2:hmax) : hmax <= 5 ? (0.5:0.25:hmax) : (1.0:0.5:hmax)
  end

  if data_symbols == 2
    T = UInt8
    src = RandomData.RandomBitSource()
    codec = ArithmeticCodec(SIMUL_TESTS >> 2)
    static_model = StaticBitModel()
    adaptive_model = AdaptiveBitModel()
  else
    # could use UInt8 when <=256, but keeping it the same as the C++ code for now
    T = UInt16
    src = RandomData.RandomDataSource()
    codec = ArithmeticCodec(SIMUL_TESTS << 1)
    static_model = StaticDataModel()
    adaptive_model = AdaptiveDataModel(UInt32(data_symbols))
  end

  # assign memory for random test data
  source_data = zeros(T, SIMUL_TESTS)
  decoded_data = zeros(T, SIMUL_TESTS)

  set = "Static & adaptive " * (data_symbols == 2 ? "bit" : "data") * " models"
  @testset "$set ($data_symbols symbols)" begin
    @testset "entropy = $entropy" for (simul, entropy) in enumerate(entropies)
      for model in (static_model, adaptive_model)

        if data_symbols == 2
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
          get_data = data_symbols == 2 ? RandomData.get_bit! : RandomData.get_data!
          foreach(ind -> source_data[ind] = get_data(src), eachindex(source_data))

          # set up distribution of model
          if model isa StaticBitModel
            FastAC.set_probability_0!(static_model, RandomData.symbol_0_probability(src))
          elseif model isa StaticDataModel
            FastAC.set_distribution!(model, UInt32(data_symbols), RandomData.probability(src))
          else
            reset!(model)
          end

          # encode data buffer
          start_encoder!(codec)
          start_time = time()
          foreach(x -> encode!(codec, UInt32(x), model), source_data)
          test_cycles > 1 && (encode_time += time() - start_time)
          code_bits += 8 * stop_encoder!(codec)

          # start with fresh distribution for decoding
          model isa Union{AdaptiveBitModel,AdaptiveDataModel} && reset!(model)

          # decode data buffer
          start_decoder!(codec)
          start_time = time()
          foreach(ind -> decoded_data[ind] = decode!(codec, model), eachindex(decoded_data))
          test_cycles > 1 && (decode_time += time() - start_time)
          stop_decoder!(codec)

          # check for decoding errors
          @test source_data == decoded_data
        end

        # reporting & check that the redundancy of the compressed data is <1%
        bit_rate = code_bits / (SIMUL_TESTS * test_cycles)
        redundancy = 100 * (bit_rate - entropy) / entropy
        encode_speed = SIMUL_TESTS * (test_cycles - 1) * 1e-6 / encode_time # M symbols / sec
        decode_speed = SIMUL_TESTS * (test_cycles - 1) * 1e-6 / decode_time # M symbols / sec
        probabilities = Tuple(floor(1000 * p) / 1000 for p in RandomData.probability(src))
        verbose && @info "$(typeof(model)) with entropy = $entropy" probabilities bit_rate redundancy (
          encode_speed) decode_speed
        @test abs(redundancy) < 1
      end
    end
  end
end

@testset "Tests from C++ reference code" verbose=true begin

  cycles = 2

  # test binary compression (two symbols)
  reference_benchmark(2, cycles; verbose = true)

  # test data compression with <=16 symbols (table is not used)
  reference_benchmark(10, cycles; verbose = true)

  # test data compression with >16 symbols (table is used)
  reference_benchmark(20, cycles; verbose = true)
end
