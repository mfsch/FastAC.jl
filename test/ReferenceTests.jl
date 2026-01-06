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

function get_uniform!(this::RandomGenerator) # open interval: 0 < r < 1
  WordToDouble::Float64 = 1 / (1 + Float64(0xFFFFFFFF))
  WordToDouble * (0.5 + Float64(get_word!(this)))
end

get_integer!(this::RandomGenerator, range::UInt32) = floor(UInt32, range * get_uniform!(this))

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

# Pseudo-random generator of data symbols with given probabilities
mutable struct RandomDataSource
  generator::RandomGenerator
  ent::Float64
  prob::Vector{Float64}
  symbols::UInt32
  dist::Vector{UInt32}
  low_bound::Vector{UInt32}

  RandomDataSource() = new(RandomGenerator(), 0.0, Float64[], 0, UInt32[], zeros(UInt32, 257))
end

entropy(this::RandomDataSource) = this.ent
probability(this::RandomDataSource) = this.prob
data_symbols(this::RandomDataSource) = this.symbols
set_seed!(this::RandomDataSource, seed) = (set_seed!(this.generator, seed); this)

function get_data!(this::RandomDataSource) # pseudo-random data with predefined distributions
  v::UInt32 = get_word!(this.generator)
  w::UInt32 = v >> 24
  u::UInt32 = this.low_bound[w + 1]
  n::UInt32 = this.low_bound[w + 2] + 1
  while n > u + 1
    m::UInt32 = (u + n) >> 1
    if this.dist[m + 1] < v
      u = m
    else
      n = m
    end
  end
  u
end

function shuffle_probabilities!(this::RandomDataSource)
  for n in (this.symbols - 1):-1:1
    m::UInt32 = get_integer!(this.generator, UInt32(n + 1))
    m == n && continue
    t = this.prob[m + 1]
    this.prob[m + 1] = this.prob[n + 1]
    this.prob[n + 1] = t
  end
  set_distribution!(this, this.symbols, this.prob)
end

function set_distribution!(this::RandomDataSource, dim::UInt32, probability::Vector{Float64})::Float64
  assign_memory!(this, dim)

  sum = this.ent = 0.0
  s::UInt32 = this.low_bound[1] = 0
  DoubleToWord::Float64 = 1.0 + Float64(0xFFFFFFFF)

  for n in 1:this.symbols
    p::Float64 = probability[n]
    p < MIN_PROBABILITY && error("invalid random source probability")
    this.prob[n] = p
    this.dist[n] = floor(UInt32, 0.49 + DoubleToWord * sum)
    w::UInt32 = this.dist[n] >> 24
    while s < w
      s += 1
      this.low_bound[s + 1] = UInt32(n) - UInt32(2)
    end
    sum += p
    this.ent -= p * log(p)
  end
  while s < 256
    s += 1
    this.low_bound[s + 1] = this.symbols - 1
  end

  abs(1 - sum) > 1e-4 && Error("invalid random source distribution")
  this.ent /= log(2)
end

# Class used by 'RandomDataSource' to find source parameters
mutable struct ZeroFinder
  phase::Int32
  iter::Int32
  x0::Float64
  y0::Float64
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  x::Float64

  ZeroFinder(first_x::Float64, second_x::Float64) = new(
    zero(Int32), zero(Int32),
    first_x, 0.0, second_x, 0.0, 0.0, 0.0,
    0.0)
end

function set_new_result!(this::ZeroFinder, y::Float64) # returns new test
  (this.iter += 1) > 30 && error("cannot find solution")

  if this.phase >= 2

    if y * this.y0 <= 0
      if this.phase == 2 || abs(this.y1) < abs(this.y2)
        this.x2 = this.x1
        this.y2 = this.y1
      end
      this.x1 = this.x
      this.y1 = y
    else
      if this.phase == 2 || abs(this.y0) < abs(this.y2)
        this.x2 = this.x0
        this.y2 = this.y0
      end
      this.x0 = this.x
      this.y0 = y
    end

    # interpolation y = [(x-x0)-f]/[g(x-x0)+h]
    if abs(this.y0) < abs(this.y1)
      r = this.y0 / this.y2
      c = this.x2 - this.x0
      s = this.y0 / this.y1
      d = this.x1 - this.x0
      this.x = this.x0 - (c * d * (s - r)) / (c * (1 - s) - d * (1 - r))
    else
      r = this.y1 / this.y2
      c = this.x2 - this.x1
      s = this.y1 / this.y0
      d = this.x0 - this.x1
      this.x = this.x1 - (c * d * ( s - r)) / (c * (1 - s) - d * (1 - r))
    end

    this.phase = 3;
    return this.x
  end

  this.iter > 8 && error("too many initial tests")

  if this.phase == 1

    if (y * this.y0 <= 0) # different signs: bracketed interval
      this.y1 = y
      this.phase = 2
      if abs(this.y0) < abs(this.y1) # regula falsi interpolation
        s = this.y0 / this.y1
        this.x = this.x0 - ((this.x1 - this.x0) * s) / (1 - s)
      else
        s = this.y1 / this.y0
        this.x = this.x1 - ((this.x0 - this.x1) * s) / (1 - s)
      end
    else # same sign: increase search interval
      this.x += this.x1 - this.x0
      this.x0 = this.x1
      this.y0 = y
      this.x1 = this.x
    end

    return this.x
  end

  this.y0 = y
  this.phase = 1

  this.x = this.x1
end


function set_truncated_geometric!(this::RandomDataSource, dim::UInt32, entropy::Float64)::Float64
  assign_memory!(this, dim)

  max_entropy = log(this.symbols) / log(2)
  mgr_prob = (dim - 1) * MIN_PROBABILITY
  min_entropy = ((mgr_prob - 1) * log(1 - mgr_prob) - mgr_prob * log(MIN_PROBABILITY)) * 1.2 / log(2)
  min_entropy < entropy <= max_entropy || error("invalid data source entropy")

  # find distribution with desired entropy
  ZF = ZeroFinder(0.0, 2.0)
  a = set_new_result!(ZF, max_entropy - entropy)

  for _ in 1:20
    ne = set_tg!(this, a) - entropy
    abs(ne) < 1e-5 && break
    a = set_new_result!(ZF, ne)
  end

  set_distribution!(this, this.symbols, this.prob)
  abs(this.ent - entropy) > 1e-4 && error("cannot set random source entropy")

  this.ent
end

function assign_memory!(this::RandomDataSource, dim::UInt32)
  this.symbols == dim && return this
  this.symbols = dim
  this.prob = zeros(dim)
  this.dist = zeros(UInt32, dim)
  this
end

function set_tg!(this::RandomDataSource, a::Float64)::Float64
  m = this.symbols

  s = if a > 1e-4
    (1 - exp(-a)) / (1 - exp(-a * m))
  else
    (2 - a) / (m * (2 - a * m))
  end

  r::Float64 = 0
  e::Float64 = 0

  for n in (this.symbols - 1):-1:0

    p::Float64 = (a * n > 30 ? 0 : s * exp(-a * n))

    if (p < MIN_PROBABILITY)
      r += MIN_PROBABILITY - p
      p = MIN_PROBABILITY
    elseif r > 0
      if r <= p - MIN_PROBABILITY
        p -= r
        r = 0
      else
        r -= p - MIN_PROBABILITY
        p = MIN_PROBABILITY
      end
    end
    this.prob[n + 1] = p
    e -= p * log(p)
  end

  return e / log(2)
end


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

        if pass == 1 # test static model

          FastAC.set_probability_0!(static_model, symbol_0_probability(src))

          # encode bit buffer
          start_encoder!(codec)
          foreach(bit -> encode!(codec, UInt32(bit), static_model), source_bits)
          code_bits = 8 * stop_encoder!(codec)

          # decode bit buffer
          start_decoder!(codec)
          foreach(ind -> decoded_bits[ind] = decode!(codec, static_model), eachindex(decoded_bits))
          stop_decoder!(codec)

        else # test adaptive model

          # encode bit buffer
          reset!(adaptive_model)
          start_encoder!(codec)
          foreach(bit -> encode!(codec, UInt32(bit), adaptive_model), source_bits)
          code_bits = 8 * stop_encoder!(codec)

          # decode bit buffer
          reset!(adaptive_model)
          start_decoder!(codec)
          foreach(ind -> decoded_bits[ind] = decode!(codec, adaptive_model), eachindex(decoded_bits))
          stop_decoder!(codec)

        end
      end

      # check for decoding errors
      @test source_bits == decoded_bits
    end

    entropy += entropy_increment
  end
end

function general_benchmark(data_symbols, num_cycles)

  # set simulation parameters
  if data_symbols <= 8
    entropy = 0.2;
    entropy_increment = 0.20;
  elseif data_symbols <= 32
    entropy = 0.5;
    entropy_increment = 0.25;
  else
    entropy = 1.0;
    entropy_increment = 0.50;
  end

  num_simulations = 1 + floor(Int, (log(data_symbols) / log(2) - entropy) / entropy_increment)

  # variables
  src = RandomDataSource()
  codec = ArithmeticCodec(SIMUL_TESTS << 1)
  static_model = StaticDataModel()
  adaptive_model = AdaptiveDataModel(UInt32(data_symbols))

  # assign memory for random test data
  source_data  = zeros(UInt16, SIMUL_TESTS)
  decoded_data = zeros(UInt16, SIMUL_TESTS)

  set_alphabet!(adaptive_model, UInt32(data_symbols))

  for simul in 1:num_simulations
    for pass in 1:2

      set_truncated_geometric!(src, UInt32(data_symbols), entropy)
      set_seed!(src, UInt32(8315739 + 1031 * (simul - 1) + 11 * data_symbols))

      for _ in 1:num_cycles

        shuffle_probabilities!(src)
        foreach(ind -> source_data[ind] = get_data!(src), eachindex(source_data))

        if pass == 1 # test static model

          FastAC.set_distribution!(static_model, UInt32(data_symbols), probability(src))

          # encode data buffer
          start_encoder!(codec)
          foreach(x -> encode!(codec, UInt32(x), static_model), source_data)
          code_bits = 8 * stop_encoder!(codec)

          # decode data buffer
          start_decoder!(codec)
          foreach(ind -> decoded_data[ind] = decode!(codec, static_model), eachindex(decoded_data))
          stop_decoder!(codec)

        else # test adaptive model

          reset!(adaptive_model)

          # encode data buffer
          start_encoder!(codec)
          foreach(x -> encode!(codec, UInt32(x), adaptive_model), source_data)
          code_bits = 8 * stop_encoder!(codec)

          reset!(adaptive_model)

          # decode data buffer
          start_decoder!(codec)
          foreach(ind -> decoded_data[ind] = decode!(codec, adaptive_model), eachindex(decoded_data))
          stop_decoder!(codec)
        end

        # check for decoding errors
        @test source_data == decoded_data
      end
    end

    entropy += entropy_increment;
  end
end

end # module ReferenceTests
