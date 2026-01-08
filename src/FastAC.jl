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

module FastAC

export Encoder, Decoder, finalize!, encode!, decode!
export StaticBitModel, StaticDataModel, AdaptiveBitModel, AdaptiveDataModel, reset!

using StaticArrays: MVector

"""
    reset!(model)

Reset the probability distribution of an adaptive model to equal probabilities
for all data symbols.
"""
function reset! end

# length bits discarded before mult. for adaptive models
const BM_LENGTH_SHIFT = 13 # maximum values for binary models
const DM_LENGTH_SHIFT = 15 # maximum values for general models

abstract type Model{N} end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Static model for binary data
"""
struct StaticBitModel <: Model{2}

  bit_0_prob::UInt32

  function StaticBitModel(; probability_0 = nothing, probability_1 = nothing)
    p0(::Nothing, ::Nothing) = nothing
    p0(p0::Real, ::Nothing) = p0
    p0(::Nothing, p1::Real) = 1 - p1
    p0(p0::Real, p1::Real) = p0 + p1 ≈ 1 ? p0 : error("")
    p0_real = p0(probability_0, probability_1)
    isnothing(p0_real) && return new(one(UInt32) << (BM_LENGTH_SHIFT - 1))
    0.0001 <= p0_real <= 0.9999 || error("invalid bit probability")
    new(floor(UInt32, p0_real * (1 << BM_LENGTH_SHIFT)))
  end
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Static model for general data
"""
struct StaticDataModel{N,D} <: Model{N}

  distribution::MVector{N,UInt32}
  decoder_table::MVector{D,UInt32}
  table_shift::Int

  function StaticDataModel{N}() where {N}
    2 <= N <= (1 << 11) || error("invalid number of data symbols")

    # define size of table for fast decoding
    D, table_shift = if N > 16
      table_bits = 3
      while N > (1 << (table_bits + 2))
        table_bits += 1
      end
      table_size = (1 << table_bits) + 4
      table_size + 6, DM_LENGTH_SHIFT - table_bits # why 6 extra values?
    else # small alphabet: no table needed
      0, 0
    end

    # assign memory for data model
    model = new{N,D}(zeros(MVector{N,UInt32}), zeros(MVector{D,UInt32}), table_shift)
    reset!(model)
  end
end

StaticDataModel(data_symbols::Integer) = StaticDataModel{Int(data_symbols)}()
StaticDataModel{N}(probability::Vector{Float64}) where {N} =
  reset!(StaticDataModel{N}(), probability)
StaticDataModel(probability::Vector{Float64}) =
  reset!(StaticDataModel{length(probability)}(), probability)

function reset!(
    model::StaticDataModel{N,D},
    probability::Union{Nothing,Vector{Float64}} = nothing, # `nothing` means uniform
  ) where {N,D}

  # compute cumulative distribution, decoder table
  s::UInt32 = 0
  sum::Float64 = 0.0
  p::Float64 = 1 / N

  for k in 1:N
    if !isnothing(probability)
      p = probability[k]
    end
    0.0001 <= p <= 0.9999 || error("invalid symbol probability")
    model.distribution[k] = floor(UInt32, sum * (1 << DM_LENGTH_SHIFT))
    sum += p
    iszero(D) && continue
    w::UInt32 = model.distribution[k] >> model.table_shift
    while (s < w)
      s += 1
      model.decoder_table[s + 1] = k - 2
    end
  end

  if !iszero(D)
    model.decoder_table[1] = 0
    while s <= D - 6 # 6 extra values in table (maybe not needed?)
      s += 1
      model.decoder_table[s + 1] = N - 1
    end
  end

  0.9999 <= sum <= 1.0001 || error("invalid probabilities")
  model
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Adaptive model for binary data
"""
mutable struct AdaptiveBitModel <: Model{2}

  update_cycle::UInt32
  bits_until_update::UInt32

  bit_0_prob::UInt32
  bit_0_count::UInt32
  bit_count::UInt32

  AdaptiveBitModel() = reset!(new(
    ntuple(_ -> zero(UInt32), 2)...,
    ntuple(_ -> zero(UInt32), 3)...,
  ))

end

function reset!(this::AdaptiveBitModel)
  # initialization to equiprobable model
  this.bit_0_count = 1
  this.bit_count   = 2
  this.bit_0_prob  = 1 << (BM_LENGTH_SHIFT - 1)
  this.update_cycle = this.bits_until_update = 4 # start with frequent updates
  this
end

function update!(this::AdaptiveBitModel)

  # halve counts when a threshold is reached
  if (this.bit_count += this.update_cycle) > (one(UInt32) << BM_LENGTH_SHIFT)
    this.bit_count = (this.bit_count + one(UInt32)) >> 1
    this.bit_0_count = (this.bit_0_count + one(UInt32)) >> 1
    this.bit_0_count == this.bit_count && (this.bit_count += one(UInt32))
  end

  # compute scaled bit 0 probability
  scale::UInt32 = div(0x80000000, this.bit_count)
  this.bit_0_prob = (this.bit_0_count * scale) >> (31 - BM_LENGTH_SHIFT)

  # set frequency of model updates
  this.update_cycle = (UInt32(5) * this.update_cycle) >> 2
  if this.update_cycle > 64
    this.update_cycle = 64
  end
  this.bits_until_update = this.update_cycle

  this
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Adaptive model for binary data
"""
mutable struct AdaptiveDataModel{N,D} <: Model{N}

  const distribution::MVector{N,UInt32}
  const symbol_count::MVector{N,UInt32}
  const decoder_table::MVector{D,UInt32}
  const table_shift::UInt32

  total_count::UInt32
  update_cycle::UInt32
  symbols_until_update::UInt32

  function AdaptiveDataModel{N}() where {N}
    2 <= N <= (1 << 11) || error("invalid number of data symbols")

    # define size of table for fast decoding
    D, table_shift = if N > 16
      table_bits = 3
      while N > (1 << (table_bits + 2))
        table_bits += 1
      end
      table_size = (1 << table_bits) + 4
      table_size + 6, DM_LENGTH_SHIFT - table_bits
    else # small alphabet: no table needed
      0, 0
    end

    distribution = zeros(MVector{N,UInt32})
    symbol_count = zeros(MVector{N,UInt32})
    decoder_table = zeros(MVector{D,UInt32})
    counts = (zero(UInt32), zero(UInt32), zero(UInt32))

    model = new{N,D}(distribution, symbol_count, decoder_table, table_shift, counts...)
    reset!(model) # initialize tables
  end
end

AdaptiveDataModel(data_symbols::Integer) = AdaptiveDataModel{Int(data_symbols)}()

function reset!(model::AdaptiveDataModel{N}) where {N}
  iszero(N) && return model
  # restore probability estimates to uniform distribution
  model.total_count = 0
  model.update_cycle = N
  model.symbol_count .= 1
  update!(model, false)
  model.symbols_until_update = model.update_cycle = (N + 6) >> 1
  model
end

function update!(model::AdaptiveDataModel{N,D}, from_encoder::Bool) where {N,D}

  # halve counts when a threshold is reached
  if (model.total_count += model.update_cycle) > (one(UInt32) << DM_LENGTH_SHIFT)
    model.total_count = 0
    for n in 1:N
      model.total_count += (model.symbol_count[n] = (model.symbol_count[n] + 1) >> 1)
    end
  end

  # compute cumulative distribution, decoder table
  sum::UInt32 = 0
  s::UInt32 = 0

  scale::UInt32 = div(0x80000000, model.total_count)

  if from_encoder || iszero(D)
    for k in 1:N
      model.distribution[k] = (scale * sum) >> (31 - DM_LENGTH_SHIFT)
      sum += model.symbol_count[k]
    end
  else
    for k in 1:N
      model.distribution[k] = (scale * sum) >> (31 - DM_LENGTH_SHIFT)
      sum += model.symbol_count[k]
      w::UInt32 = model.distribution[k] >> model.table_shift
      while s < w
        s += 1
        model.decoder_table[s + 1] = UInt32(k) - UInt32(2)
      end
    end
    model.decoder_table[1] = 0
    while s <= D - 6 # table has 6 extra values for some reason (necessary?)
      s += 1
      model.decoder_table[s + 1] = N - 1
    end
  end

  # set frequency of model updates
  model.update_cycle = min((5 * model.update_cycle) >> 2, (N + 3) << 3)
  model.symbols_until_update = model.update_cycle

  model
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

const AC_MIN_LENGTH::UInt32 = 0x01000000 # threshold for renormalization
const AC_MAX_LENGTH::UInt32 = 0xFFFFFFFF # maximum AC interval length

mutable struct CodecState
  base::UInt32
  value::UInt32
  length::UInt32
end

struct Encoder{T}
  stream::T
  state::CodecState
  Encoder(stream::T) where {T<:IO} =
    new{T}(stream, CodecState(zero(UInt32), zero(UInt32), AC_MAX_LENGTH))
end

Encoder() = Encoder(IOBuffer())
Encoder(filename::AbstractString) = Encoder(open(filename, "w"))

struct Decoder{T}
  stream::T
  state::CodecState
  function Decoder(stream::T) where {T<:IO}
    value = ntoh(read(stream, UInt32))
    new{T}(stream, CodecState(zero(UInt32), value, AC_MAX_LENGTH))
  end
end

Decoder(data::AbstractArray{UInt8}) = Decoder(IOBuffer(data))
Decoder(filename::AbstractString) = Decoder(open(filename, "r"))

"""
Read code data, start decoder
"""
function read_from_file!(dec::Decoder, code_file::IOStream)
  error("not implemented")
  this
end

"""
Stop encoder, write code data
"""
function write_to_file!(enc::Encoder, code_file::IOStream)::UInt32
  error("not implemented")
  zero(UInt32)
end

function finalize!(enc::Encoder{T}) where T
  state = enc.state

  init_base::UInt32 = state.base # done encoding: set final data bytes

  if state.length > 2 * AC_MIN_LENGTH
    state.base  += AC_MIN_LENGTH # base offset
    state.length = AC_MIN_LENGTH >> 1 # set new length for 1 more byte
  else
    state.base  += AC_MIN_LENGTH >> 1 # base offset
    state.length = AC_MIN_LENGTH >> 9 # set new length for 2 more bytes
  end

  propagate_carry!(enc, init_base) # overflow = carry
  renormalize_interval!(enc) # renormalization = output last bytes

  T <: IOBuffer ? take!(seekstart(enc.stream)) : stream
end

function encode!(enc::Encoder, data::T, number_of_bits::Integer = 8 * sizeof(T)) where {T<:Integer}
  @boundscheck begin
    1 <= number_of_bits <= 20 || error("invalid number of bits")
    data < (1 << number_of_bits) || error("invalid data")
  end
  init_base = enc.state.base
  enc.state.base += UInt32(data) * (enc.state.length >>= number_of_bits) # new interval base and length
  propagate_carry!(enc, init_base) # overflow = carry
  renormalize_interval!(enc)
end

function encode!(enc::Encoder, bit::Bool)
  enc.state.length >>= 1 # halve interval
  if bit
    init_base = enc.state.base
    enc.state.base += enc.state.length # move base
    propagate_carry!(enc, init_base) # overflow = carry
  end
  renormalize_interval!(enc)
end

function decode!(dec::Decoder, ::Type{Bool})::Bool
  dec.state.length >>= 1 # halve interval
  bit = (dec.state.value >= dec.state.length) # decode bit
  bit && (dec.state.value -= dec.state.length) # move base
  renormalize_interval!(dec)
  bit # return data bit value
end

function decode!(dec::Decoder, ::Type{UInt32}, number_of_bits::Integer)::UInt32
  @boundscheck 1 <= number_of_bits <= 20 || error("invalid number of bits")
  s::UInt32 = div(dec.state.value, (dec.state.length >>= number_of_bits)) # decode symbol, change length
  dec.state.value -= dec.state.length * s # update interval
  renormalize_interval!(dec)
  s
end

# allow reading bits into a specific integer type
decode!(dec::Decoder, ::Type{T}, number_of_bits::Integer = 8 * sizeof(T)) where {T<:Integer} =
  convert(T, decode!(dec, UInt32, number_of_bits))

# by default, return bits as an `Int`
decode!(dec::Decoder, number_of_bits::Integer) = decode!(dec, Int, number_of_bits)

# encode a whole vector of values
encode!(enc::Encoder, data::AbstractVector{<:Integer}, model) =
  (foreach(val -> encode!(enc, val, model), data); enc)

# decode a whole vector of values
decode!(dec::Decoder, data::AbstractVector{<:Integer}, model) =
  (foreach(ind -> (@inbounds data[ind] = decode!(dec, model)), eachindex(data)); data)

# decode a whole vector of values
decode!(dec::Decoder, count::Integer, model) =
  map(_ -> decode!(dec, model), 1:count)

function encode!(enc::Encoder, bit::Integer, model::StaticBitModel)
  x::UInt32 = model.bit_0_prob * (enc.state.length >> BM_LENGTH_SHIFT) # product l x p0

  # update interval
  if iszero(bit)
    enc.state.length = x
  else
    init_base::UInt32 = enc.state.base
    enc.state.base   += x
    enc.state.length -= x
    propagate_carry!(enc, init_base) # overflow = carry
  end

  renormalize_interval!(enc)
end

function decode!(dec::Decoder, model::StaticBitModel)::Bool
  x::UInt32 = model.bit_0_prob * (dec.state.length >> BM_LENGTH_SHIFT) # product l x p0
  bit = (dec.state.value >= x) # decision

  # update & shift interval
  if bit
    dec.state.value  -= x # shifted interval base = 0
    dec.state.length -= x
  else
    dec.state.length = x
  end

  renormalize_interval!(dec)

  bit # return data bit value
end

function encode!(enc::Encoder, data::Integer, model::StaticDataModel{N}) where {N}
  @boundscheck data in 0:N-1 || error("invalid data symbol")

  init_base::UInt32 = enc.state.base

  # compute products
  x::UInt32 = @inbounds model.distribution[data + 1]
  if data == N - 1
    x *= (enc.state.length >> DM_LENGTH_SHIFT)
    enc.state.base  += x # update interval
    enc.state.length -= x # no product needed
  else
    x *= (enc.state.length >>= DM_LENGTH_SHIFT) # also updates length!
    enc.state.base  += x # update interval
    enc.state.length  = @inbounds model.distribution[data + 2] * enc.state.length - x
  end

  propagate_carry!(enc, init_base) # overflow = carry
  renormalize_interval!(enc)
end

function decode!(dec::Decoder, model::StaticDataModel{N})::UInt32 where {N}
  y::UInt32 = dec.state.length

  if !isempty(model.decoder_table) # use table look-up for faster decoding

    dv::UInt32 = div(dec.state.value, (dec.state.length >>= DM_LENGTH_SHIFT))
    t::UInt32 = dv >> model.table_shift

    s::UInt32 = model.decoder_table[t + 1] # initial decision based on table look-up
    n::UInt32 = model.decoder_table[t + 2] + 1

    while n > s + 1 # finish with bisection search
      m::UInt32 = (s + n) >> 1
      if model.distribution[m + 1] > dv
        n = m
      else
        s = m
      end
    end

    # compute products
    x::UInt32 = model.distribution[s + 1] * dec.state.length
    if s != N - 1
      y = model.distribution[s + 2] * dec.state.length
    end

  else # decode using only multiplications

    x = s = 0
    dec.state.length >>= DM_LENGTH_SHIFT
    m = (n = N) >> 1

    # decode via bisection search
    while true
      z::UInt32 = dec.state.length * model.distribution[m + 1]
      if (z > dec.state.value) # value is smaller
        n = m
        y = z
      else # value is larger or equal
        s = m
        x = z
      end
      (m = (s + n) >> 1) != s || break
    end
  end

  # update interval
  dec.state.value -= x
  dec.state.length = y - x

  renormalize_interval!(dec)

  s
end

function encode!(enc::Encoder, bit::Integer, model::AdaptiveBitModel)
  x::UInt32 = model.bit_0_prob * (enc.state.length >> BM_LENGTH_SHIFT) # product l x p0

  # update interval
  if iszero(bit)
    enc.state.length = x
    model.bit_0_count += 1
  else
    init_base::UInt32 = enc.state.base
    enc.state.base   += x
    enc.state.length -= x
    propagate_carry!(enc, init_base) # overflow = carry
  end

  renormalize_interval!(enc)
  iszero(model.bits_until_update -= 1) && update!(model) # periodic model update

  enc
end

function decode!(dec::Decoder, model::AdaptiveBitModel)::Bool
  x::UInt32 = model.bit_0_prob * (dec.state.length >> BM_LENGTH_SHIFT) # product l x p0
  bit = (dec.state.value >= x) # decision

  # update interval
  if bit
    dec.state.value  -= x
    dec.state.length -= x
  else
    dec.state.length = x
    model.bit_0_count += 1
  end

  renormalize_interval!(dec)
  iszero(model.bits_until_update -= 1) && update!(model) # periodic model update

  bit # return data bit value
end

function encode!(enc::Encoder, data::Integer, model::AdaptiveDataModel{N}) where {N}
  @boundscheck data in 0:N-1 || error("invalid data symbol")

  init_base::UInt32 = enc.state.base

  # compute products
  x::UInt32 = @inbounds model.distribution[data + 1]
  if data == N - 1
    x *= (enc.state.length >> DM_LENGTH_SHIFT)
    enc.state.base   += x # update interval
    enc.state.length -= x # no product needed
  else
    x *= (enc.state.length >>= DM_LENGTH_SHIFT) # updates length too!
    enc.state.base   += x # update interval
    enc.state.length = @inbounds model.distribution[data + 2] * enc.state.length - x
  end

  propagate_carry!(enc, init_base) # overflow = carry
  renormalize_interval!(enc)

  @inbounds model.symbol_count[data + 1] += 1
  iszero(model.symbols_until_update -= 1) && update!(model, true) # periodic model update

  enc
end

function decode!(dec::Decoder, model::AdaptiveDataModel{N})::UInt32 where {N}
  y::UInt32 = dec.state.length

  if !isempty(model.decoder_table) # use table look-up for faster decoding

    dv::UInt32 = div(dec.state.value, (dec.state.length >>= DM_LENGTH_SHIFT))
    t::UInt32 = dv >> model.table_shift

    # initial decision based on table look-up
    s::UInt32 = model.decoder_table[t + 1]
    n::UInt32 = model.decoder_table[t + 2] + 1

    while (n > s + 1) # finish with bisection search
      m::UInt32 = (s + n) >> 1
      if model.distribution[m + 1] > dv
        n = m
      else
        s = m
      end
    end

    # compute products
    x::UInt32 = model.distribution[s + 1] * dec.state.length
    if s != N - 1
      y = model.distribution[s + 2] * dec.state.length
    end

  else # decode using only multiplications

    x = s = zero(UInt32)
    dec.state.length >>= DM_LENGTH_SHIFT
    m = (n = N) >> 1

    # decode via bisection search
    while true
      z::UInt32 = dec.state.length * model.distribution[m + 1]
      if (z > dec.state.value) # value is smaller
        n = m
        y = z
      else # value is larger or equal
        s = m
        x = z
      end
      (m = (s + n) >> 1) != s || break
    end
  end

  dec.state.value -= x # update interval
  dec.state.length = y - x

  renormalize_interval!(dec)

  model.symbol_count[s + 1] += 1
  iszero(model.symbols_until_update -= 1) && update!(model, false) # periodic model update

  return s
end

@inline function propagate_carry!(enc::Encoder, init_base)
  init_base > enc.state.base || return enc
  # carry propagation on compressed data buffer
  io = enc.stream
  mark(io)
  while read(skip(io, -1), UInt8) == 0xff
    write(skip(io, -1), 0x0)
    skip(io, -1)
  end
  final_val = read(skip(io, -1), UInt8)
  write(skip(io, -1), final_val + 0x1)
  reset(io)
  enc
end

@inline function renormalize_interval!(enc::Encoder)
  # output and discard top byte
  while enc.state.length < AC_MIN_LENGTH
    write(enc.stream, UInt8(enc.state.base >> 24))
    enc.state.base <<= 8
    enc.state.length <<= 8
  end
  enc
end

@inline function renormalize_interval!(dec::Decoder)
  while dec.state.length < AC_MIN_LENGTH
    # read least-significant byte
    lsb = eof(dec.stream) ? 0x0 : read(dec.stream, UInt8)
    dec.state.value = (dec.state.value << 8) | lsb
    dec.state.length <<= 8 # length multiplied by 256
  end
  dec
end

end # module FastAC
