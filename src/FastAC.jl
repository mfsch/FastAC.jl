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

export StaticBitModel, set_probability_0!
export StaticDataModel, model_symbols, set_distribution!
export AdaptiveBitModel, reset!
export AdaptiveDataModel, model_symbols, reset!, set_alphabet!
export Encoder, Decoder, finalize!,
  read_from_file!, write_to_file!, put_bit!,
  get_bit!, put_bits!, get_bits!, encode!, decode!


"""
    reset!(model)

Reset the probability distribution of an adaptive model to equal probabilities
for all data symbols.
"""
function reset! end

# length bits discarded before mult. for adaptive models
const BM_LENGTH_SHIFT = 13 # maximum values for binary models
const DM_LENGTH_SHIFT = 15 # maximum values for general models
const DEBUG = true

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Static model for binary data
"""
mutable struct StaticBitModel

  bit_0_prob::UInt32

  StaticBitModel() = new(one(UInt32) << (BM_LENGTH_SHIFT - 1))

end

"""
Set probability of symbol '0'
"""
function set_probability_0!(this::StaticBitModel, p0::Float64)
  0.0001 <= p0 <= 0.9999 || error("invalid bit probability")
  this.bit_0_prob = floor(UInt32, p0 * (1 << BM_LENGTH_SHIFT))
  this
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Static model for general data
"""
mutable struct StaticDataModel

  distribution::Vector{UInt32}
  decoder_table::Vector{UInt32}

  data_symbols::UInt32
  last_symbol::UInt32
  table_size::UInt32
  table_shift::UInt32

  StaticDataModel() = new(
    ntuple(_ -> UInt32[], 2)...,
    ntuple(_ -> zero(UInt32), 4)...,
  )

end

model_symbols(this::StaticDataModel)::UInt32 = this.data_symbols

function set_distribution!(
  this::StaticDataModel,
  number_of_symbols::UInt32,
  probability::Union{Nothing,Vector{Float64}} = nothing, # `nothing` means uniform
)
  2 <= number_of_symbols <= (1 << 11) || error("invalid number of data symbols")

  # assign memory for data model
  if this.data_symbols != number_of_symbols
    this.data_symbols = number_of_symbols
    this.last_symbol = this.data_symbols - 1

    # define size of table for fast decoding
    if this.data_symbols > 16
      table_bits = 3
      while this.data_symbols > (1 << (table_bits + 2))
        table_bits += 1
      end
      this.table_size = (1 << table_bits) + 4
      this.table_shift = DM_LENGTH_SHIFT - table_bits
      this.decoder_table = zeros(UInt32, this.table_size + 6)
    else # small alphabet: no table needed
      this.table_size = this.table_shift = 0
      this.decoder_table = UInt32[]
    end
    this.distribution = zeros(UInt32, this.data_symbols)
  end

  # compute cumulative distribution, decoder table
  s::UInt32 = 0
  sum::Float64 = 0.0
  p::Float64 = 1 / Float64(this.data_symbols)

  for k in 1:this.data_symbols
    if !isnothing(probability)
      p = probability[k]
    end
    0.0001 <= p <= 0.9999 || error("invalid symbol probability")
    this.distribution[k] = floor(UInt32, sum * (1 << DM_LENGTH_SHIFT))
    sum += p
    iszero(this.table_size) && continue
    w::UInt32 = this.distribution[k] >> this.table_shift
    while (s < w)
      s += 1
      this.decoder_table[s + 1] = k - 2
    end
  end

  if !iszero(this.table_size)
    this.decoder_table[1] = 0
    while s <= this.table_size
      s += 1
      this.decoder_table[s + 1] = this.data_symbols - 1
    end
  end

  0.9999 <= sum <= 1.0001 || error("invalid probabilities")
  this
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
Adaptive model for binary data
"""
mutable struct AdaptiveBitModel

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
mutable struct AdaptiveDataModel

  distribution::Vector{UInt32}
  symbol_count::Vector{UInt32}
  decoder_table::Vector{UInt32}

  total_count::UInt32
  update_cycle::UInt32
  symbols_until_update::UInt32

  data_symbols::UInt32
  last_symbol::UInt32
  table_size::UInt32
  table_shift::UInt32

  AdaptiveDataModel(number_of_symbols::UInt32 = zero(UInt32)) = set_alphabet!(new(
      ntuple(_ -> UInt32[], 3)...,
      ntuple(_ -> zero(UInt32), 3)...,
      ntuple(_ -> zero(UInt32), 4)...,
    ), number_of_symbols)

end

function model_symbols(this::AdaptiveDataModel)::UInt32
  this.data_symbols
end

function reset!(this::AdaptiveDataModel)
  iszero(this.data_symbols) && return this
  # restore probability estimates to uniform distribution
  this.total_count = 0
  this.update_cycle = this.data_symbols
  this.symbol_count .= 1
  update!(this, false)
  this.symbols_until_update = this.update_cycle = (this.data_symbols + UInt32(6)) >> 1
  this
end

function set_alphabet!(this::AdaptiveDataModel, number_of_symbols::UInt32)
  2 <= number_of_symbols <= (1 << 11) || error("invalid number of data symbols")

  # assign memory for data model
  if this.data_symbols != number_of_symbols
    this.data_symbols = number_of_symbols
    this.last_symbol = this.data_symbols - 1

    # define size of table for fast decoding
    if this.data_symbols > 16
      table_bits = 3
      while this.data_symbols > (one(UInt32) << (table_bits + 2))
        table_bits += 1
      end
      this.table_size = (UInt32(1) << table_bits) + UInt32(4)
      this.table_shift = DM_LENGTH_SHIFT - table_bits
      this.decoder_table = zeros(UInt32, this.table_size + 6)
    else # small alphabet: no table needed
      this.table_size = this.table_shift = 0
      this.decoder_table = UInt32[]
    end

    this.distribution = zeros(UInt32, this.data_symbols)
    this.symbol_count = zeros(UInt32, this.data_symbols)
  end

  reset!(this) # initialize model
end

function update!(this::AdaptiveDataModel, from_encoder::Bool)

  # halve counts when a threshold is reached
  if (this.total_count += this.update_cycle) > (one(UInt32) << DM_LENGTH_SHIFT)
    this.total_count = 0
    for n in 1:this.data_symbols
      this.total_count += (this.symbol_count[n] = (this.symbol_count[n] + 1) >> 1)
    end
  end

  # compute cumulative distribution, decoder table
  sum::UInt32 = 0
  s::UInt32 = 0

  scale::UInt32 = div(0x80000000, this.total_count)

  if from_encoder || iszero(this.table_size)
    for k in 1:this.data_symbols
      this.distribution[k] = (scale * sum) >> (31 - DM_LENGTH_SHIFT)
      sum += this.symbol_count[k]
    end
  else
    for k in 1:this.data_symbols
      this.distribution[k] = (scale * sum) >> (31 - DM_LENGTH_SHIFT)
      sum += this.symbol_count[k]
      w::UInt32 = this.distribution[k] >> this.table_shift
      while s < w
        s += 1
        this.decoder_table[s + 1] = UInt32(k) - UInt32(2)
      end
    end
    this.decoder_table[1] = 0
    while s <= this.table_size
      s += 1
      this.decoder_table[s + 1] = this.data_symbols - UInt32(1)
    end
  end

  # set frequency of model updates
  this.update_cycle = (5 * this.update_cycle) >> 2
  max_cycle::UInt32 = (this.data_symbols + 6) << 3
  if this.update_cycle > max_cycle
    this.update_cycle = max_cycle
  end
  this.symbols_until_update = this.update_cycle

  this
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

struct ArithmeticCodec end # TODO: remove

"""
Read code data, start decoder
"""
function read_from_file!(this::ArithmeticCodec, code_file::IOStream)
  error("not implemented")
  this
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

  T <: IOBuffer ? take!(seekstart(enc.stream)) : nothing
end

"""
Stop encoder, write code data
"""
function write_to_file!(this::ArithmeticCodec, code_file::IOStream)::UInt32
  error("not implemented")
  zero(UInt32)
end

function put_bit!(this::ArithmeticCodec, bit::UInt32)
  error("not implemented")
  this
end

function get_bit!(this::ArithmeticCodec)::UInt32
  error("not implemented")
  zero(UInt32)
end

function put_bits!(this::ArithmeticCodec, data::UInt32, number_of_bits::UInt32)
  error("not implemented")
  this
end

function get_bits!(this::ArithmeticCodec, number_of_bits::UInt32)::UInt32
  error("not implemented")
  zero(UInt32)
end

function encode!(enc::Encoder, bit::Integer, M::StaticBitModel)

  x::UInt32 = M.bit_0_prob * (enc.state.length >> BM_LENGTH_SHIFT) # product l x p0

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

function decode!(dec::Decoder, M::StaticBitModel)::Bool
  x::UInt32 = M.bit_0_prob * (dec.state.length >> BM_LENGTH_SHIFT) # product l x p0
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

function encode!(enc::Encoder, data::Integer, M::StaticDataModel)
  DEBUG && data >= M.data_symbols && error("invalid data symbol")

  init_base::UInt32 = enc.state.base

  # compute products
  if (data == M.last_symbol)
    x = M.distribution[data + 1] * (enc.state.length >> DM_LENGTH_SHIFT)
    enc.state.base   += x # update interval
    enc.state.length -= x # no product needed
  else
    x = M.distribution[data + 1] * (enc.state.length >>= DM_LENGTH_SHIFT)
    enc.state.base   += x # update interval
    enc.state.length  = M.distribution[data + 2] * enc.state.length - x
  end

  propagate_carry!(enc, init_base) # overflow = carry
  renormalize_interval!(enc)
end

function decode!(dec::Decoder, M::StaticDataModel)::UInt32
  y::UInt32 = dec.state.length

  if !isempty(M.decoder_table) # use table look-up for faster decoding

    dv::UInt32 = div(dec.state.value, (dec.state.length >>= DM_LENGTH_SHIFT))
    t::UInt32 = dv >> M.table_shift

    s::UInt32 = M.decoder_table[t + 1] # initial decision based on table look-up
    n::UInt32 = M.decoder_table[t + 2] + 1

    while n > s + 1 # finish with bisection search
      m::UInt32 = (s + n) >> 1
      if M.distribution[m + 1] > dv
        n = m
      else
        s = m
      end
    end

    # compute products
    x::UInt32 = M.distribution[s + 1] * dec.state.length
    if s != M.last_symbol
      y = M.distribution[s + 2] * dec.state.length
    end

  else # decode using only multiplications

    x = s = 0
    dec.state.length >>= DM_LENGTH_SHIFT
    m = (n = M.data_symbols) >> 1

    # decode via bisection search
    while true
      z::UInt32 = dec.state.length * M.distribution[m + 1]
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

function encode!(enc::Encoder, bit::Integer, M::AdaptiveBitModel)
  x::UInt32 = M.bit_0_prob * (enc.state.length >> BM_LENGTH_SHIFT) # product l x p0

  # update interval
  if iszero(bit)
    enc.state.length = x
    M.bit_0_count += 1
  else
    init_base::UInt32 = enc.state.base
    enc.state.base   += x
    enc.state.length -= x
    propagate_carry!(enc, init_base) # overflow = carry
  end

  renormalize_interval!(enc)
  iszero(M.bits_until_update -= 1) && update!(M) # periodic model update

  enc
end

function decode!(dec::Decoder, M::AdaptiveBitModel)::Bool
  x::UInt32 = M.bit_0_prob * (dec.state.length >> BM_LENGTH_SHIFT) # product l x p0
  bit = (dec.state.value >= x) # decision

  # update interval
  if bit
    dec.state.value  -= x
    dec.state.length -= x
  else
    dec.state.length = x
    M.bit_0_count += 1
  end

  renormalize_interval!(dec)
  iszero(M.bits_until_update -= 1) && update!(M) # periodic model update

  bit # return data bit value
end

function encode!(enc::Encoder, data::Integer, M::AdaptiveDataModel)
  DEBUG && (0 <= data < M.data_symbols || error("invalid data symbol"))

  init_base::UInt32 = enc.state.base

  # compute products
  if data == M.last_symbol
    x::UInt32 = M.distribution[data + 1] * (enc.state.length >> DM_LENGTH_SHIFT)
    enc.state.base   += x # update interval
    enc.state.length -= x # no product needed
  else
    x = M.distribution[data + 1] * (enc.state.length >>= DM_LENGTH_SHIFT)
    enc.state.base   += x # update interval
    enc.state.length  = M.distribution[data + 2] * enc.state.length - x
  end

  propagate_carry!(enc, init_base) # overflow = carry
  renormalize_interval!(enc)

  M.symbol_count[data + 1] += 1
  iszero(M.symbols_until_update -= 1) && update!(M, true) # periodic model update

  enc
end

function decode!(dec::Decoder, M::AdaptiveDataModel)::UInt32
  y::UInt32 = dec.state.length

  if !isempty(M.decoder_table) # use table look-up for faster decoding

    dv::UInt32 = div(dec.state.value, (dec.state.length >>= DM_LENGTH_SHIFT))
    t::UInt32 = dv >> M.table_shift

    # initial decision based on table look-up
    s::UInt32 = M.decoder_table[t + 1]
    n::UInt32 = M.decoder_table[t + 2] + 1

    while (n > s + 1) # finish with bisection search
      m::UInt32 = (s + n) >> 1
      if M.distribution[m + 1] > dv
        n = m
      else
        s = m
      end
    end

    # compute products
    x::UInt32 = M.distribution[s + 1] * dec.state.length
    if s != M.last_symbol
      y = M.distribution[s + 2] * dec.state.length
    end

  else # decode using only multiplications

    x = s = zero(UInt32)
    dec.state.length >>= DM_LENGTH_SHIFT
    m = (n = M.data_symbols) >> 1

    # decode via bisection search
    while true
      z::UInt32 = dec.state.length * M.distribution[m + 1]
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

  M.symbol_count[s + 1] += 1
  iszero(M.symbols_until_update -= 1) && update!(M, false) # periodic model update

  return s
end

function propagate_carry!(enc::Encoder, init_base)
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

function renormalize_interval!(dec::Decoder)
  while dec.state.length < AC_MIN_LENGTH
    # read least-significant byte
    lsb = eof(dec.stream) ? 0x0 : read(dec.stream, UInt8)
    dec.state.value = (dec.state.value << 8) | lsb
    dec.state.length <<= 8 # length multiplied by 256
  end
  dec
end

end # module FastAC
