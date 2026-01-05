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

const DEBUG = true

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

export StaticBitModel, set_probability_0!

# Maximum values for binary models length bits discarded before mult. for adaptive models
const BM_LENGTH_SHIFT = 13

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

export StaticDataModel, model_symbols, set_distribution!

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

function model_symbols(this::StaticDataModel)::UInt32
  this.data_symbols
end

function set_distribution!(this::StaticDataModel, number_of_symbols::UInt32,
    probability::Union{Nothing,Vector{Float64}} = nothing) # `nothing` means uniform
  this
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

export AdaptiveBitModel, reset!

"""
Adaptive model for binary data
"""
mutable struct AdaptiveBitModel

  update_cycle::UInt32
  bits_until_update::UInt32

  bit_0_prob::UInt32
  bit_0_count::UInt32
  bit_count::UInt32

  AdaptiveBitModel() = new(
    ntuple(_ -> zero(UInt32), 2)...,
    ntuple(_ -> zero(UInt32), 3)...,
  )

end

"""
Reset to equiprobable model
"""
function reset!(this::AdaptiveBitModel)
  error("not implemented")
  this
end

function update!(this::AdaptiveBitModel)
  error("not implemented")
  this
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

export AdaptiveDataModel, model_symbols, reset!, set_alphabet!

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

  AdaptiveDataModel(number_of_symbols::UInt32 = zero(UInt32)) = new(
    ntuple(_ -> UInt32[], 3)...,
    ntuple(_ -> zero(UInt32), 3)...,
    ntuple(_ -> zero(UInt32), 4)...,
  )

end

function model_symbols(this::AdaptiveDataModel)::UInt32
  this.data_symbols
end

"""
Reset to equiprobable model
"""
function reset!(this::AdaptiveDataModel)
  error("not implemented")
  this
end

function set_alphabet!(number_of_symbols::UInt32)
  error("not implemented")
  this
end

function update!(this::AdaptiveDataModel, ::Bool)
  error("not implemented")
  this
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

export ArithmeticCodec, buffer, set_buffer!, start_encoder!, start_decoder!,
  read_from_file!, stop_encoder!, write_to_file!, stop_decoder!, put_bit!,
  get_bit!, put_bits!, get_bits!, encode!, decode!

const AC_MIN_LENGTH::UInt32 = 0x01000000 # threshold for renormalization
const AC_MAX_LENGTH::UInt32 = 0xFFFFFFFF # maximum AC interval length

"""
Encoder and decoder class

Class with both the arithmetic encoder and decoder.  All compressed data is
saved to a memory buffer
"""
mutable struct ArithmeticCodec

  code_buffer::Vector{UInt8}
  ac_index::Int

  # arithmetic coding state
  base::UInt32
  value::UInt32
  length::UInt32

  buffer_size::UInt32
  mode::UInt32 # mode: 0 = undef, 1 = encoder, 2 = decoder

  ArithmeticCodec() = new(
    UInt32[], 0,
    ntuple(_ -> zero(UInt32), 3)...,
    ntuple(_ -> zero(UInt32), 2)...,
  )
  ArithmeticCodec(args...) = set_buffer!(ArithmeticCodec(), args...)

end

function buffer(this::ArithmeticCodec)::Vector{UInt8}
  this.code_buffer
end

function set_buffer!(
    this::ArithmeticCodec,
    max_code_bytes::UInt32,
    user_buffer::Union{Nothing, Vector{UInt8}} = nothing, # `nothing` = assign new
  )

  # test for reasonable sizes
  16 <= max_code_bytes <= 0x1000000 || error("invalid codec buffer size")

  iszero(this.mode) || error("cannot set buffer while encoding or decoding")

  if !isnothing(user_buffer) # user provides memory buffer
    this.buffer_size = max_code_bytes
    this.code_buffer = user_buffer # set buffer for compressed data
    return this
  end

  max_code_bytes <= this.buffer_size && return # enough available

  # assign new memory & set buffer for compressed data
  this.buffer_size = max_code_bytes
  this.code_buffer = zeros(UInt8, max_code_bytes + 16) # 16 extra bytes

  this
end

function start_encoder!(this::ArithmeticCodec)
  iszero(this.mode) || error("cannot start encoder")
  iszero(this.buffer_size) && error("no code buffer set")

  # initialize encoder variables: interval and pointer
  this.mode = 1
  this.base = 0
  this.length = AC_MAX_LENGTH
  this.ac_index = 1 # index of next data byte

  this
end

function start_decoder!(this::ArithmeticCodec)
  iszero(this.mode) || error("cannot start decoder")
  iszero(this.buffer_size) && error("no code buffer set")

  # initialize decoder: interval, index, initial code value
  this.mode = 2;
  this.length = AC_MAX_LENGTH
  this.ac_index = 4
  this.value = (
    UInt32(this.code_buffer[1]) << 24 |
    UInt32(this.code_buffer[2]) << 16 |
    UInt32(this.code_buffer[3]) <<  8 |
    UInt32(this.code_buffer[4])
  )
  this
end

"""
Read code data, start decoder
"""
function read_from_file!(this::ArithmeticCodec, code_file::IOStream)
  error("not implemented")
  this
end

function stop_encoder!(this::ArithmeticCodec)::UInt32 # returns number of bytes used
  this.mode == 1 || error("invalid to stop encoder")
  this.mode = 0

  init_base::UInt32 = this.base # done encoding: set final data bytes

  if this.length > 2 * AC_MIN_LENGTH
    this.base  += AC_MIN_LENGTH # base offset
    this.length = AC_MIN_LENGTH >> 1 # set new length for 1 more byte
  else
    this.base  += AC_MIN_LENGTH >> 1 # base offset
    this.length = AC_MIN_LENGTH >> 9 # set new length for 2 more bytes
  end

  init_base > this.base && propagate_carry!(this) # overflow = carry

  renorm_enc_interval!(this) # renormalization = output last bytes

  code_bytes::UInt32 = UInt32(this.ac_index - 1)
  code_bytes > this.buffer_size && error("code buffer overflow")
  code_bytes # number of bytes used
end

"""
Stop encoder, write code data
"""
function write_to_file!(this::ArithmeticCodec, code_file::IOStream)::UInt32
  error("not implemented")
  zero(UInt32)
end

function stop_decoder!(this::ArithmeticCodec)
  this.mode == 2 || error("invalid to stop decoder")
  this.mode = 0
  this
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

function encode!(this::ArithmeticCodec, bit::UInt32, M::StaticBitModel)
  DEBUG && this.mode != 1 && error("encoder not initialized")

  x::UInt32 = M.bit_0_prob * (this.length >> BM_LENGTH_SHIFT) # product l x p0

  # update interval
  if iszero(bit)
    this.length = x
  else
    init_base::UInt32 = this.base
    this.base   += x
    this.length -= x
    init_base > this.base && propagate_carry!(this) # overflow = carry
  end

  this.length < AC_MIN_LENGTH && renorm_enc_interval!(this) # renormalization

  this
end

function decode!(this::ArithmeticCodec, M::StaticBitModel)::UInt32
  DEBUG && this.mode != 2 && error("decoder not initialized")

  x::UInt32 = M.bit_0_prob * (this.length >> BM_LENGTH_SHIFT) # product l x p0
  bit::UInt32 = (this.value >= x) # decision

  # update & shift interval
  if (bit == 0)
    this.length = x
  else
    this.value  -= x # shifted interval base = 0
    this.length -= x
  end

  this.length < AC_MIN_LENGTH && renorm_dec_interval!(this) # renormalization

  bit # return data bit value
end

function encode!(this::ArithmeticCodec, data::UInt32, ::StaticDataModel)
  error("not implemented")
  this
end

function decode!(this::ArithmeticCodec, ::StaticDataModel)::UInt32
  error("not implemented")
  zero(UInt32)
end

function encode!(this::ArithmeticCodec, bit::UInt32, ::AdaptiveBitModel)
  error("not implemented")
  this
end

function decode!(this::ArithmeticCodec, ::AdaptiveBitModel)::UInt32
  error("not implemented")
  zero(UInt32)
end

function encode!(this::ArithmeticCodec, data::UInt32, ::AdaptiveDataModel)
  error("not implemented")
  this
end

function decode!(this::ArithmeticCodec, ::AdaptiveDataModel)::UInt32
  error("not implemented")
  zero(UInt32)
end

function propagate_carry!(this::ArithmeticCodec)
  # carry propagation on compressed data buffer
  p = this.ac_index - 1
  while this.code_buffer[p] == 0xff
    this.code_buffer[p] = 0
    p -= 1
  end
  this.code_buffer[p] += 1
  this
end

function renorm_enc_interval!(this::ArithmeticCodec)
  # output and discard top byte
  while true
    this.code_buffer[this.ac_index] = UInt8(this.base >> 24)
    this.ac_index += 1
    this.base <<= 8
    (this.length <<= 8) < AC_MIN_LENGTH || break # length multiplied by 256
  end
  this
end

function renorm_dec_interval!(this::ArithmeticCodec)
  # read least-significant byte
  while true
    this.ac_index += 1
    this.value = (this.value << 8) | this.code_buffer[this.ac_index]
    (this.length <<= 8) < AC_MIN_LENGTH || break # length multiplied by 256
  end
  this
end

end # module FastAC
