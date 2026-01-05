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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

export StaticBitModel, set_probability_0!

"""
Static model for binary data
"""
mutable struct StaticBitModel

  bit_0_prob::UInt32

  StaticBitModel() = new(zero(UInt32))

end

"""
Set probability of symbol '0'
"""
function set_probability_0!(::StaticBitModel, ::Float64) end

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
  this
end

function update!(this::AdaptiveBitModel)
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
  this
end

function set_alphabet!(number_of_symbols::UInt32)
  this
end

function update!(this::AdaptiveDataModel, ::Bool)
  this
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

export ArithmeticCodec, buffer, set_buffer!, start_encoder!, start_decoder!,
  read_from_file!, stop_encoder!, write_to_file!, stop_decoder!, put_bit!,
  get_bit!, put_bits!, get_bits!, encode!, decode!

"""
Encoder and decoder class

Class with both the arithmetic encoder and decoder.  All compressed data is
saved to a memory buffer
"""
mutable struct ArithmeticCodec

  code_buffer::Vector{UInt8}
  new_buffer::Vector{UInt8}
  ac_pointer::Vector{UInt8}

  # arithmetic coding state
  base::UInt32
  value::UInt32
  length::UInt32

  buffer_size::UInt32
  mode::UInt32 # mode: 0 = undef, 1 = encoder, 2 = decoder

  # TODO: finish constructors
  Arithmetic_Codec() = new()
  Arithmetic_Codec(
    max_code_bytes::UInt32,
    user_buffer::Union{Nothing, Vector{UInt8}} = nothing, # `nothing` = assign new
  ) = new()
end

function buffer(this::ArithmeticCodec)::Vector{UInt8}
  this.code_buffer
end

function set_buffer!(max_code_bytes::UInt32, user_buffer::Union{Nothing, Vector{UInt8}} = nothing) # `nothing` = assign new
  this
end

function start_encoder!(this::ArithmeticCodec)
  this
end

function start_decoder!(this::ArithmeticCodec)
  this
end

"""
Read code data, start decoder
"""
function read_from_file!(this::ArithmeticCodec, code_file::IOStream)
  this
end

function stop_encoder!(this::ArithmeticCodec)::UInt32 # returns number of bytes used
  zero(UInt32)
end

"""
Stop encoder, write code data
"""
function write_to_file!(this::ArithmeticCodec, code_file::IOStream)::UInt32
  zero(UInt32)
end

function stop_decoder!(this::ArithmeticCodec)
  this
end

function put_bit!(this::ArithmeticCodec, bit::UInt32)
  this
end

function get_bit!(this::ArithmeticCodec)::UInt32
  zero(UInt32)
end

function put_bits!(this::ArithmeticCodec, data::UInt32, number_of_bits::UInt32)
  this
end

function get_bits!(this::ArithmeticCodec, number_of_bits::UInt32)::UInt32
  zero(UInt32)
end

function encode!(this::ArithmeticCodec, bit::UInt32, ::StaticBitModel)
  this
end

function decode!(this::ArithmeticCodec, ::StaticBitModel)::UInt32
  zero(UInt32)
end

function encode!(this::ArithmeticCodec, data::UInt32, ::StaticDataModel)
  this
end

function decode!(this::ArithmeticCodec, ::StaticDataModel)::UInt32
  zero(UInt32)
end

function encode!(this::ArithmeticCodec, bit::UInt32, ::AdaptiveBitModel)
  this
end

function decode!(this::ArithmeticCodec, ::AdaptiveBitModel)::UInt32
  zero(UInt32)
end

function encode!(this::ArithmeticCodec, data::UInt32, ::AdaptiveDataModel)
  this
end

function decode!(this::ArithmeticCodec, ::AdaptiveDataModel)::UInt32
  zero(UInt32)
end

function propagate_carry!(this::ArithmeticCodec)
  this
end

function renorm_enc_interval!(this::ArithmeticCodec)
  this
end

function renorm_dec_interval!(this::ArithmeticCodec)
  this
end

end # module FastAC
