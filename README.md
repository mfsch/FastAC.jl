# FastAC.jl

This project aims to provide a Julia implementation of the “FastAC” algorithm
for arithmetic coding as originally published by Amir Said.
It is meant to match the original C++ implementation exactly such that it can
be used as a drop-in replacement for functionality built on the original code.

The Julia code is based on the `int_32_32` version of the [original C++
code][1]; other versions might be added later.

## Overview

The FastAC package relies on arithmetic coding to encode a stream of “symbols”
represented by integer values into a byte stream and to decode the byte stream
later to recover the original sequence of symbols.
In the encoded data, each symbol is represented with the optimal number of bits
corresponding to the probability distribution of the symbols.
The probability distribution is either provided *a priori* or it is derived
from the statistics of the symbol sequence while encoding and decoding.
The probability distribution can be specified for each symbol in the stream,
allowing to freely intersperse values with separate distributions, as long as
the same sequence of distributions can be reproduced when decoding the data.

To start encoding, first create an `Encoder` that writes the encoded data to an
internal buffer or an user-provided I/O stream.
Then create one or multiple “models” representing the probability distribution
of the unencoded symbols and encode them one by one with the `encode!`
function.
Finally, call `close` on the encoder to write the final bytes to the output
stream and obtain either.

To start decoding, create a `Decoder` with the encoded data or a n I/O stream
from which to read it.
Then create the same “models” that were used while encoding and use the
`decode!` function to recover the original data.
Note that the decoder cannot determine by itself when the decoding has
finished, so the number of valid unencoded symbols has to be known when
decoding.

FastAC includes four models, which can also be used as building blocks for more
complex models.
The `StaticDataModel{N}` encodes numbers from the range `0:N-1` according to a
user-provided probability distribution.
The `AdaptiveDataModel{N}` encodes the same range, but estimates the
probability distribution from the stream of symbols.
It starts out with an equiprobable distribution and then periodically updates
the probabilities based on the number of times each value has been
encoded/decoded.
The `StaticBitModel` and `AdaptiveBitModel` are specialized implementations for
the case of `N = 2`, i.e. only predicting `0` and `1`.

## Examples

First a simple example just to illustrate the syntax, encoding random values
from the range `0:9`. Since each digit has the same probability, all the
encoding does is to compactly store the data with an average of `log2(10)` bits
but it cannot further reduce the size.

```julia
using FastAC
original_data = rand(0:9, 10_000)

encoder = Encoder()
encode!(encoder, original_data, StaticDataModel(10))
encoded_data = finalize!(encoder)

decoder = Decoder(encoded_data)
decoded_data = decode!(decoder, length(original_data), StaticDataModel(10))
@assert original_data == decoded_data
@info "bit rate" (length(encoded_data) * 8 / length(original_data)) log2(10)
```

Now a more complete example with data that has a non-uniform probability
distribution. In this case, the size of the encoded data should match the
entropy of the probability distribution. The adaptive model “learns” the
probabilities from the data stream both during the encoding and decoding
process, so they do not have to be known a priori.

```julia
using FastAC

nsym = 10 # encoding values from the range 0:9
nval = 10_000

# create random non-uniform probability distribution
pdf = rand(nsym).^5 # higher power gives less uniform pdf
pdf /= sum(pdf)

# create random values with the above PDF
cdf = cumsum(pdf)
vals = map(_ -> (x = rand(); findfirst(>=(x), cdf) - 1), 1:nval)
pdf_vals = [sum(==(x), vals) / nval for x in 0:nsym-1]

# encode these values
encoder = Encoder()
model = AdaptiveDataModel(nsym)
encode!(encoder, vals, model)
encoded_vals = finalize!(encoder)

# decode same data
decoder = Decoder(encoded_vals)
reset!(model) # or create a new one
decoded_vals = decode!(decoder, nval, model)
@assert vals == decoded_vals

# evaluate size of encoded data
entropy_uniform = log2(nsym)
entropy_pdf = sum(-p * log2(p) for p in pdf)
entropy_vals = sum(-p * (iszero(p) ? 0 : log2(p)) for p in pdf_vals)
bit_rate = length(encoded_vals) * 8 / nval
redundancy = (bit_rate - entropy_vals) / entropy_vals * 100
@info "" entropy_uniform entropy_pdf entropy_vals bit_rate redundancy
```

You can also interleave values from different probability distributions by
defining multiple models and calling `encode!`/`decode!` for each value with
the corresponding model.
You can even decide which model to use based on previously encoded/decoded
data, as long as the same sequence of models can be determined both when
encoding and decoding the data.

## License

The [original C++ code][1] as well as the modifications made to port it Julia
are provided under a BSD 2-clause license.

[1]: https://sites.ecse.rpi.edu/~pearlman/SPIHT/EW_Code/FastAC_fix-nh.zip
