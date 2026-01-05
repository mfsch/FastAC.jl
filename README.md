# FastAC.jl

This project aims to provide a Julia implementation of the “FastAC” algorithm
for arithmetic coding as originally published by Amir Said.
It is meant to match the original C++ implementation exactly such that it can
be used as a drop-in replacement for functionality built on the original code.

The Julia code is based on the `int_32_32` version of the [original C++
code][1]; other versions might be added later.

## License

The [original C++ code][1] as well as the modifications made to port it Julia
are provided under a BSD 2-clause license.

[1]: https://sites.ecse.rpi.edu/~pearlman/SPIHT/EW_Code/FastAC_fix-nh.zip
