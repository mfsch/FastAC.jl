# This file contains recipes for the `just` command runner.
# See https://github.com/casey/just for more information.
set positional-arguments

_default:
  @just --list

test *params:
  #!/usr/bin/env -S julia --project
  import Pkg
  Pkg.precompile()
  try Pkg.test(test_args = ARGS) catch _ exit(1) end
