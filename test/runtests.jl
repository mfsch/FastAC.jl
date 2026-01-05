using FastAC, Test

@testset "Tests from C++ reference code" begin
  include("ReferenceTests.jl")
  using .ReferenceTests

  # test binary compression (two symbols)
  # TODO: increase number of cycles
  reference_tests(2, 1)
end
