using FastAC, Test

@testset "Tests from C++ reference code" begin
  include("ReferenceTests.jl")
  using .ReferenceTests

  # test binary compression (two symbols)
  # TODO: increase number of cycles
  reference_tests(2, 1)

  # test data compression with <=16 symbols (table is not used)
  reference_tests(10, 1)

  # test data compression with >16 symbols (table is used)
  reference_tests(20, 1)
end
