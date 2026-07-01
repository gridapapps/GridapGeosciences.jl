using Test

@testset "ForwardInverseMap" begin include("ForwardInverseMapTests.jl") end

@testset "Overloads" begin include("OverloadTests.jl") end
