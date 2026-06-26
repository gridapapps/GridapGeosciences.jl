using Test

@testset "AdvectionTutorial" begin include("AdvectionTest.jl") end
@testset "AmbientModelHodgeLaplacianScalar" begin include("AmbientModelHodgeLaplacianScalarTest.jl") end
@testset "LaplaceBeltramiTutorial" begin include("LaplaceBeltramiTest.jl") end
