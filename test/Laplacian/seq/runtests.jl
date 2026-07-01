using Test

@testset "HelmholtzTests" begin include("HelmholtzTests.jl") end
@testset "HodgeLaplacianScalarTests" begin include("HodgeLaplacianScalarTests.jl") end
@testset "HodgeLaplacianVectorTests" begin include("HodgeLaplacianVectorTests.jl") end
@testset "LaplaceBeltramiTests" begin include("LaplaceBeltramiTests.jl") end
