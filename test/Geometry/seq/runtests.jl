using Test

@testset "AmbientNormalVector" begin include("AmbientNormalVectorTests.jl") end

@testset "CellMap" begin include("CellMapTests.jl") end

@testset "NormalVector" begin include("NormalVectorTests.jl") end

@testset "Refinement" begin include("RefinementTests.jl") end

@testset "SkeletonTriangulation" begin include("SkeletonTriangulationTests.jl") end

@testset "SurfaceArea" begin include("SurfaceAreaTests.jl") end
