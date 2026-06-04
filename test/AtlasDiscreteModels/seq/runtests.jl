using Test

@testset "CompressedOptimizationTests" begin include("CompressedOptimizationTests.jl") end
@testset "DarcyCubedSphereTests" begin include("DarcyCubedSphereTests.jl") end
@testset "DarcyCylinderTests" begin include("DarcyCylinderTests.jl") end
@testset "GetCellMapTests" begin include("GetCellMapTests.jl") end
@testset "HdivCylinderTests" begin include("HdivCylinderTests.jl") end
@testset "MetricFieldsTests" begin include("MetricFieldsTests.jl") end
@testset "PoissonCylinderIntrinsicTests" begin include("PoissonCylinderIntrinsicTests.jl") end
@testset "PoissonCylinderTests" begin include("PoissonCylinderTests.jl") end
@testset "QuadratureCylinderTests" begin include("QuadratureCylinderTests.jl") end
@testset "RefinementOrderingTests" begin include("RefinementOrderingTests.jl") end
@testset "CubedSphereTests" begin include("CubedSphereTests.jl") end
@testset "CylinderTests" begin include("CylinderTests.jl") end
@testset "MobiusTests" begin include("MobiusTests.jl") end
