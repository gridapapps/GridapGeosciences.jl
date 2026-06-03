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
@testset "SerialCubedSphereTests" begin include("SerialCubedSphereTests.jl") end
@testset "SerialCylinderTests" begin include("SerialCylinderTests.jl") end
@testset "SerialMobiusTests" begin include("SerialMobiusTests.jl") end