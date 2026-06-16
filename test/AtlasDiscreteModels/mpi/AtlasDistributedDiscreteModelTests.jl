module AtlasDistributedDiscreteModelTests
#
# Smoke tests for the non-octree AtlasDiscreteModel distributed constructor.
# Uses PartitionedArrays.DebugArray (2 parts, no MPI required).
#
# Checks:
#   1. Constructor returns a DistributedDiscreteModel.
#   2. Global cell count matches expected value.
#   3. Each local model is an AtlasDiscreteModel.
#   4. Each local model has owned + ghost cells (ghost ≥ 0).
#   5. Local atlas data arrays are consistent with local cell count.
#

    using Gridap
    using GridapGeosciences
    using MPI
    using PartitionedArrays
    using GridapDistributed

    function test_atlas_distributed(ranks, mesh, num_ref, expected_total_cells)
        dmodel = AtlasDiscreteModel(ranks, mesh, num_ref)

        @assert dmodel isa GridapDistributed.DistributedDiscreteModel

        gids = get_cell_gids(dmodel)
        @assert last(gids) == expected_total_cells "Expected $expected_total_cells global cells, got $(last(gids))"

        map(local_views(dmodel), partition(gids)) do lm, lindices
            @assert lm isa AtlasDiscreteModel "Local model should be AtlasDiscreteModel, got $(typeof(lm))"
            n_local = Gridap.Geometry.num_cells(lm)
            n_own   = length(own_to_local(lindices))
            @assert n_local >= n_own "Local cells ($n_local) must be >= owned ($n_own)"
            g = get_atlas_grid(lm)
            @assert length(g.cell_chart_coords) == n_local
            @assert length(g.cell_ambient_maps) == n_local
            @assert length(g.cell_metric)       == n_local
            println("  part $(part_id(lindices)): n_local=$n_local owned=$n_own ghost=$(n_local - n_own)")
        end
        println("PASSED: $(typeof(mesh)), num_ref=$num_ref, total_cells=$expected_total_cells")
    end

    MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    ranks  = distribute_with_mpi(LinearIndices((nprocs,)))

    #println("=== CylinderMesh, num_ref=2 (9×16=144 cells) ===")
    #test_atlas_distributed(ranks, CylinderMesh(1.0, 1.0), 2, 9 * 4^2)

    println("=== CubedSphereMesh, num_ref=1 (6×4=24 cells) ===")
    test_atlas_distributed(ranks, CubedSphereMesh(1.0), 1, 6 * 4^1)

    # println("=== CylinderMesh, num_ref=0 (9 coarse cells) ===")
    # test_atlas_distributed(ranks, CylinderMesh(1.0, 1.0), 0, 9)

    println("ALL TESTS PASSED")

    MPI.Finalize()
end # module
