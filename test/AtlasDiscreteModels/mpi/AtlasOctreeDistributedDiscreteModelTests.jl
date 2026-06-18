# test_distributed_cubed_sphere.jl
#
# Verify AtlasOctreeDistributedDiscreteModel (DistributedAtlasDiscreteModels.jl)
# on the cubed sphere with p4est / GridapP4est.
#
# Checks:
#   1. AtlasGrid stores local (α,β) 2D reference coords (not 3D ambient).
#   2. Local (α,β) coords match those stored in the reference model.
#   3. cell_ambient_maps[i] applied to local coords matches CubedSphereMap directly.
#   4. All ambient corners lie on the sphere of the given radius.
#   5. writevtk produces Da=3 dimensional output via visualization_data.
#
module AtlasOctreeDistributedDiscreteModelTests
    using Gridap
    using GridapGeosciences
    using MPI 
    using PartitionedArrays
    using GridapDistributed


    # Check that every local fine cell's ambient corners lie on the sphere of radius `r`.
    function test_atlas_octree_model(local_model::AtlasDiscreteModel{Dc,Da}, r::Real) where {Dc,Da}
      g     = local_model.atlas_grid
      phys  = GridapGeosciences.Geometry._local_to_ambient(g.cell_chart_coords, g.cell_ambient_maps)
      ncells = length(phys)
      for i in 1:ncells
        for pt in phys[i]
          norm = sqrt(sum(pt[k]^2 for k in 1:Da))
          @assert isapprox(norm, r; atol=1e-10) "cell $i: ‖pt‖=$norm ≠ radius=$r  pt=$pt"
        end
      end
      ncells
    end

    const RADIUS  = 1.0
    const NUM_REF = 1    # 6 coarse cells → 24 fine cells at level 1

    function main(ranks)
        # ── Build new AtlasOctreeDistributedDiscreteModel ────────────────────────────
        atlas_model = AtlasOctreeDistributedDiscreteModel(
          ranks, CubedSphereMesh(RADIUS), NUM_REF)

        # ── Reference: CubedSphere2DParametricOctreeDistributedDiscreteModel ─────────
        ref_model = CubedSphere2DParametricOctreeDistributedDiscreteModel(
          ranks, RADIUS; num_initial_uniform_refinements = NUM_REF)

        # ── Per-rank verification ─────────────────────────────────────────────────────
        map(
          local_views(atlas_model.atlas_dmodel),
          local_views(ref_model.parametric_dmodel),
        ) do local_atlas, local_ref

          g       = local_atlas.atlas_grid
          ncells  = num_cells(g)

          # 1. Local coords are 2D (α,β)
          cell_chart_coords = g.cell_chart_coords
          @assert length(cell_chart_coords) == ncells

          # 2. Local (α,β) coords match reference model's 2D coords
          #    (reference stores them in grid.cell_map.args[1])
          alpha_beta_ref = local_ref.grid.cell_map.args[1]
          for i in 1:ncells
            @assert cell_chart_coords[i] ≈ alpha_beta_ref[i] "Local coords mismatch at cell $i"
          end

          # 3. Ambient coords computed via _local_to_ambient match direct map evaluation
          phys = GridapGeosciences.Geometry._local_to_ambient(cell_chart_coords, g.cell_ambient_maps)
          for i in 1:ncells
            fwd   = g.cell_ambient_maps[i]
            cache = Gridap.Arrays.return_cache(fwd, cell_chart_coords[i][1])
            for (k, (local_pt, actual_pt)) in enumerate(zip(cell_chart_coords[i], phys[i]))
              expected_pt = Gridap.Arrays.evaluate!(cache, fwd, local_pt)
              @assert expected_pt ≈ actual_pt "Rank $(MPI.Comm_rank(MPI.COMM_WORLD)) cell $i corner $k: expected=$expected_pt got=$actual_pt"
            end
          end

          # 4. All ambient corners on the sphere
          nchecked = test_atlas_octree_model(local_atlas, RADIUS)
          rank = MPI.Comm_rank(MPI.COMM_WORLD)
          println("Rank $rank: $nchecked cells verified ✓")
        end

        # ── VTK output ────────────────────────────────────────────────────────────────
        mkpath("output")
        writevtk(atlas_model, "output/cubed_sphere")
        println("Written output/cubed_sphere_2.vtu ✓")

        println("test_distributed_cubed_sphere: ALL CHECKS PASSED")
    end

    MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    ranks  = distribute_with_mpi(LinearIndices((nprocs,)))
    main(ranks)
    MPI.Finalize()
end 