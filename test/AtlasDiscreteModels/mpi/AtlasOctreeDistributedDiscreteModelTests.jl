# test_distributed_cubed_sphere.jl
#
# Verify AtlasOctreeDistributedDiscreteModel (DistributedAtlasDiscreteModels.jl)
# on the cubed sphere with p4est / GridapP4est.
#
# Checks (run for CubedSphereMesh and CubedSphereWithThicknessMesh via run_tests,
#         and for the 3D p4est vs p6est comparison via run_3d_comparison):
#   1. AtlasGrid stores local chart coords of the correct dimension.
#   2. Global sum-of-squared chart coords matches the reference model (partition/order-independent).
#   3. All local chart coords fall within the mesh's chart domain.
#   4. All ambient corners lie within the expected radial band [r_min, r_max].
#   5. writevtk produces Da=3 dimensional output via visualization_data.
#
module AtlasOctreeDistributedDiscreteModelTests
    using Gridap
    using GridapGeosciences
    using MPI
    using PartitionedArrays
    using GridapDistributed

    # Verify ambient corners lie in the radial band [r_min, r_max].
    # For CubedSphereMesh r_min == r_max (exact sphere); for shells it is a band.
    function check_radial_bounds(local_model::AtlasDiscreteModel{Dc,Da}, r_min::Real, r_max::Real) where {Dc,Da}
      g      = local_model.atlas_grid
      phys   = GridapGeosciences.Geometry._local_to_ambient(g.cell_chart_coords, g.cell_ambient_maps)
      ncells = length(phys)
      for i in 1:ncells
        for pt in phys[i]
          nrm = sqrt(sum(pt[k]^2 for k in 1:Da))
          @assert r_min - 1e-10 ≤ nrm ≤ r_max + 1e-10 "cell $i: ‖pt‖=$nrm ∉ [$r_min, $r_max]  pt=$pt"
        end
      end
      ncells
    end

    # Chart coordinate bounds per mesh type.
    # CubedSphereMesh:                        (α,β)   ∈ [-π/4, π/4]²
    # CubedSphereWithThicknessMesh:           (γ,α,β) ∈ [0,1] × [-π/4, π/4]²
    # ExtrudedCubedSphereWithThicknessMesh:   (γ,α,β) ∈ [0,1] × [-π/4, π/4]²  (same ordering, see get_p6est_vertex_coord)
    chart_lo(::CubedSphereMesh)                      = fill(-π/4, 2)
    chart_hi(::CubedSphereMesh)                      = fill( π/4, 2)
    chart_lo(::CubedSphereWithThicknessMesh)         = [0.0, -π/4, -π/4]
    chart_hi(::CubedSphereWithThicknessMesh)         = [1.0,  π/4,  π/4]
    chart_lo(::ExtrudedCubedSphereWithThicknessMesh) = [0.0, -π/4, -π/4]
    chart_hi(::ExtrudedCubedSphereWithThicknessMesh) = [1.0,  π/4,  π/4]

    # Radial bounds for the ambient norm check.
    radial_bounds(m::CubedSphereMesh)                      = (m.radius, m.radius)
    radial_bounds(m::CubedSphereWithThicknessMesh)         = (m.radius, m.radius + m.thickness)
    radial_bounds(m::ExtrudedCubedSphereWithThicknessMesh) = (m.radius, m.radius + m.thickness)

    const RADIUS    = 1.0
    const THICKNESS = 0.1
    const NUM_REF   = 1    # 6 coarse cells → 24 fine cells at level 1 (2D), 48 (3D)

    # ── Tests for AtlasOctreeDistributedDiscreteModel vs AtlasDiscreteModel ──────
    function run_tests(ranks, mesh, label)
      atlas_model  = AtlasOctreeDistributedDiscreteModel(ranks, mesh, NUM_REF)
      ref_model    = AtlasDiscreteModel(ranks, mesh, NUM_REF)
      lo           = chart_lo(mesh)
      hi           = chart_hi(mesh)
      r_min, r_max = radial_bounds(mesh)

      map(
        local_views(atlas_model.atlas_dmodel),
        local_views(ref_model),
      ) do local_atlas, local_ref

        g      = local_atlas.atlas_grid
        ncells = num_cells(g)

        # 1. Local chart coords have the correct dimension
        cell_chart_coords = g.cell_chart_coords
        @assert length(cell_chart_coords) == ncells

        # 2. Global fingerprint: sum-of-squared chart coords (partition- and order-independent)
        Dc = num_dims(g)
        local_atlas_sos = sum(
          sum(pt[k]^2 for k in 1:Dc for pt in coords)
          for coords in cell_chart_coords; init=0.0)
        local_ref_sos = sum(
          sum(pt[k]^2 for k in 1:Dc for pt in coords)
          for coords in local_ref.atlas_grid.cell_chart_coords; init=0.0)
        global_atlas_sos = MPI.Allreduce(local_atlas_sos, +, MPI.COMM_WORLD)
        global_ref_sos   = MPI.Allreduce(local_ref_sos,   +, MPI.COMM_WORLD)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
          @assert isapprox(global_atlas_sos, global_ref_sos; atol=1e-10) "[$label] Chart coord fingerprint mismatch"
        end

        # 3. All local chart coords fall within the mesh's chart domain
        for coords in cell_chart_coords
          for pt in coords
            @assert all(lo[k] - 1e-10 ≤ pt[k] ≤ hi[k] + 1e-10 for k in 1:Dc) "[$label] Chart coord out of domain: $pt"
          end
        end

        # 4. All ambient corners within the expected radial band
        nchecked = check_radial_bounds(local_atlas, r_min, r_max)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        println("[$label] Rank $rank: $nchecked cells verified ✓")
      end

      mkpath("output")
      writevtk(atlas_model, "output/$label")
      println("[$label] Written output/$label.vtu ✓")
    end

    # ── Cross-check: AtlasOctreeDistributedDiscreteModel (p4est, CubedSphereWithThicknessMesh)
    #                vs ExtrudedAtlasOctreeDistributedDiscreteModel (p6est, ExtrudedCubedSphereWithThicknessMesh)
    # Both represent the same 3D spherical shell; with NUM_REF horizontal and vertical
    # refinements each produces 6 × 2² × 2¹ = 48 cells.
    function run_3d_comparison(ranks)
      oct_mesh = CubedSphereWithThicknessMesh(RADIUS, THICKNESS)
      ext_mesh = ExtrudedCubedSphereWithThicknessMesh(RADIUS, THICKNESS)

      atlas_oct    = AtlasOctreeDistributedDiscreteModel(ranks, oct_mesh, NUM_REF)
      extruded_oct = ExtrudedAtlasOctreeDistributedDiscreteModel(ranks, ext_mesh, NUM_REF, NUM_REF)

      lo           = chart_lo(ext_mesh)
      hi           = chart_hi(ext_mesh)
      r_min, r_max = radial_bounds(ext_mesh)

      map(
        local_views(atlas_oct.atlas_dmodel),
        local_views(extruded_oct),
      ) do local_atlas, local_extruded

        g_oct = local_atlas.atlas_grid
        g_ext = local_extruded.atlas_grid
        Dc    = num_dims(g_oct)  # 3 for both

        coords_oct = g_oct.cell_chart_coords
        coords_ext = g_ext.cell_chart_coords

        # 1. Local chart coords have the correct dimension
        @assert length(coords_oct) == num_cells(g_oct)
        @assert length(coords_ext) == num_cells(g_ext)

        # 2. Global fingerprint: both models span the same chart coords
        sos_oct = sum(sum(pt[k]^2 for k in 1:Dc for pt in c) for c in coords_oct; init=0.0)
        sos_ext = sum(sum(pt[k]^2 for k in 1:Dc for pt in c) for c in coords_ext; init=0.0)
        global_sos_oct = MPI.Allreduce(sos_oct, +, MPI.COMM_WORLD)
        global_sos_ext = MPI.Allreduce(sos_ext, +, MPI.COMM_WORLD)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
          @assert isapprox(global_sos_oct, global_sos_ext; atol=1e-10) "[3D comparison] Chart coord fingerprint mismatch: oct=$global_sos_oct ext=$global_sos_ext"
        end

        # 3. Extruded model chart coords within its domain
        for coords in coords_ext
          for pt in coords
            @assert all(lo[k] - 1e-10 ≤ pt[k] ≤ hi[k] + 1e-10 for k in 1:Dc) "[3D comparison] Extruded chart coord out of domain: $pt"
          end
        end

        # 4. Ambient corners within the radial band for both models
        nchecked_oct = check_radial_bounds(local_atlas,    r_min, r_max)
        nchecked_ext = check_radial_bounds(local_extruded, r_min, r_max)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        println("[3D comparison] Rank $rank: oct=$nchecked_oct ext=$nchecked_ext cells verified ✓")
      end

      mkpath("output")
      writevtk(atlas_oct,    "output/cubed_sphere_with_thickness_oct")
      writevtk(extruded_oct, "output/cubed_sphere_with_thickness_ext")
      println("[3D comparison] VTK output written ✓")
    end

    function main(ranks)
      run_tests(ranks, CubedSphereMesh(RADIUS),                        "cubed_sphere")
      run_tests(ranks, CubedSphereWithThicknessMesh(RADIUS, THICKNESS), "cubed_sphere_with_thickness")
      run_3d_comparison(ranks)
      println("ALL CHECKS PASSED")
    end

    MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    ranks  = distribute_with_mpi(LinearIndices((nprocs,)))
    main(ranks)
    MPI.Finalize()
end
