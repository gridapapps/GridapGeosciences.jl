# test_serial_cubed_sphere.jl
#
# Tests AtlasGrid / AtlasDiscreteModel on the cubed sphere (6 QUAD panels,
# gnomonic projection) using the CubedSphereMesh coarse mesh library.
#
# Coarse mesh (from CoarseMeshes.jl): 8 nodes, 6 QUAD cells.
# See get_coarse_mesh(::CubedSphereMesh) for the full topology diagram.
#
# Checks:
#   1. Cell count = NPANELS × 4^NUM_REF
#   2. Local (α,β) coords are 2D and lie in [−π/4, π/4]²
#   3. Ambient coords lie on the sphere: ‖(X,Y,Z)‖ ≈ RADIUS
#   4. writevtk produces a 3D sphere surface VTK file
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_serial_cubed_sphere.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const n_panels = 6   # cubed sphere has 6 panels

const RADIUS  = 1.0
const NUM_REF = 2    # 6 coarse cells → 6 × 4² = 96 fine cells

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build AtlasDiscreteModel via CubedSphereMesh
# ─────────────────────────────────────────────────────────────────────────────

atlas_model = AtlasDiscreteModel(CubedSphereMesh(RADIUS), NUM_REF)
atlas_grid  = get_atlas_grid(atlas_model)

println("AtlasGrid:  Dc=$(Gridap.Geometry.num_cell_dims(atlas_grid)), Da=$(get_ambient_dim(atlas_grid))")
println("            cells = $(Gridap.Geometry.num_cells(atlas_grid))")
println("AtlasModel: built ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cell count
# ─────────────────────────────────────────────────────────────────────────────

expected_cells = n_panels * 4^NUM_REF
@assert Gridap.Geometry.num_cells(atlas_grid) == expected_cells
println("Cell count check passed ✓  ($expected_cells cells)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Local (α,β) coords in [−π/4, π/4]²
# ─────────────────────────────────────────────────────────────────────────────

cell_chart_coords = Gridap.Geometry.get_cell_coordinates(atlas_grid)
half_edge = π / 4
for i in 1:expected_cells
  for pt in cell_chart_coords[i]
    @assert -half_edge - 1e-12 ≤ pt[1] ≤ half_edge + 1e-12 "cell $i: α=$(pt[1]) out of [-π/4, π/4]"
    @assert -half_edge - 1e-12 ≤ pt[2] ≤ half_edge + 1e-12 "cell $i: β=$(pt[2]) out of [-π/4, π/4]"
  end
end
println("Local coord range check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ambient coords on the sphere: ‖(X,Y,Z)‖ ≈ RADIUS
# ─────────────────────────────────────────────────────────────────────────────

phys_coords = _local_to_ambient(
  atlas_grid.cell_chart_coords,
  atlas_grid.cell_ambient_maps,
)
for i in 1:expected_cells
  for pt in phys_coords[i]
    r = sqrt(pt[1]^2 + pt[2]^2 + pt[3]^2)
    @assert isapprox(r, RADIUS; atol=1e-12) "cell $i: ‖pt‖=$r ≠ $RADIUS"
  end
end
println("Ambient coord sphere check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  VTK output
# ─────────────────────────────────────────────────────────────────────────────

mkpath("output")
writevtk(atlas_model, "output/cubed_sphere_serial")
println("Written output/cubed_sphere_serial_2.vtu — open in Paraview ✓")

println("test_serial_cubed_sphere: ALL CHECKS PASSED")
