# test_serial_mobius.jl
#
# Tests AtlasGrid / AtlasDiscreteModel on a 2-chart Möbius strip atlas using
# the MobiusStripMesh coarse mesh library.
#
# Coarse mesh: 4 nodes, 2 QUAD cells with a half-twist at the seam.
# The twist is encoded purely in the node connectivity:
#
#   3 - 4 - 1
#   |   |   |
#   1 - 2 - 3
#   [C1] [C2]
#
# C1 = [1,2,3,4], C2 = [2,3,4,1]:
# the left edge of C1 {1,3} is identified with the right edge of C2 {3,1}
# (same nodes, reversed orientation) — this is the half-twist.
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_serial_mobius.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const RADIUS    = 1.0
const HALFWIDTH = 0.3
const NUM_REF   = 2   # 2 coarse × 4^2 = 32 fine cells

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build AtlasDiscreteModel via MobiusStripMesh
# ─────────────────────────────────────────────────────────────────────────────

atlas_model = AtlasDiscreteModel(MobiusStripMesh(RADIUS, HALFWIDTH), NUM_REF)
atlas_grid  = get_atlas_grid(atlas_model)

println("AtlasGrid:  Dc=$(Gridap.Geometry.num_cell_dims(atlas_grid)), Da=$(get_ambient_dim(atlas_grid))")
println("            cells = $(Gridap.Geometry.num_cells(atlas_grid))")
println("AtlasModel: built ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Structural checks
# ─────────────────────────────────────────────────────────────────────────────

@assert get_ambient_dim(atlas_grid) == 3
@assert Gridap.Geometry.num_cells(atlas_grid) == 2 * 4^NUM_REF

cell_chart_coords = Gridap.Geometry.get_cell_coordinates(atlas_grid)
@assert length(cell_chart_coords) == 2 * 4^NUM_REF
for i in 1:Gridap.Geometry.num_cells(atlas_grid)
  for pt in cell_chart_coords[i]
    @assert -1.0 - 1e-12 ≤ pt[1] ≤ 1.0 + 1e-12 "cell $i: local s=$(pt[1]) out of [-1,1]"
    @assert -1.0 - 1e-12 ≤ pt[2] ≤ 1.0 + 1e-12 "cell $i: local t=$(pt[2]) out of [-1,1]"
  end
end
println("Local coord range check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ambient coords within expected bounding box
# ─────────────────────────────────────────────────────────────────────────────

phys_coords = _local_to_ambient(atlas_grid.cell_chart_coords, atlas_grid.cell_ambient_maps)
for i in 1:Gridap.Geometry.num_cells(atlas_grid)
  for pt in phys_coords[i]
    r_xy = sqrt(pt[1]^2 + pt[2]^2)
    @assert RADIUS - HALFWIDTH - 1e-12 ≤ r_xy ≤ RADIUS + HALFWIDTH + 1e-12 "cell $i: r_xy=$r_xy out of [$(RADIUS-HALFWIDTH), $(RADIUS+HALFWIDTH)]"
    @assert -HALFWIDTH - 1e-12 ≤ pt[3] ≤ HALFWIDTH + 1e-12 "cell $i: z=$(pt[3]) out of [-$HALFWIDTH, $HALFWIDTH]"
  end
end
println("Ambient coord bounding-box check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Seam continuity via the maps stored in CoarseMeshInfo
# ─────────────────────────────────────────────────────────────────────────────

map_C1, map_C2 = get_coarse_mesh(MobiusStripMesh(RADIUS, HALFWIDTH)).ambient_maps

# Interior seam (θ = π): right edge of C1 = left edge of C2, same orientation.
for t in range(-1.0, 1.0; length=9)
  @assert map_C1(Point(1.0, t)) ≈ map_C2(Point(-1.0, t)) "Interior seam mismatch at t=$t"
end
println("Interior seam check passed ✓")

# Twist seam (θ = 0 ≡ 2π): left edge of C1 = right edge of C2 with t ↦ −t.
for t in range(-1.0, 1.0; length=9)
  @assert map_C1(Point(-1.0, t)) ≈ map_C2(Point(1.0, -t)) "Twist seam mismatch at t=$t"
end
println("Twist seam check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  VTK output
# ─────────────────────────────────────────────────────────────────────────────

mkpath("output")
writevtk(atlas_model, "output/mobius")
println("Written output/mobius_2.vtu — open in Paraview ✓")
