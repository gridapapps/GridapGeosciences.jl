# test_get_cell_map.jl
#
# Smoke test: AtlasGrid.get_cell_map returns the composed
# (ref element → chart → ambient) map, not a bilinear interpolation.
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_get_cell_map.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const RADIUS  = 1.0
const HEIGHT  = 1.0
const NUM_REF = 1   # 2 coarse × 4 = 8 fine cells

atlas_model = AtlasDiscreteModel(CylinderMesh(RADIUS, HEIGHT), NUM_REF)
atlas_grid  = get_atlas_grid(atlas_model)

cell_maps = Gridap.Geometry.get_cell_map(atlas_grid)
println("get_cell_map type: $(typeof(cell_maps))")
println("element type:      $(eltype(cell_maps))")

# Gridap's QUAD reference element is [0,1]² (not [-1,1]²).
# Node corners: (0,0) BL, (1,0) BR, (0,1) TL, (1,1) TR.  Centre: (0.5,0.5).
ref_corners = [Point(0.0,0.0), Point(1.0,0.0), Point(0.0,1.0), Point(1.0,1.0)]
ref_centre  = Point(0.5, 0.5)

# Evaluate first cell's map at the reference-element centre
m1    = cell_maps[1]
cache = Gridap.Arrays.return_cache(m1, ref_centre)
pt    = Gridap.Arrays.evaluate!(cache, m1, ref_centre)
println("Cell 1 at ξ=$(ref_centre) → $pt  ($(typeof(pt)))")

@assert pt isa Point{3,Float64} "Expected Point{3,Float64}, got $(typeof(pt))"

# All cells: centre and all four corners must land on the cylinder surface
ncells = Gridap.Geometry.num_cells(atlas_grid)
for i in 1:ncells
  mi = cell_maps[i]
  for ξ in [ref_centre; ref_corners]
    c   = Gridap.Arrays.return_cache(mi, ξ)
    p   = Gridap.Arrays.evaluate!(c, mi, ξ)
    r_xy = sqrt(p[1]^2 + p[2]^2)
    @assert isapprox(r_xy, RADIUS; atol=1e-12) "cell $i at $ξ: r_xy=$r_xy ≠ $RADIUS"
    @assert -1e-12 ≤ p[3] ≤ HEIGHT + 1e-12    "cell $i at $ξ: z=$(p[3]) out of [0,$HEIGHT]"
  end
end
println("All $ncells cell maps: centre + corner on-surface check passed ✓")

println("test_get_cell_map: ALL CHECKS PASSED")
