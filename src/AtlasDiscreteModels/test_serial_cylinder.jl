# test_serial_cylinder.jl
#
# Tests AtlasGrid / AtlasDiscreteModel on the cylinder atlas using the
# CylinderMesh coarse mesh (default: 4 cells around × 1 row).
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_serial_cylinder.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const RADIUS = 1.0
const HEIGHT = 1.0
const NUM_REF = 3   # 4 coarse × 4^3 = 256 fine cells

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build AtlasDiscreteModel via CylinderMesh
# ─────────────────────────────────────────────────────────────────────────────

atlas_model = AtlasDiscreteModel(CylinderMesh(RADIUS, HEIGHT), NUM_REF)
atlas_grid  = get_atlas_grid(atlas_model)

println("AtlasGrid:  Dc=$(Gridap.Geometry.num_cell_dims(atlas_grid)), Da=$(get_ambient_dim(atlas_grid))")
println("            cells = $(Gridap.Geometry.num_cells(atlas_grid))")
println("AtlasModel: built ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Structural checks
# ─────────────────────────────────────────────────────────────────────────────

@assert Gridap.Geometry.num_cells(atlas_grid) == 9 * 4^NUM_REF

# get_cell_coordinates returns LOCAL 2D coords in (θ,z) ∈ [0,2π] × [0,HEIGHT]
cell_chart_coords = Gridap.Geometry.get_cell_coordinates(atlas_grid)
@assert length(cell_chart_coords) == 9 * 4^NUM_REF
for i in 1:Gridap.Geometry.num_cells(atlas_grid)
  for pt in cell_chart_coords[i]
    @assert 0.0 - 1e-12 ≤ pt[1] ≤ 2π + 1e-12 "cell $i: θ=$(pt[1]) out of [0,2π]"
    @assert 0.0 - 1e-12 ≤ pt[2] ≤ HEIGHT + 1e-12 "cell $i: z=$(pt[2]) out of [0,$HEIGHT]"
  end
end
println("Local coord (θ,z) range check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ambient coords: r = RADIUS, z ∈ [0, HEIGHT]
# ─────────────────────────────────────────────────────────────────────────────

phys_coords = _local_to_ambient(
  atlas_grid.cell_chart_coords,
  atlas_grid.cell_ambient_maps,
)
for i in 1:Gridap.Geometry.num_cells(atlas_grid)
  for pt in phys_coords[i]
    r = sqrt(pt[1]^2 + pt[2]^2)
    @assert isapprox(r, RADIUS; atol=1e-12) "cell $i: r=$r ≠ $RADIUS"
    @assert 0.0 - 1e-12 ≤ pt[3] ≤ HEIGHT + 1e-12 "cell $i: z=$(pt[3]) out of [0,$HEIGHT]"
  end
end
println("Ambient coord cylinder check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Seam continuity via the maps stored in CoarseMeshInfo
# ─────────────────────────────────────────────────────────────────────────────

coarse_info = get_coarse_mesh(CylinderMesh(RADIUS, HEIGHT))
maps        = coarse_info.ambient_maps
dθ = 2π/3;  dz = HEIGHT/3
# θ-seam: within each row, check that adjacent charts agree at their shared θ boundary,
# and that the wrap seam (C3/C6/C9 at θ=2π matches C1/C4/C7 at θ=0) is satisfied.
for row in 0:2
  z_vals = range(row*dz, (row+1)*dz; length=9)
  for i in 1:3
    k     = row*3 + i
    knext = row*3 + (i % 3) + 1
    θ_seam_right = i * dθ          # right edge of chart k  (= 3dθ=2π for i=3)
    θ_seam_left  = (i % 3) * dθ    # left  edge of chart k+1 (= 0 for i=3)
    for z in z_vals
      @assert maps[k](Point(θ_seam_right, z)) ≈ maps[knext](Point(θ_seam_left, z)) "Row $row: C$k→C$knext mismatch at z=$z"
    end
  end
end
println("Seam continuity check passed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Face label propagation: "bottom" and "top" survive refinement
# ─────────────────────────────────────────────────────────────────────────────

fine_labels   = Gridap.Geometry.get_face_labeling(atlas_model)
edge_entities = Gridap.Geometry.get_face_entity(fine_labels, 1)
bottom_tag    = Gridap.Geometry.get_tag_from_name(fine_labels, "bottom")
top_tag       = Gridap.Geometry.get_tag_from_name(fine_labels, "top")
bottom_entity = fine_labels.tag_to_entities[bottom_tag][1]
top_entity    = fine_labels.tag_to_entities[top_tag][1]
bottom_edges  = findall(==(bottom_entity), edge_entities)
top_edges     = findall(==(top_entity),    edge_entities)
@assert !isempty(bottom_edges) "\"bottom\" tag not found on fine mesh edges"
@assert !isempty(top_edges)    "\"top\" tag not found on fine mesh edges"
# 3 coarse bottom/top edges, each refines into 2^NUM_REF fine edges.
expected_bdry = 3 * 2^NUM_REF
@assert length(bottom_edges) == expected_bdry "Expected $expected_bdry bottom edges, got $(length(bottom_edges))"
@assert length(top_edges)    == expected_bdry "Expected $expected_bdry top edges, got $(length(top_edges))"
println("Face label propagation check passed ✓  ($expected_bdry bottom, $expected_bdry top edges)")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  VTK output
# ─────────────────────────────────────────────────────────────────────────────

mkpath("output")
writevtk(atlas_model, "output/cylinder")
println("Written output/cylinder_2.vtu — open in Paraview ✓")
