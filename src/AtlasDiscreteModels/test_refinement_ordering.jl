# test_refinement_ordering.jl
#
# Verifies that Gridap.Adaptivity.refine(model, n) produces parent-major cell
# ordering when applied to an UnstructuredDiscreteModel of QUADs.
#
# "Parent-major" means: all n^Dc children of coarse cell 1 appear first (in child
# order 1..n^Dc), then all children of coarse cell 2, etc.  This is the ordering
# guaranteed by blocked_refinement_glue (used by uniformly_refine, which is
# what refine(model, n) calls for UnstructuredDiscreteModel).
#
# _build_atlas_grid uses refine(coarse_unstr, 2^num_refinements) — a single call
# with subdivision factor n=2^K — and relies on parent-major ordering when it sets
#   child_ids = repeat(1:n_per_chart, ncharts)
# coarse_unstr is obtained by converting the input coarse_model to
# UnstructuredDiscreteModel unconditionally (see _build_atlas_grid step 1).
#
# Key facts verified here:
#   1. refine(UnstructuredDiscreteModel, 2): 4 children/cell, parent-major
#   2. refine(UnstructuredDiscreteModel, 4): one-shot 16 children/cell, parent-major
#   3. Two sequential refine(.,2) give identical coarse-parent mapping as one-shot refine(.,4)
#   4. refine(CartesianDiscreteModel, 4): lexicographic ordering, NOT parent-major
#      (demonstrates why the Unstructured conversion in _build_atlas_grid is required)
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_refinement_ordering.jl

using Gridap

# -- Build a 3x2 Cartesian coarse QUAD mesh -----------------------------------
cart_model   = CartesianDiscreteModel((0,1,0,1), (3,2))   # 3 cols x 2 rows = 6 cells
coarse_model = Gridap.Geometry.UnstructuredDiscreteModel(cart_model)

n_coarse = Gridap.Geometry.num_cells(coarse_model)
@assert n_coarse == 6

# -- Single refinement --------------------------------------------------------
adapted1 = Gridap.Adaptivity.refine(coarse_model, 2)
glue1    = adapted1.glue

n_c = Gridap.Geometry.num_cells(adapted1.model) / n_coarse   # children per coarse cell
n_c = Int(n_c)
@assert n_c == 4   # each QUAD -> 4 children

# parent-major: n2o_faces_map[end][i] = ceil(i / n_c)
expected_parents1 = repeat(1:n_coarse, inner=n_c)
@assert Vector{Int}(glue1.n2o_faces_map[end]) == expected_parents1 "n2o_faces_map not parent-major"

# child ids cycle 1..n_c within each parent block
expected_child_ids1 = repeat(1:n_c, n_coarse)
@assert Vector{Int}(glue1.n2o_cell_to_child_id) == expected_child_ids1 "n2o_cell_to_child_id not 1..n_c"

println("Single refinement: parent-major ordering confirmed  (n_coarse=$n_coarse, n_c=$n_c)")

# -- Double refinement (sequential) ------------------------------------------
adapted2  = Gridap.Adaptivity.refine(adapted1.model, 2)
glue2     = adapted2.glue
n_fine2   = Gridap.Geometry.num_cells(adapted2.model)
n_c2      = Int(n_fine2 / Gridap.Geometry.num_cells(adapted1.model))   # 4 again

# child ids at level 2 also cycle 1..n_c within each level-1 parent block
expected_child_ids2 = repeat(1:n_c2, Gridap.Geometry.num_cells(adapted1.model))
@assert Vector{Int}(glue2.n2o_cell_to_child_id) == expected_child_ids2 "level-2 child ids not 1..n_c"

# After 2 refinements, fine cells should be in parent-major order relative to
# original coarse cells: all n_c^2=16 descendants of coarse cell 1 first, etc.
# Verify by composing the two n2o_faces_maps.
level1_parent = Vector{Int}(glue2.n2o_faces_map[end])      # level-2 -> level-1
coarse_parent = Vector{Int}(glue1.n2o_faces_map[end])[level1_parent]  # level-1 -> coarse

n_per_chart2    = n_c * n_c2   # 16 descendants per coarse cell
expected_coarse = repeat(1:n_coarse, inner=n_per_chart2)
@assert coarse_parent == expected_coarse "two-level composition not parent-major"

# Verify repeat(1:n_per_chart, ncharts) gives the correct ref_coords index.
child_ids_2level = repeat(1:n_per_chart2, n_coarse)
for i in 1:n_fine2
  c2    = glue2.n2o_cell_to_child_id[i]
  p1    = glue2.n2o_faces_map[end][i]
  c1    = glue1.n2o_cell_to_child_id[p1]
  expected_flat = (c1 - 1) * n_c2 + c2
  @assert child_ids_2level[i] == expected_flat "child_ids mismatch at i=$i"
end

println("Double refinement (sequential): parent-major composition confirmed  (n_per_chart=$(n_per_chart2))")

# -- One-shot refinement: refine(model, 4) ------------------------------------
#    refine(model, n) uses subdivision factor n (not n recursive levels).
#    n=4 -> 4^2=16 children per QUAD in a single AdaptivityGlue.
#    _build_atlas_grid calls refine(coarse_unstr, 2^num_refinements) for this.
adapted_shot  = Gridap.Adaptivity.refine(coarse_model, 4)
glue_shot     = adapted_shot.glue
n_per_shot    = Int(Gridap.Geometry.num_cells(adapted_shot.model) / n_coarse)
@assert n_per_shot == n_c * n_c2  "one-shot should give $(n_c*n_c2) children, got $n_per_shot"

expected_parents_shot = repeat(1:n_coarse, inner=n_per_shot)
@assert Vector{Int}(glue_shot.n2o_faces_map[end]) == expected_parents_shot "one-shot n2o_faces_map not parent-major"

expected_child_ids_shot = repeat(1:n_per_shot, n_coarse)
@assert Vector{Int}(glue_shot.n2o_cell_to_child_id) == expected_child_ids_shot "one-shot child ids not 1..n_per_chart"

# One-shot and sequential give the same coarse-parent mapping
@assert Vector{Int}(glue_shot.n2o_faces_map[end]) == coarse_parent "one-shot and two-level parent maps differ"

println("One-shot refine(model,4): parent-major ordering confirmed  (n_per_chart=$n_per_shot)")

# -- CartesianDiscreteModel gives lexicographic ordering (wrong path) ----------
#    refine(::CartesianDiscreteModel, n) dispatches to _create_cartesian_f2c_maps,
#    which gives lexicographic (column-major) ordering, NOT parent-major.
#    _build_atlas_grid converts to UnstructuredDiscreteModel unconditionally to
#    prevent this regardless of what type the caller provides.
cart_adapted = Gridap.Adaptivity.refine(cart_model, 4)
cart_parents = Vector{Int}(cart_adapted.glue.n2o_faces_map[end])
@assert cart_parents != expected_parents_shot "CartesianDiscreteModel unexpectedly gave parent-major ordering"
println("CartesianDiscreteModel gives lexicographic (non-parent-major) ordering (as expected)")
println("  -> the UnstructuredDiscreteModel conversion in _build_atlas_grid is required")

println("test_refinement_ordering: ALL CHECKS PASSED")
