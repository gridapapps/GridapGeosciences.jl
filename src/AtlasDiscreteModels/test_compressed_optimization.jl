# test_compressed_optimization.jl
#
# Verifies that the lazy_map optimization chain produces CompressedArrays with
# only one entry per chart (not per cell) throughout the metric/ambient-map pipeline.
#
# The key insight: cell_metric and cell_ambient_maps are CompressedArrays built by
#   lazy_map(Reindex(per_chart_vector), cell_to_chart)
# which (after our Reindex specialization) directly stores per_chart_vector as
# .values with cell_to_chart as .ptrs.  Any subsequent lazy_map(f, CompressedArray)
# fires Gridap's _lazy_map_compressed, producing a new CompressedArray whose .values
# is obtained by applying f to only the small per-chart values array.
#
# Checks:
#   1. Basic: lazy_map(Reindex(v), ptrs) isa CompressedArray
#   2. Basic: lazy_map(f, CompressedArray) isa CompressedArray
#   3. For each shape: every step in the chain is a CompressedArray with
#      length(values) == n_charts (not n_cells).
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_compressed_optimization.jl

using Gridap
using Gridap.Arrays: CompressedArray, Reindex

include("AtlasDiscreteModels.jl")

# ── 1. Basic Reindex → CompressedArray ───────────────────────────────────────

let v = [10.0, 20.0, 30.0],
    ptrs = [1, 2, 3, 1, 2, 3, 1]

  ca = lazy_map(Reindex(v), ptrs)
  @assert ca isa CompressedArray "lazy_map(Reindex(Vector), ptrs) returned $(typeof(ca)), expected CompressedArray"
  @assert ca.values === v
  @assert ca.ptrs   === ptrs
  @assert ca[1] == 10.0 && ca[4] == 10.0 && ca[3] == 30.0
  println("Basic Reindex: lazy_map(Reindex(v), ptrs) isa CompressedArray ✓")
end

# ── 2. Basic f(CompressedArray) → CompressedArray ────────────────────────────

let vals = [1.0, 2.0, 3.0],
    ptrs = [1, 1, 2, 2, 3, 3]

  ca   = CompressedArray(vals, ptrs)
  ca2  = lazy_map(x -> x^2, ca)
  @assert ca2 isa CompressedArray "lazy_map(f, CompressedArray) returned $(typeof(ca2)), expected CompressedArray"
  @assert length(ca2.values) == 3
  @assert ca2[1] == 1.0 && ca2[3] == 4.0 && ca2[5] == 9.0
  println("Basic f(CompressedArray): lazy_map(f, ca) isa CompressedArray ✓")
end

# ── 3. Per-shape chain checks ─────────────────────────────────────────────────

function check_compressed_chain(shape_name, mesh, n_charts_expected, nref=1)
  g       = AtlasGrid(mesh, nref)
  n_cells = num_cells(g)

  # ── cell_ambient_maps ─────────────────────────────────────────────────────
  amaps = get_cell_ambient_maps(g)
  @assert amaps isa CompressedArray "[$shape_name] cell_ambient_maps isa $(typeof(amaps)), expected CompressedArray"
  @assert length(amaps.values) == n_charts_expected "[$shape_name] cell_ambient_maps.values has $(length(amaps.values)) entries, expected $n_charts_expected"
  @assert length(amaps) == n_cells

  # ── cell_metric ───────────────────────────────────────────────────────────
  cmet = g.cell_metric
  @assert cmet isa CompressedArray "[$shape_name] cell_metric isa $(typeof(cmet)), expected CompressedArray"
  @assert length(cmet.values) == n_charts_expected "[$shape_name] cell_metric.values has $(length(cmet.values)) entries, expected $n_charts_expected"
  @assert length(cmet) == n_cells

  # ── lazy_map(gradient, amaps) ─────────────────────────────────────────────
  grads = lazy_map(gradient, amaps)
  @assert grads isa CompressedArray "[$shape_name] lazy_map(gradient, amaps) isa $(typeof(grads)), expected CompressedArray"
  @assert length(grads.values) == n_charts_expected "[$shape_name] gradient values has $(length(grads.values)) entries, expected $n_charts_expected"

  # ── lazy_map(Operation(JtJ()), grads) ────────────────────────────────────
  jtj_met = lazy_map(Operation(JtJ()), grads)
  @assert jtj_met isa CompressedArray "[$shape_name] lazy_map(JtJ, grads) isa $(typeof(jtj_met)), expected CompressedArray"
  @assert length(jtj_met.values) == n_charts_expected "[$shape_name] JtJ values has $(length(jtj_met.values)) entries, expected $n_charts_expected"

  # ── get_cell_inv_metric (via inverse_metric_field per chart) ─────────────
  inv_met = get_cell_inv_metric(g)
  @assert inv_met isa CompressedArray "[$shape_name] get_cell_inv_metric isa $(typeof(inv_met)), expected CompressedArray"
  @assert length(inv_met.values) == n_charts_expected "[$shape_name] inv_metric values has $(length(inv_met.values)) entries, expected $n_charts_expected"

  # ── lazy_map(Operation(inv), cell_metric) ────────────────────────────────
  inv_met2 = lazy_map(Operation(inv), cmet)
  @assert inv_met2 isa CompressedArray "[$shape_name] lazy_map(Operation(inv), cmet) isa $(typeof(inv_met2)), expected CompressedArray"
  @assert length(inv_met2.values) == n_charts_expected "[$shape_name] Operation(inv) values has $(length(inv_met2.values)) entries, expected $n_charts_expected"

  println("$shape_name: all compression checks passed ✓  ($n_cells fine cells, $n_charts_expected charts)")
end

# CylinderMesh: 3×3 = 9 coarse charts
check_compressed_chain("Cylinder (r=1.5)", CylinderMesh(1.5), 9)

# MobiusStripMesh: 2 coarse charts (C1, C2)
check_compressed_chain("Möbius strip (R=1.0, W=0.3)", MobiusStripMesh(1.0, 0.3), 2)

# CubedSphereMesh: 6 panels
check_compressed_chain("CubedSphere (r=1.2)", CubedSphereMesh(1.2), n_panels)

println("test_compressed_optimization: ALL CHECKS PASSED")
