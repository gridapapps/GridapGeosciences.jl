# test_metric_fields.jl
#
# Validates the concrete metric and inverse-metric Fields against the generic
# JtJ formula for all three built-in shapes.
#
# Three checks per shape × test point:
#   1. ConcreteMetricField(x)    ≈ JtJ(∇chart_map)(x)
#   2. ConcreteInvMetricField(x) ≈ inv(JtJ(∇chart_map)(x))
#   3. g(x) ⋅ g⁻¹(x)            ≈ I₂   (self-consistency)
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_metric_fields.jl

using Gridap
import Gridap.TensorValues: symmetric_part, SymTensorValue

include("AtlasDiscreteModels.jl")

const ATOL = 1e-12

# ── Helpers ───────────────────────────────────────────────────────────────────

function g_jtj(chart_map::Field, pt::Point)
  J = evaluate(gradient(chart_map), pt)
  symmetric_part(J ⋅ transpose(J))
end

function inv2x2(g::SymTensorValue{2,Float64,3})
  g11, g12, g22 = g[1,1], g[1,2], g[2,2]
  d = g11*g22 - g12^2
  SymTensorValue{2,Float64,3}(g22/d, -g12/d, g11/d)
end

function check_metric_at(shape, chart_map, metric, inv_metric, pt)
  g_ref  = g_jtj(chart_map, pt)
  g_conc = evaluate(metric, pt)
  ginv_conc = evaluate(inv_metric, pt)

  err_g = norm(g_conc - g_ref)
  @assert err_g < ATOL "[$shape] g mismatch at $pt: err=$err_g"

  ginv_ref = inv2x2(g_ref)
  err_ginv = norm(ginv_conc - ginv_ref)
  @assert err_ginv < ATOL "[$shape] g⁻¹ mismatch at $pt: err=$err_ginv"

  prod = g_conc ⋅ ginv_conc
  @assert abs(prod[1,1] - 1) < ATOL &&
          abs(prod[1,2])     < ATOL &&
          abs(prod[2,1])     < ATOL &&
          abs(prod[2,2] - 1) < ATOL "[$shape] g⋅g⁻¹ ≠ I at $pt: prod=$prod"
end

function check_shape(shape, chart_map, metric, inv_metric, pts)
  for pt in pts
    check_metric_at(shape, chart_map, metric, inv_metric, pt)
  end
  println("$shape: all metric checks passed ✓")
end

# ── Cylinder ──────────────────────────────────────────────────────────────────

let r = 1.5
  pts = [Point(0.0, 0.0), Point(π/3, 0.5), Point(π, 0.3), Point(5π/4, 0.9)]
  check_shape(
    "Cylinder (r=$r)",
    CylinderChartMap(r),
    CylinderMetricField(r),
    CylinderInvMetricField(r),
    pts,
  )
end

# ── Möbius strip ──────────────────────────────────────────────────────────────

let R = 1.0, W = 0.3
  pts = [Point(-0.8, 0.4), Point(0.0, 0.0), Point(0.6, -0.5), Point(1.0, 0.9)]
  for (offset, chart) in ((1.0, "C1"), (3.0, "C2"))
    check_shape(
      "Möbius $chart (R=$R, W=$W, offset=$offset)",
      MobiusChartMap(R, W, offset),
      MobiusMetricField(R, W, offset),
      MobiusInvMetricField(R, W, offset),
      pts,
    )
  end
end

# ── Cubed sphere ──────────────────────────────────────────────────────────────
#
# All 6 panels share the same metric formula — verify this too by checking
# that the metric is panel-independent (all panels give the same value).

let r = 1.2, half = cube_half_edge
  pts = [
    Point(0.0, 0.0),
    Point(half/2, half/3),
    Point(-half/3, half/2),
    Point(-0.6*half, -0.4*half),
  ]

  for p in 1:n_panels
    check_shape(
      "CubedSphere panel $p (r=$r)",
      CubedSphereMap(p, r),
      CubedSphereMetricField(r),
      CubedSphereInvMetricField(r),
      pts,
    )
  end

  # Verify panel-independence: metrics at the same (α,β) point must agree
  # across all panels (gnomonic projection is isometric per panel).
  for pt in pts
    g1 = g_jtj(CubedSphereMap(1, r), pt)
    for p in 2:n_panels
      gp = g_jtj(CubedSphereMap(p, r), pt)
      err = norm(g1 - gp)
      @assert err < ATOL "CubedSphere metric differs between panels 1 and $p at $pt: err=$err"
    end
  end
  println("CubedSphere: metric is panel-independent ✓")
end

println("test_metric_fields: ALL CHECKS PASSED")
