# ── MobiusMetric ──────────────────────────────────────────────────────────────
#
# Chart (s,t) ∈ [−1,1]², θ = π(s+offset)/2, ρ = R + W·t·cos(θ/2).
# Metric is diagonal:
#   g₁₁ = (π/4)²·W²·t² + (π/2)²·ρ²
#   g₁₂ = 0
#   g₂₂ = W²
# Derivation: |∂φ/∂s|² expands to W²t²(π/4)² + ρ²(π/2)² (cross terms cancel by
# cos² + sin² = 1 for the angular part and dρ/dt terms); ∂φ/∂s · ∂φ/∂t = 0
# (verified by direct computation); |∂φ/∂t|² = W².

@inline function _mobius_metric(R, W, offset, x)
  θ   = π*(x[1] + offset)/2
  ρ   = R + W*x[2]*cos(θ/2)
  g11 = (π/4)^2 * W^2 * x[2]^2 + (π/2)^2 * ρ^2
  SymTensorValue{2,Float64,3}(g11, 0.0, W^2)
end

struct MobiusMetric <: Field
  radius :: Float64; half_width :: Float64; theta_offset :: Float64
end
Gridap.Arrays.evaluate!(_, m::MobiusMetric, x::Point) =
  _mobius_metric(m.radius, m.half_width, m.theta_offset, x)
function Gridap.Arrays.return_cache(_::MobiusMetric, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::MobiusMetric, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  R, W, offset = m.radius, m.half_width, m.theta_offset
  @inbounds for i in eachindex(xs)
    cache.array[i] = _mobius_metric(R, W, offset, xs[i])
  end
  cache.array
end
