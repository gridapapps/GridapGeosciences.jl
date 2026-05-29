# test_poisson_cylinder_intrinsic.jl
#
# Intrinsic Poisson (Laplace-Beltrami) on the cylinder using IntrinsicManifold().
# Chart-local coordinates are the physical cylinder parameters (θ,z) ∈ [0,2π]×[0,h],
# so functions receive Point{2} with x[1]=θ, x[2]=z — no ambient map needed.
#
# Metric of a cylinder of radius r in (θ,z) coordinates:
#   g = diag(r², 1),   g⁻¹ = diag(1/r², 1),   √g = r
# This is constant (no θ or z dependence) and is stored in AtlasGrid.cell_metric.
#
# Area check (first thing to verify):
#   ∫ √g dΩ = r · 2π · h
#
# Bilinear form (weak Laplace-Beltrami, no boundary terms):
#   a(u,v) = ∫ ∇v · (g⁻¹ · ∇u) · √g  dΩ
#   l(v)   = ∫ f · v · √g  dΩ
#
# Manufactured solution (zero mean for any r, h):
#   u(θ,z) = cos θ + cos(2πz/h)
#   −Δ_S u = (1/r²)cosθ + (2π/h)²cos(2πz/h)
#   ∫u dA = 0 for any h  (each term integrates to zero independently)
#   Natural Neumann at top/bottom: ∂u/∂z|_{z=0,h} = 0  → no boundary term
#
# Expected L2 convergence rate: p + 1  (≈ 3 for Q2)
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_poisson_cylinder_intrinsic.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

radius = 1.0
height = 1.0
order  = 2

# Manufactured solution in physical chart coordinates (θ,z) — valid for any radius, height
u_exact(x) = cos(x[1]) + cos(2π * x[2] / height)
f_rhs(x)   = (1/radius^2) * cos(x[1]) + (2π/height)^2 * cos(2π * x[2] / height)

println("Intrinsic Poisson on cylinder  (r=$radius, h=$height, Q$order)")
println("────────────────────────────────────────────────────────────────────")

# ── Area check ────────────────────────────────────────────────────────────────
let
  model_check = AtlasDiscreteModel(CylinderMesh(radius, height), 2;
                                   manifold_style = IntrinsicManifold())
  Ω_check  = Triangulation(model_check)
  dΩ_check = Measure(Ω_check, 4)
  g_cf     = MetricCellField(Ω_check)
  sqrtg_cf = Operation(x -> sqrt(det(x)))(g_cf)
  area     = sum(∫(sqrtg_cf)dΩ_check)
  expected = radius * 2π * height
  println("Area check:  computed=$(round(area; digits=6))  expected=$(round(expected; digits=6))")
  @assert isapprox(area, expected; rtol=1e-10) "Area mismatch: $area ≠ $expected"
  println("Area check passed ✓")
end

errors = Float64[]

for nref in 2:4
  model = AtlasDiscreteModel(CylinderMesh(radius, height), nref;
                             manifold_style = IntrinsicManifold())
  Ω  = Triangulation(model)
  dΩ = Measure(Ω, 2*(order + 2))

  # Metric from the stored AtlasGrid data — no ambient map used
  g_cf          = MetricCellField(Ω)
  inv_metric_cf = Operation(inv)(g_cf)
  sqrtg_cf      = Operation(x -> sqrt(det(x)))(g_cf)

  mean_u = sum(∫(u_exact * sqrtg_cf)dΩ)
  @assert abs(mean_u) < 1e-8 "u_exact not zero mean: $mean_u"

  reffe = ReferenceFE(lagrangian, Float64, order)
  V = TestFESpace(model, reffe; conformity=:H1, constraint=:zeromean)
  U = TrialFESpace(V)

  a(u,v) = ∫( (∇(v) ⋅ (inv_metric_cf ⋅ ∇(u))) * sqrtg_cf )dΩ
  l(v)   = ∫( f_rhs * v * sqrtg_cf )dΩ

  op = AffineFEOperator(a, l, U, V)
  uh = solve(op)

  e = uh - u_exact
  push!(errors, sqrt(sum(∫(e * e * sqrtg_cf)dΩ)))

  ncells = num_cells(Ω)
  ndofs  = num_free_dofs(V)
  println("nref=$nref  ncells=$ncells  ndofs=$ndofs  L2_err=$(errors[end])")
end

println()
println("Convergence rates (expected ≈ $(order+1)):")
for i in 2:length(errors)
  rate = log(errors[i-1]/errors[i]) / log(2)
  println("  nref $(i+1)→$(i+2): $(round(rate; digits=3))")
end
for i in 2:length(errors)
  rate = log(errors[i-1]/errors[i]) / log(2)
  @assert rate > order + 0.5 "Convergence rate $rate below expected $(order+1)"
end
println("All convergence checks passed ✓")
