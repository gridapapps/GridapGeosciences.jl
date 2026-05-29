# test_hdiv_cylinder.jl
#
# Diagnostic H(div) tests for RT elements on the cylinder.
# No pressure, no BoundaryTriangulation.
#
# Two stages:
#
#   1. Interpolation check — verify a function analytically in RT_k
#      reproduces with L2 error ≈ machine epsilon.
#      Candidate: u = (0, 0, z)
#        Reference components: u_ref = (0, dz/2 · z) = (0, linear in t) ∈ RT1 ✓
#        Tangent to cylinder ✓   div_S = 1   u·n|_{z=0} = 0
#
#   2. H(div) convergence — solve
#        (u, v) + (∇·u, ∇·v) = (f, v) + (g, ∇·v)   ∀ v ∈ H(div)
#      with smooth u_exact NOT in RT1. Well-posed without any BC because
#      the bilinear form is coercive on the full H(div) space.
#      Candidate: u = (0, 0, sin(z))
#        Tangent ✓   div_S = cos(z)   u·n = 0 on both caps → no BC term
#        Expected rate: ORDER + 1
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_hdiv_cylinder.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const RADIUS = 1.0
const HEIGHT = 2π
const ORDER  = 1

# ── Manufactured solution (NOT in RT1) ──────────────────────────────────
u_exact(x)     = VectorValue(0.0, 0.0, sin(x[3]))
div_u_exact(x) = cos(x[3])

# ── Function expected to be exactly in RT1 ───────────────────────────────
# u_ref = (0, dz/2 · z) is linear in t → RT1 on each quad chart
u_in_rt1(x) = VectorValue(0.0, 0.0, x[3])

println("H(div) diagnostics on cylinder  (r=$RADIUS, h=$HEIGHT, RT$ORDER)")
println("────────────────────────────────────────────────────────────────────")

eu_errors = Float64[]

for nref in 2:4
  model = AtlasDiscreteModel(CylinderMesh(RADIUS, HEIGHT), nref)
  Ω     = Triangulation(model)
  dΩ    = Measure(Ω, 2*(ORDER + 2))

  reffeU = ReferenceFE(raviart_thomas, Float64, ORDER)
  V = TestFESpace(model, reffeU; conformity=:HDiv)
  U = TrialFESpace(V)

  # ── Stage 1: interpolation ────────────────────────────────────────────
  uh_i = interpolate(u_in_rt1, U)
  ei   = uh_i - u_in_rt1
  err_i = sqrt(sum(∫( ei⋅ei )dΩ))
  println("nref=$nref  interp L2 error (u_in_rt1): $err_i")
  @assert err_i < 1e-10 "RT1 interpolation error $err_i too large"

  # ── Stage 2: H(div) Galerkin projection ──────────────────────────────
  a(u, v) = ∫( u⋅v + (∇⋅u)*(∇⋅v) )dΩ
  l(v)    = ∫( u_exact⋅v + div_u_exact*(∇⋅v) )dΩ

  op = AffineFEOperator(a, l, U, V)
  uh = solve(op)

  eu = uh - u_exact
  push!(eu_errors, sqrt(sum(∫( eu⋅eu )dΩ)))

  ncells = num_cells(Ω)
  ndofs  = num_free_dofs(U)
  println("         H(div) L2 error: $(eu_errors[end])  (ncells=$ncells, ndofs=$ndofs)")
end

println()
println("Convergence rates (expected ≈ $(ORDER+1)):")
for i in 2:length(eu_errors)
  r = log(eu_errors[i-1]/eu_errors[i]) / log(2)
  println("  nref $(i+1)→$(i+2):  rate=$(round(r; digits=3))")
  @assert r > ORDER + 0.5 "Convergence rate $r below expected $(ORDER+1)"
end
println("All H(div) checks passed ✓")
