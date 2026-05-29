# test_darcy_cylinder.jl
#
# Augmented mixed Darcy on the cylinder.
#
# PDE:  u + ∇_S p = 0,   ∇_S · u = f   on the cylinder (r=RADIUS, h=2π)
#
# Augmented bilinear form (adds H(div) diagonal block):
#   a((u,p),(v,q)) = ∫(u·v + (∇·u)(∇·v) - p(∇·v) + q(∇·u))dΩ
#
# The (∇·u)(∇·v) term makes the u-block coercive on H(div).
# Pressure is still unique only up to a constant → zero-mean constraint used.
#
# Strong Dirichlet BC: u·n = 0 on top/bottom caps (imposed in H(div) space).
# This makes v·n = 0 on ∂Ω, so the natural boundary term ∫(p_bc v·n)dΓ
# vanishes identically — no BoundaryTriangulation needed.
#
# Manufactured solution (unit cylinder, x=cosθ, y=sinθ):
#   p_exact = cos z + cos θ = cos(x[3]) + x[1]
#   u_exact = −∇_S p = (y², −xy, sin z)  [tangent, u·n=0 at z=0,2π ✓]
#   f       = −Δ_S p = p_exact            [eigenfunction of Laplace-Beltrami]
#
# Augmented RHS derivation:
#   Augmented strong form: u − ∇(∇·u) + ∇p = F_u,  ∇·u = f
#   Since ∇·u = f = p_exact and ∇p = −u (Darcy), and ∇f = ∇p_exact:
#     F_u = u − ∇f + ∇p = u − ∇p_exact + (−∇p_exact)... wait:
#     u + ∇p = 0 (standard Darcy) → ∇p = −u_exact
#     ∇(∇·u) = ∇f = ∇p_exact = −u_exact
#     F_u = u_exact − (−u_exact) + (−u_exact) = u_exact
#   So: l((v,q)) = ∫(u_exact·v + f·q)dΩ  (no boundary term!)
#
# Expected L2 convergence rate: k+1 for both u and p.
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_darcy_cylinder.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const RADIUS = 1.0
const HEIGHT = 2π
const ORDER  = 1   # RT1/DG1 → expected rate 2

p_exact(x)   = cos(x[3]) + x[1]
u_exact(x)   = VectorValue(-x[2]^2, x[1]*x[2], sin(x[3]))
f_forcing(x) = cos(x[3]) + x[1]

println("Darcy (augmented mixed) on cylinder  (r=$RADIUS, h=$HEIGHT, RT$ORDER/P$ORDER)")
println("──────────────────────────────────────────────────────────────────────────────")

eu_errors = Float64[]
ep_errors = Float64[]

for nref in 2:4
  model = AtlasDiscreteModel(CylinderMesh(RADIUS, HEIGHT), nref)
  Ω     = Triangulation(model)

  degree = 2*(ORDER + 2)
  dΩ = Measure(Ω, degree)

  reffeU = ReferenceFE(raviart_thomas, Float64, ORDER)
  reffeP = ReferenceFE(lagrangian,     Float64, ORDER)

  # Strong Dirichlet u·n = 0 on top/bottom (v·n = 0 → boundary term vanishes)
  V = TestFESpace(model,  reffeU; conformity=:HDiv,
                  dirichlet_tags=["bottom","top"])
  U = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))

  Q = TestFESpace(model,  reffeP; conformity=:L2, constraint=:zeromean)
  P = TrialFESpace(Q)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # Augmented Darcy: (u,v) + (∇·u,∇·v) saddle point
  a((u,p),(v,q)) = ∫( u⋅v + (∇⋅u)*(∇⋅v) - p*(∇⋅v) + q*(∇⋅u) )dΩ
  l((v,q))       = ∫( u_exact⋅v + f_forcing*q )dΩ

  op  = AffineFEOperator(a, l, X, Y)
  xh  = solve(op)
  uh, ph = xh

  eu = uh - u_exact
  ep = ph - p_exact
  push!(eu_errors, sqrt(sum(∫( eu⋅eu )dΩ)))
  push!(ep_errors, sqrt(sum(∫( ep*ep )dΩ)))

  ncells = num_cells(Ω)
  ndofs  = num_free_dofs(X)
  println("nref=$nref  ncells=$ncells  ndofs=$ndofs  eu=$(eu_errors[end])  ep=$(ep_errors[end])")
end

println()
println("Convergence rates (expected ≈ $(ORDER+1)):")
for i in 2:length(ep_errors)
  ru = log(eu_errors[i-1]/eu_errors[i]) / log(2)
  rp = log(ep_errors[i-1]/ep_errors[i]) / log(2)
  println("  nref $(i+1)→$(i+2):  u-rate=$(round(ru;digits=3))  p-rate=$(round(rp;digits=3))")
  @assert ru > ORDER + 0.5 "u convergence rate $ru below expected $(ORDER+1)"
  @assert rp > ORDER + 0.5 "p convergence rate $rp below expected $(ORDER+1)"
end
println("All convergence checks passed ✓")
