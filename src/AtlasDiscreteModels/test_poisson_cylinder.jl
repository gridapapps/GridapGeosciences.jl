# test_poisson_cylinder.jl
#
# Solves the Poisson equation −Δ_S u = f on a cylinder surface using
# AtlasDiscreteModel and P2 Lagrangian elements.
#
# Manufactured solution:
#   u(θ, z) = cos(z) + cos(θ)  [x = cos θ, so u(x,y,z) = cos(z) + x]
#   −Δ_S u  = cos(θ) + cos(z) = u  →  f = u
#
# Dirichlet BCs on the top (z=2π) and bottom (z=0) circles (face labels from
# CylinderMesh).
#
# Expected L2 convergence rate: 3 for P2 elements.
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_poisson_cylinder.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const RADIUS = 1.0
const HEIGHT = 2π
const ORDER  = 2

u_exact(x) = cos(x[3]) + x[1]
f_rhs(x)   = cos(x[3]) + x[1]

println("Poisson on cylinder  (r=$RADIUS, h=$HEIGHT, P$ORDER)")
println("─────────────────────────────────────────────────")

errors = Float64[]
for nref in 2:4
  model = AtlasDiscreteModel(CylinderMesh(RADIUS, HEIGHT), nref)
  trian = Triangulation(model)
  dΩ    = Measure(trian, 2*ORDER + 2)
  reffe = ReferenceFE(lagrangian, Float64, ORDER)
  V = FESpace(model, reffe; conformity=:H1, dirichlet_tags=["bottom","top"])
  U = TrialFESpace(V, u_exact)
  a(u,v) = ∫(∇(u) ⋅ ∇(v)) * dΩ
  l(v)   = ∫(f_rhs * v) * dΩ
  uh = solve(AffineFEOperator(a, l, U, V))
  e  = uh - u_exact
  push!(errors, sqrt(sum(∫(e * e) * dΩ)))
  println("nref=$nref  ncells=$(num_cells(trian))  ndofs=$(num_free_dofs(V))  L2_err=$(errors[end])")
end

println()
println("Convergence rates:")
for i in 2:length(errors)
  rate = log(errors[i-1]/errors[i]) / log(2)
  println("  level $(i+1)→$(i+2): $(round(rate; digits=3))  (expected ≈ $(ORDER+1))")
end

# Regression: rate should be above 2.8 at each level
for i in 2:length(errors)
  rate = log(errors[i-1]/errors[i]) / log(2)
  @assert rate > 2.8 "Convergence rate $rate < 2.8 at level $i"
end
println("All convergence checks passed ✓")
