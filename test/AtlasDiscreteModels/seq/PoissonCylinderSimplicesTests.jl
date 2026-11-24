module PoissonCylinderSimplicesTests

#
# Solves the Poisson equation −Δ_S u = f on a cylinder surface split into triangles,
# using AtlasDiscreteModel and P2 Lagrangian elements. The problem is that of
# PoissonCylinderTests. The quadrilaterals of the coarsest level are split with
# `simplexify`, and the next levels are refinements of those triangles.
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

using Gridap
using Gridap.Adaptivity: refine
using Gridap.Geometry: get_cell_polytopes
using Gridap.ReferenceFEs: is_simplex, simplexify
using GridapGeosciences
using Test

const RADIUS = 1.0
const HEIGHT = 2π
const ORDER  = 2

u_exact(x) = cos(x[3]) + x[1]
f_rhs(x)   = cos(x[3]) + x[1]

function solve_poisson(model)
  trian = Triangulation(model)
  dΩ    = Measure(trian, 2*ORDER + 2)
  reffe = ReferenceFE(lagrangian, Float64, ORDER)
  V = FESpace(model, reffe; conformity=:H1, dirichlet_tags=["bottom","top"])
  U = TrialFESpace(V, u_exact)
  a(u,v) = ∫(∇(u) ⋅ ∇(v)) * dΩ
  l(v)   = ∫(f_rhs * v) * dΩ
  uh = solve(AffineFEOperator(a, l, U, V))
  e  = uh - u_exact
  error = sqrt(sum(∫(e * e) * dΩ))
  println("ncells=$(num_cells(trian))  ndofs=$(num_free_dofs(V))  L2_err=$error")
  error
end

println("Poisson on cylinder split into triangles  (r=$RADIUS, h=$HEIGHT, P$ORDER)")
println("─────────────────────────────────────────────────")

let model = simplexify(AtlasDiscreteModel(CylinderMesh(RADIUS, HEIGHT), 2))
  errors = [solve_poisson(model)]
  for _ in 1:2
    model = refine(model)
    @test all(is_simplex, get_cell_polytopes(model))
    push!(errors, solve_poisson(model))
  end
  rates = log2.(errors[1:end-1] ./ errors[2:end])
  println("Convergence rates: $(round.(rates; digits=3))  (expected ≈ $(ORDER+1))")
  @test all(>(2.8), rates)
end

end # module PoissonCylinderSimplicesTests
