# test_darcy_cubed_sphere.jl
#
# Scalar Hodge Laplacian (Darcy) in mixed form on the unit sphere.
#
# PDE:  u + ∇_S p = 0,   ∇_S · u = f   on S² (closed manifold, no boundary)
#
# Manufactured solution  (l=1 spherical harmonic, Laplace–Beltrami eigenvalue 2):
#   p_exact = z = x[3]
#   u_exact = (xz, yz, z²−1)   [= −∇_S p, tangent to sphere]
#   f       = 2z                [= −Δ_S p = l(l+1) p with l=1]
#
# Verification:
#   ∇_S z = ∇z − (∇z·n̂) n̂ = (0,0,1) − z(x,y,z) = (−xz,−yz,1−z²)
#   u = −∇_S p = (xz,yz,z²−1) ✓
#   u·n̂ = x²z+y²z+z³−z = z(x²+y²+z²)−z = z−z = 0 (tangent) ✓
#   −Δ_S p = 2z = f ✓
#
# Mixed FE: RT_k (H(div)) for velocity, DG_k (L²) with zero-mean for pressure.
# No boundary term (closed manifold).
#
# Expected L2 convergence rate: k+1 for both u and p.
#
# NOTE: The cylinder coarse mesh has a PERIODIC topology that forces pindex=2
# on its shared bottom/top boundary circle edges: both cells share those edges
# as topologically-interior, and one cell must traverse each in decreasing
# global-node-ID order (pindex=2).  This is not a general quad phenomenon — for
# non-periodic oriented quad meshes all pindices are 1.  Gridap's RT elements
# only register one permutation per edge, so pindex=2 crashes H(div) assembly.
# The cubed sphere is used here because its non-periodic topology ensures all
# edge permutation indices are 1.
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_darcy_cubed_sphere.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const RADIUS = 1.0
const ORDER  = 1   # RT1/DG1 → expected rate 2

p_exact(x) = x[3]
u_exact(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)
f_forcing(x) = 2*x[3]

println("Darcy (mixed) on unit sphere  (RT$ORDER/P$ORDER, zero-mean pressure)")
println("────────────────────────────────────────────────────────────────────────")

eu_errors = Float64[]
ep_errors = Float64[]

for nref in 1:3
  model = AtlasDiscreteModel(CubedSphereMesh(RADIUS), nref)
  Ω     = Triangulation(model)

  degree = 2*(ORDER + 2)
  dΩ    = Measure(Ω, degree)

  reffeU = ReferenceFE(raviart_thomas, Float64, ORDER)
  reffeP = ReferenceFE(lagrangian,     Float64, ORDER)
  V  = TestFESpace(model,  reffeU; conformity=:HDiv)
  Q  = TestFESpace(model,  reffeP; conformity=:L2, constraint=:zeromean)
  U  = TrialFESpace(V)
  P  = TrialFESpace(Q)
  Y  = MultiFieldFESpace([V, Q])
  X  = MultiFieldFESpace([U, P])

  a((u,p),(v,q)) = ∫( u⋅v - p*(∇⋅v) + q*(∇⋅u) )dΩ
  l((v,q))       = ∫( f_forcing * q )dΩ

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
for i in 2:length(eu_errors)
  ru = log(eu_errors[i-1]/eu_errors[i]) / log(2)
  rp = log(ep_errors[i-1]/ep_errors[i]) / log(2)
  println("  nref $(i-1)→$i:  u-rate=$(round(ru;digits=3))  p-rate=$(round(rp;digits=3))")
  @assert ru > ORDER + 0.5 "u convergence rate $ru below expected $(ORDER+1)"
  @assert rp > ORDER + 0.5 "p convergence rate $rp below expected $(ORDER+1)"
end
println("All convergence checks passed ✓")
