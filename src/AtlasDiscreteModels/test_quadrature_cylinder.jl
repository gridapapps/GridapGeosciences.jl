# test_quadrature_cylinder.jl
#
# Quadrature convergence test on the cylinder.
#
# Manufactured solution in chart coordinates:
#   u(s,t) = 1 - t²   (P2 polynomial, z-only, no angular dependence)
#
# This is the only P2 polynomial that satisfies both value AND gradient
# continuity across the angular seams of the 2-chart cylinder:
#   At seam θ=π  (s_C1=+1, s_C2=−1): u = 1−t², grad u = (0, −2t) in both charts ✓
#   At seam θ=0≡2π (s_C1=−1, s_C2=+1): same ✓
# Any polynomial with s-dependence (e.g. s², st) has ∂u/∂s|_{s=1}=+2 vs −2 on the other
# chart — a distributional seam flux that invalidates the Galerkin formulation.
#
# Surface metric for cylinder r=1, h=2π (constant in chart coords):
#   g₁₁ = (π/2)²,  g₂₂ = π²,  g₁₂ = 0
#   g⁻¹: g¹¹ = 4/π², g²² = 1/π²
#   Δ_S u = g¹¹·∂²u/∂s² + g²²·∂²u/∂t² = (4/π²)·0 + (1/π²)·(−2) = −2/π²
#   f = −Δ_S u = 2/π²   (constant forcing)
#
# Homogeneous Dirichlet BCs: u(s,±1) = 1−(±1)² = 0  → satisfied at z=0,2π ✓
#
# In physical coordinates: t = z/π − 1, so
#   u_exact(x) = 1 − (x[3]/π − 1)²
#
# Since u ∈ P2 is in the FE space exactly, the Galerkin error with EXACT
# integration is (machine) zero.  Any nonzero L2 error comes purely from
# numerical quadrature of the bilinear form and the RHS.  As quadrature
# order increases with the mesh fixed, the error should converge to machine zero.
#
# Run:
#   julia --project=. AtlasDiscreteModels/test_quadrature_cylinder.jl

using Gridap
using GridapGeosciences

include("AtlasDiscreteModels.jl")

const R = 1.0
const H = 2π

# u(s,t) = 1 - t²,  t = z/π - 1  →  u_exact(x) = 1 - (x[3]/π - 1)²
u_exact(x) = 1.0 - (x[3]/π - 1.0)^2

# f = -Δ_S u = 2/π²  (from surface Laplacian with cylinder metric g²² = π²)
const f_src = 2.0/π^2

println("Quadrature convergence on cylinder (r=$R, h=$H)")
println("Manufactured: u(s,t) = 1−t² ∈ P2  →  u ∈ FE space exactly")
println("Forcing: f = 2/π² ≈ $(round(f_src; digits=8))")
println()

model = AtlasDiscreteModel(CylinderMesh(R, H), 3)   # 128 fine cells, fixed mesh
trian = Triangulation(model)
order = 2
reffe = ReferenceFE(lagrangian, Float64, order)
V = FESpace(model, reffe; conformity=:H1, dirichlet_tags=["bottom","top"])
U = TrialFESpace(V, u_exact)

# High-order reference quadrature for error measurement (not for the solve)
dΩ_ref = Measure(trian, 14)

println("qorder  L2_error")
println("─────────────────────────────────────")

for qorder in 2:10
  dΩ = Measure(trian, qorder)
  a(u,v) = ∫(∇(u) ⋅ ∇(v)) * dΩ
  l(v)   = ∫(f_src * v) * dΩ
  uh     = solve(AffineFEOperator(a, l, U, V))
  e      = uh - u_exact
  err    = sqrt(sum(∫(e * e) * dΩ_ref))
  println("  $qorder       $err")
end
