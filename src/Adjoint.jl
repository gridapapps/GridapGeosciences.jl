using Gridap

# Geometry
domain = (0,1,0,1); cells = (10,10)
model = CartesianDiscreteModel(domain,cells)

# Integration
degree = 2
Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
dΩ = LebesgueMeasure(Ω,degree)
dΓ = LebesgueMeasure(Γ,degree)

# Manufactured solution
u(x) = x[1]+x[2]

# Parameter-dependent weak form
a(p,u,v) = ∫( (p*p)*∇(v)⋅∇(u) )*dΩ
l(v) = ∫( -v*Δ(u) )*dΩ

# Objective function in terms of primal solution
# (squared distance wrt exact solution u on Ω and Γ
# as an example involving different domains)
# Note: the following function returns cell contributions.
# The final objective is computed as sum(j(v))
j(v) = ∫( abs2(u-v) )*dΩ + ∫( abs2(u-v) )*dΓ

# FE spaces: P for params, V, U for primal solution
P = FESpace(model,ReferenceFE(:Lagrangian,Float64,0),conformity=:L2)
V = TestFESpace(model,ReferenceFE(:Lagrangian,Float64,1),dirichlet_tags="boundary")
U = TrialFESpace(V,u)

using Gridap.FESpaces

# Objective function (scalar), plus its gradient (vector) in terms of p
# fp, ∇fp = f_∇f(p)
function f_∇f(p)
  # Direct solve
  a_at_p(u,v) = a(p,u,v)
  op = AffineFEOperator(a_at_p,l,U,V)
  u = solve(op)
  # Adjoint solve
  A = get_matrix(op)
  dj_du = ∇(j)(u) # autodiff
  b = assemble_vector(dj_du,V)
  λ = FEFunction(V,A\b) # assuming A self-adjoint
  # we want to have something like this
  # op_t = AdjointFEOperator(op, ∇(j), selfadjoint=true )
  # u, lambda = solve(op,op_t)
  # for the moment, we can do more low level things at the driver level, e.g., extracting matrices, etc for the adjoint
  # in any case, it is not needed for self-adjoint problems
  # Objective
  fp = sum(j(u))
  # Its gradient
  ∂a_∂p = ∇(x -> a(x,u,λ))(p) # Autodiff
  ∇fp = -1*assemble_vector(∂a_∂p,P)
  fp, ∇fp
end

# run for a given realization of p
p = FEFunction(P,rand(num_free_dofs(P)).+1)
fp, ∇fp = f_∇f(p)
# Print results
@show fp
writevtk(Ω,"trian",cellfields=["∇fp"=>FEFunction(P,∇fp)])

function df_dp(uvec,pvec,λvec)
  da(p,u,v,dp) = (2*p*dp)*∇(v)⋅∇(u)
  uh = FEFunction(U,uvec)
  ph = FEFunction(P,pvec)
  λh = FEFunction(V,λvec)
  t = FESource((dp)->da(ph,uh,λh,dp),trian,quad)
  op = AffineFEOperator(P,Q,t)
  get_vector(op)
end
