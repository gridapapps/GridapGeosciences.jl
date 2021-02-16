module TMP

using Gridap

# Geometry
domain = (0,1,0,1); cells = (10,10)
model = CartesianDiscreteModel(domain,cells)

# FE spaces
order = 2
reffe_u = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_u,dirichlet_tags=[7,8])
u1, u2 = 0.0, 1.0
U = TrialFESpace(V,[u1,u2])

# Integration
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

# Parameter-dependent and space-dependent conductivity
# This can eventually replaced by a NN.
α(p,x) = p[1]*(1+p[2]*(x[1]+x[2]))

# Conductivity-dependent weak form
a(α,du,dv) = ∫( α*∇(dv)⋅∇(du) )dΩ
l(dv) = 0

# sensitivity of a wrt α. This can also be computed with AD,
# but it would require to express α in a FESpace
da_dα(dα,du,dv) = ∫( dα⊙∇(dv)⋅∇(du) )dΩ

# primal problem in terms of p
function p_to_u(p)
  αp(x) = α(p,x)
  ap(du,dv) = a(αp,du,dv)
  op = AffineFEOperator(ap,l,U,V)
  solve(op)
end

# Select the "target" parameters
p̂ = VectorValue(1.5,0.25)

# Compute the associated "target" direct solution
û = p_to_u(p̂)

writevtk(Ω,"Ω",cellfields=["û"=>û])

# Objective function.
# I.e. the l2 error wrt the "target" u.
# It returns a cell-wise value.
# The actual objective is computed as sum(f(u))
f(u) = ∫( abs2(u-û) )dΩ

@assert sum(f(û)) + 1 ≈ 1

# Objective function (scalar), plus its gradient (vector) in terms of p
# fp, ∇fp = f_∇f(p)
function f_∇f(p)

  # Direct problem
  αp(x) = α(p,x)
  ap(du,dv) = a(αp,du,dv)
  op = AffineFEOperator(ap,l,U,V)
  u = solve(op)

  # Adjoint solve
  A = get_matrix(op)
  df_du = ∇(f)(u) # Gridap's built-in autodiff
  b = assemble_vector(df_du,V)
  λ = FEFunction(V,A\b) # assuming A self-adjoint

  # Gradient of the α wrt p
  # if α is a NN, one can use the gradient implemented for the NN
  # Here, using Gridap's built-in autodiff
  dα_dp(x) = ∇(p->α(p,x))(p)

  # Objective
  fp = sum(f(u))

  # Its gradient wrt p (assuming a linear wrt α)
  ∇fp = -1*sum(da_dα(dα_dp,u,λ))

  fp, ∇fp
end

fp̂, ∇fp̂ = f_∇f(p̂)

@show fp̂
@show ∇fp̂

p = VectorValue(1.25,0.35)

fp, ∇fp = f_∇f(p)

@show fp
@show ∇fp

end # module
