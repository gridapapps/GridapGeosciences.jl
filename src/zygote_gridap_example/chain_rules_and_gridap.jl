module TMP

using Gridap
using ChainRulesCore
import ChainRulesCore: rrule

domain = (0,1,0,1)
cells = (10,10)
model = CartesianDiscreteModel(domain,cells)

order = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

# Manufactured solution
u(x) = x[1]+x[2]

# Weak form functions
a(p,u,v) = ∫( (p*p)*∇(v)⋅∇(u) )dΩ
l(v) = ∫( -v*Δ(u) )dΩ

# Space for the design dofs
reffe_S = ReferenceFE(lagrangian,Float64,0)
S = TestFESpace(model,reffe_S)
R = TrialFESpace(S)

# Space for the material params
reffe_Q = ReferenceFE(lagrangian,Float64,1)
Q = TestFESpace(model,reffe_Q)
P = TrialFESpace(Q)

# Space for the primal solution
reffe_V = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_V,dirichlet_tags="boundary")
U = TrialFESpace(V,u)

"""
    f(uvec) -> Number

Objective function
"""
function f(uvec)
  function α(uh)
    eh = u-uh
    eh*eh
  end
  uh = FEFunction(U,uvec)
  sum(∫(α(uh))dΩ)
end

"""
    uvec = g(pvec)

Direct problem from material parameters
"""
function g(pvec)
  ph = FEFunction(P,pvec)
  op = AffineFEOperator((u,v)->a(ph,u,v),l,U,V)
  uh = solve(op)
  uvec = get_free_values(uh)
  uvec
end

"""
    pvec = h(rvec)

Material parameters from design dofs `rvec`.
"""
function h(rvec)
  rh = FEFunction(R,rvec)
  op = AffineFEOperator((p,q)->h1(p,q),q->h1(rh,q),P,Q)
  ph = solve(op)
  pvec = get_free_values(ph)
  pvec
end
h1(p,q) = ∫( ∇(p)⋅∇(q) + p*q )dΩ

"""
   F(rvec) -> Number

Objective function from design dofs
"""
function F(rvec)
  pvec = h(rvec) # Smoothing
  uvec = g(pvec) # Forward PDE
  f(uvec) # eval objective
end

# reverse rules

function rrule(::typeof(f),uvec)
  function f_pullback(dFdF)
    NO_FIELDS, dFdF*df_du(uvec)
  end
  f(uvec), f_pullback
end

function df_du(uvec)
  function dα(uh,du)
    eh = uh-u
    de = du
    ∫( 2*eh*de )dΩ
  end
  uh = FEFunction(U,uvec)
  assemble_vector(du->dα(uh,du),V)
end

function rrule(::typeof(g),pvec)
  uvec = g(pvec)
  function g_pullback(dFdu)
    NO_FIELDS, dg_dp(dFdu,uvec,pvec)
  end
  uvec, g_pullback
end

function dg_dp(dFdu,uvec,pvec)

  # Adjoint problem
  ph = FEFunction(P,pvec)
  A = assemble_matrix((λ,v)->a(ph,v,λ),U,V)
  λvec = A\dFdu

  # Total derivative wrt pvec
  da(p,u,v,dp) = ∫( (2*p*dp)*∇(v)⋅∇(u) )dΩ
  uh = FEFunction(U,uvec)
  λh = FEFunction(V,λvec)
  -1*assemble_vector((dp)->da(ph,uh,λh,dp),Q)
end

function rrule(::typeof(h),rvec)
  pvec = h(rvec)
  function h_pullback(dFdp)
    NO_FIELDS, dh_dr(dFdp)
  end
  pvec, h_pullback
end

function dh_dr(dFdp)
  A = assemble_matrix((p,q)->h1(q,p),P,Q)
  ϕvec = A\dFdp

  ϕh = FEFunction(S,ϕvec)
  assemble_vector((dr)->h1(dr,ϕh),S)
end

rvec = rand(num_free_dofs(R))
@show F(rvec)

pvec, h_pullback = rrule(h,rvec)
uvec, g_pullback = rrule(g,pvec)
obj, f_pullback = rrule(f,uvec)

dFdF = 1
_, dFdu = f_pullback(dFdF)
_, dFdp = g_pullback(dFdu)
_, dFdr = h_pullback(dFdp)

display(dFdr)

@show num_free_dofs(U)
@show num_free_dofs(P)
@show num_free_dofs(R)

using Zygote
dFdr2, = Zygote.gradient(F,rvec)

display(dFdr2)

end # module
