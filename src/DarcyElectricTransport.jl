using Gridap
import Gridap: ∇
# Here we start writing our driver

# Domain
domain = (0,1,0,1)
partition = (100,100)
model = CartesianDiscreteModel(domain,partition)

# Analytical functions

u(x) = x[1]+x[2]
∇u(x) = VectorValue(1,1)
∇(::typeof(u)) = ∇u
Δu(x) = 0.0
using LinearAlgebra: tr
divu(x) = tr(∇u(x))

p(x) = ...
∇p(x) = ...

ϕ(x) = ...
∇ϕ(x) = ...

c(x) = ...
∇c(x) = ...

# Spaces

order = 1

V = TestFESpace(
  reffe=:RaviartThomas, order=order, valuetype=VectorValue{2,Float64},
  conformity=:HDiv, model=model, dirichlet_tags=[5,6])

U = TrialFESpace(V,u)

Q = TestFESpace(
  reffe=:QLagrangian, order=order, valuetype=Float64,
  conformity=:L2, model=model)

P = TrialFESpace(Q)

Ξ = TestFESpace(
  reffe=:Lagrangian, order=order, valuetype=Float64,
  conformity=:HDiv, model=model, dirichlet_tags=[5,6])

Φ = TrialFESpace(Ξ,ϕ)

C = TestFESpace(
  reffe=:Lagrangian, order=order, valuetype=Float64,
  conformity=:L2, model=model) #e.g. using DG here

D = TrialFESpace(C)

X = MultiFieldFESpace([U, P, Ξ, C])
Y = MultiFieldFESpace([V, Q, Φ, D])

# Darcy problem

const κ = TensorValue(1.0,0.0,0.0,1.0)
f(x) = κ*u(x) + ∇p(x)
j(x) = divu(x)


function res_darcy(u,p,v,q)
   v*(κ*u) - (∇*v)*p + q*(∇*u) - v*g - q*j
end

function jac_darcy(u,du,p,dp,v,q)
   v*(κ*du) - (∇*v)*dp + q*(∇*du)
end

# Electrostatics

# Tranport

g(x) = u(x)*∇c(x)

const Pe = 10.0

@law conv(u,∇c) = Pe*(∇c')*u
@law dconv(du,∇dc,u,∇c) = conv(u,∇dc)+conv(du,∇c)

a(c,d) = inner(∇(c),∇(d))
c(u,c,d) = d*conv(u,∇(c))
dc(u,c,du,dc,v) = v*dconv(du,∇(dc),u,∇(c))

function res_transport(u,c,d)
  a(c,d) + c(u,c,d) - d*g
end

function jac_transport(u,du,c,dc,d)
  a(dc,d)+ dc(u,c,du,dc,v)
end

# Missing DG terms (see DG tutorials... or add SUPG and make the space continuous)

# Check also how to use @law in tutorials for your constitutive models

# whole problem

function res(x,y)
  u, p, ϕ, c = x
  v, q, ξ, d = y
  res_darcy(u,p,v,q) + res_tranport(u,c,d) + res_electric(ϕ,ξ)
end

function jac(x,dx,y)
  u, p, ϕ, c = x
  du, dp, dϕ, dc = x
  v, q, ξ, d = y
  jac_darcy(u,du,p,dp,v,q) + jac_tranport(u,du,c,dc,d) + jac_electric(dϕ,ξ)
end

trian = Triangulation(model)
degree = order*2
quad = CellQuadrature(trian,degree)
t_Ω = FETerm(res,jac,trian,quad)
op = FEOperator(X,Y,t_Ω)

neumanntags = [8,]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,degree)

# Here add boundary terms (e.g. weak bc's, etc)


# Solve the problem
using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

uh, ph, ϕh, ch = solve(solver,op)

writevtk(trian,"ins-results",cellfields=["uh"=>uh,"ph"=>ph,"ϕh"=>ϕh,"ch"=>ch])
using Base.Algebr
