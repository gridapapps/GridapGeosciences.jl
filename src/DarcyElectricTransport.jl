using Gridap
using ForwardDiff
using LinearAlgebra: tr
using GridapODEs.ODETools
using GridapODEs.TransientFETools
#
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t
#
# Here we start writing our driver
#
#=================================================#
# Domain
domain = (0,2,0,1)
partition = (80,40)
model = CartesianDiscreteModel(domain,partition)
#
#=================================================#
# Polinomial order
order = 1
#
#=================================================#
# Dirichlet boundary conditions
# - Unp = 0
unp = VectorValue(0.0,0.0)
# - Uin = -3.3e-5
uin = VectorValue(3.3e-5,0.0)
# - cl = 0.0
cD0 = 0.0
# - cr = 1.0
cD1 = 1.0
#
#=================================================#
# Finite Element Spaces
V = TestFESpace(reffe=:RaviartThomas, order=order, valuetype=VectorValue{2,Float64},
                conformity=:HDiv,     model=model, dirichlet_tags=[7,1,2,3,4,5,6])
Q = TestFESpace(reffe=:QLagrangian,   order=order, valuetype=Float64,
                conformity=:L2,       model=model)
W = TestFESpace(reffe=:Lagrangian,    order=order, valuetype=Float64,
                conformity=:H1,       model=model, dirichlet_tags=[1,3,7,2,4,8])
Ξ = TestFESpace(reffe=:Lagrangian,    order=order, valuetype=Float64,
                conformity=:H1,       model=model, dirichlet_tags=[1])
#
U = TrialFESpace(V,[uin,unp,unp,unp,unp,unp,unp])
P = TrialFESpace(Q)
C = TrialFESpace(W,[cD0,cD0,cD0,cD1,cD1,cD1])
Φ = TrialFESpace(Ξ,[0.0])
#
Y = MultiFieldFESpace([V, Q, W, Ξ])
X = MultiFieldFESpace([U, P, C, Φ])
#
#=================================================#
trian = Triangulation(model)
degree = order*2
quad = CellQuadrature(trian,degree)
#
#=================================================#
# Storativity
const S = 6.87e-4   
# Porosity          
const phi = 0.35                
# Vertical unitary vector
const ∇z = VectorValue(0.0,1.0)         
# Hydraulic resistivity
const kinv1 = TensorValue(100.0,0.0,0.0,100.0)
@law function κ(x,u)
   return kinv1*u
end
# Density functions
const ρ0 = 1000.0 # Denisty water
const ρ1 = 1025.0 # Denisty brine
const Δρ = ρ1-ρ0
@law  ρ(c)  = ρ0 + Δρ*c
@law ∇ρ(∇c) =      Δρ*∇c
@law dρ(c)  =      Δρ*c
#
const I = TensorValue(1.0,0.0,0.0,1.0)
# Dispersive coefficient
const Dm  = 1.886e-5
const α_L = 0.0
const α_T = 0.0
@law  function D(u)
   nrm  = sqrt(inner(u,u))
   uou  = outer(u,u)
   if (nrm == 0)
      return Dm*I
   else
      return Dm*I + (α_L-α_T)*uou/nrm + α_T*nrm*I
   end
end
@law function dD(u,du)
   nrm  = sqrt(inner(u,u))
   uou  = outer(u,u)
   uodu = outer(u,du)
   nrm3 = nrm*nrm*nrm
   if (nrm == 0)
      return 0.0*I
   else
      return (α_L-α_T)*( 2*uodu/nrm - uou*inner(du,u)/nrm3 ) + α_T*inner(du,u)/nrm*I
   end
end
# Not working full Dispersion. I don't understand why
#
# Electric conductivity
const σ0 = 0.001                 # Conductivity of the freshwater
const σ1 = 0.072                 # Conductivity of the fluid at saturation
const σ_v = TensorValue(1.0,0.0,0.0,1.0)
@law function σ(c,u)
   aux = (σ0 + c*σ1)*1.0
   sol = aux*u
   return sol
end
#
#=================================================#
#
px = get_physical_coordinate(trian)
# Darcy
# Residual r
function res_darcy(u,p,c,v,q)
   q*∇ρ(∇(c))*u  + q*ρ(c)*(∇*u)  + ( v*κ(px,u) )  - (∇*v)*p  + v*(1/ρ0*(Δρ*c)*∇z)
end
# Jacobian dr/du
function jac_darcy(u,du,p,dp,c,dc,v,q)
   q*∇ρ(∇(c))*du + q*ρ(c)*(∇*du) + ( v*κ(px,du) ) - (∇*v)*dp + q*∇ρ(∇(dc))*u + q*dρ(dc)*(∇*u) + v*(1/ρ0*(Δρ*dc)*∇z)
end
#
# Transport
# Residual r
function res_transport(u,c,w)
   w*ρ(c)*u*∇(c) + ∇(w)*(ρ(c)*D(u)*∇(c))
end
# Jacobian dr/du
function jac_transport(u,du,c,dc,w)
   w*ρ(c)*u*∇(dc) + ∇(w)*(ρ(c)*D(u)*∇(dc)) + w*ρ(c)*du*∇(c) + ∇(w)*(ρ(c)*dD(u,du)*∇(c)) + w*dρ(dc)*u*∇(c) + ∇(w)*(dρ(dc)*D(u)*∇(c))
end
#
# Electrostatics
const σ_s = 0.01
const σ_m = 1.0
const n_src = 2
const xsrc = (0.4,1.0,0.6,1.0)
function fe(x)  # Source as exponential function
  s = 0
  for i in 1:n_src
    s += ((-1)^(i+1))*σ_m*exp(-((x[1]-xsrc[2*i-1])^2+(x[2]-xsrc[2*i])^2)/(2*σ_s^2))  
  end
  return s
end
# Residual r
function res_electrostatics(c,ψ,ξ)
   ∇(ξ)*σ(c,∇(ψ)) - ξ*fe 
end
# Jacobian dr/du
function jac_electrostatics(c,ψ,dc,dψ,ξ)
   ∇(ξ)*σ(c,∇(dψ)) + ∇(ξ)*σ(dc,∇(ψ))
end
#
#=================================================#
# Global system
function res(x,y)
  u,  p,  c,  ψ  = x
  v,  q,  w,  ξ  = y
  res_darcy(u,p,c,v,q) + res_transport(u,c,w) + res_electrostatics(c,ψ,ξ)
end

function jac(x,dx,y)
  u,  p,  c,  ψ  = x
  du, dp, dc, dψ = dx
  v,  q,  w,  ξ  = y
  jac_darcy(u,du,p,dp,c,dc,v,q) + jac_transport(u,du,c,dc,w) + jac_electrostatics(c,ψ,dc,dψ,ξ)
end
#
#=================================================#
# Here add boundary terms (e.g. weak bc's, etc)
neumanntags = [8]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,degree)
nb = get_normal_vector(btrian)
# - hr = hydro_____ head
pD1(x) = ρ1/ρ0-(ρ1-ρ0)/ρ0*x[2]
# Set boundary
function b_ΓN(y)
  v,  q,  w,  ξ  = y
  #
  - inner(v*nb,pD1)
end
#
#=================================================#
#t_Ω = LinearFETerm(a,trian,quad)
t_Ω = FETerm(res,jac,trian,quad)
t_ΓN = FESource(b_ΓN,btrian,bquad)
#op = AffineFEOperator(X,Y,t_Ω,t_ΓN)
op = FEOperator(X,Y,t_Ω,t_ΓN)
#
#=================================================#
using LineSearches: BackTracking
nls = NLSolver(show_trace=true, method=:newton)
solver = FESolver(nls)
xh = solve(solver,op)
#xh = solve(op)
uh, ph, ch, ϕh = xh
#
#=================================================#
writevtk(trian,"darcyresults",cellfields=["uh"=>uh,"ph"=>ph,"ch"=>ch,"ψh"=>ϕh])