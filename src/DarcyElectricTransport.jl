using Gridap
import Gridap: ∇
using LinearAlgebra: tr
# Here we start writing our driver
#
#=================================================#
# Domain
domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)
#
#=================================================#
# Polinomial order
order = 1
#
#=================================================#
# Dirichlet boundary conditions
uD  = VectorValue(0.0,0.0)
pD0 = 1
pD1 = 0
cD0 = 1
cD1 = 0
#
#=================================================#
# Finite Element Spaces
V = TestFESpace(reffe=:RaviartThomas, order=order, valuetype=VectorValue{2,Float64},
                conformity=:HDiv,     model=model, dirichlet_tags=[5,6])
Q = TestFESpace(reffe=:QLagrangian, order=order, valuetype=Float64,
                conformity=:L2,     model=model)
W = TestFESpace(reffe=:Lagrangian, order=order, valuetype=Float64,
                conformity=:H1,    model=model, dirichlet_tags=[1,3,7,2,4,8])
Ξ = TestFESpace(reffe=:Lagrangian,    order=order, valuetype=Float64,
                conformity=:H1,       model=model, dirichlet_tags=[1])
#
U = TrialFESpace(V,uD)
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
# Vertical unitary vector
const ∇z = VectorValue(0.0,1.0)         
# Hydraulic resistivity
const kinv1 = TensorValue(1.0,0.0,0.0,1.0)
const kinv2 = TensorValue(100.0,90.0,90.0,100.0)
@law function κ(x,u)
   if ((abs(x[1]-0.5) <= 0.1) && (abs(x[2]-0.5) <= 0.1))
      return kinv2*u
   else
      return kinv1*u
   end
end
# Density functions
const ρ0 = 1000.0 # Denisty water
const ρ1 = 1025.0 # Denisty brine
@law ρ(c)   = ρ0 + (ρ1-ρ0)*c
@law Δρ(c)  = (ρ1-ρ0)*c
@law ∇ρ(∇c)  = (ρ1-ρ0)*∇c
@law dρ(dc) = (ρ1-ρ0)*dc
#
const I = TensorValue(1.0,0.0,0.0,1.0)
# Diffusion coefficient
const Dm  = 1.0
const α_L = 0.0
const α_T = 0.0
@law D(u) = Dm*I #+ (α_L-α_T)*outer(u,u)/sqrt(inner(u,u)) + α_T*sqrt(inner(u,u))*I
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
   q*∇ρ(∇(c))*u + q*ρ(c)*(∇*u) + ( v*κ(px,u) ) - (∇*v)*p + v*(1/ρ0*Δρ(c)*∇z)
end
# Jacobian dr/du
function jac_darcy(u,du,p,dp,c,dc,v,q)
   q*∇ρ(∇(c))*du + q*ρ(c)*(∇*du) + ( v*κ(px,du) ) - (∇*v)*dp + q*∇ρ(∇(dc))*u + q*ρ(dc)*(∇*u) + v*(1/ρ0*Δρ(dc)*∇z)
end
#
# Transport
# Residual r
function res_transport(u,c,w)
   w*ρ(c)*u*∇(c) + ∇(w)*(ρ(c)*D(u)*∇(c))
end
# Jacobian dr/du
function jac_transport(u,du,c,dc,w)
   w*ρ(c)*u*∇(dc) + ∇(w)*(ρ(c)*D(u)*∇(dc)) + w*ρ(c)*du*∇(c) + ∇(w)*(ρ(c)*D(du)*∇(c)) + w*∇ρ(dc)*u*∇(c) + ∇(w)*(∇ρ(dc)*D(u)*∇(c))
end
#
# Electrostatics
const σ_s = 0.01
const σ_m = 1.0
n_src = 2
xsrc = zeros((2,2))
xsrc[1,:] = [0.4,1.0]
xsrc[2,:] = [0.6,1.0]
@law function fe(x)  # Source as exponential function
  s = 0
  for i in 1:n_src
    s += ((-1)^(i+1))*σ_m*exp(-((x[1]-xsrc[i,1])^2+(x[2]-xsrc[i,2])^2)/(2*σ_s^2))  
  end
  return s
end
# Residual r
function res_electrostatics(c,ϕ,ξ)
   ∇(ξ)*σ(c,∇(ϕ)) + ξ*fe(px)
end
# Jacobian dr/du
function jac_electrostatics(c,ϕ,dc,dϕ,ξ)
   ∇(ξ)*σ(c,∇(dϕ)) + ∇(ξ)*σ(dc,∇(ϕ))
end
#
#=================================================#
# Global system
function res(x,y)
  u, p, c, ϕ = x
  v, q, w, ξ = y
  res_darcy(u,p,c,v,q) + res_transport(u,c,w) + res_electrostatics(c,ϕ,ξ)
end

function jac(x,dx,y)
  u,  p,  c, ϕ  = x
  v,  q,  w, ξ  = y
  du, dp, dc, dϕ = dx
  jac_darcy(u,du,p,dp,c,dc,v,q) + jac_transport(u,du,c,dc,w) + jac_electrostatics(c,ϕ,dc,dϕ,ξ)
end

#
#=================================================#
# Here add boundary terms (e.g. weak bc's, etc)
neumanntags = [7,]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,degree)

nb = get_normal_vector(btrian)
function b_ΓN(y)
  v, q = y
  (v*nb)*(-pD0) # Negative beause of test function vector direction??
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
writevtk(trian,"darcyresults",cellfields=["uh"=>uh,"ph"=>ph,"ch"=>ch,"ϕh"=>ϕh])