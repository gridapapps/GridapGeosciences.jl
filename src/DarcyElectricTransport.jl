using Gridap
using ForwardDiff
using LinearAlgebra: tr
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using WriteVTK
#using Plots
#
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t
#
# Here we start writing our driver
#
#=================================================#
# - Subroutines
#=================================================#
# PostProcessing
# Write to vtk
function writePVD(filePath::String, trian::Triangulation, sol; append=false)
    outfiles = paraview_collection(filePath, append=append) do pvd
        for (i, (xh, t)) in enumerate(sol)
            uh = xh.blocks[1]
            ph = xh.blocks[2]
            ch = xh.blocks[3]
            ϕh = xh.blocks[4]
            #uh, ph, ch, ϕh = xh
            #
            #writevtk(trian,"darcyresults",cellfields=["uh"=>uh,"ph"=>ph,"ch"=>ch,"ψh"=>ϕh])
            pvd[t] = createvtk(
                trian,
                filePath * "_$t.vtu",
                cellfields = ["uh"=>uh,"ph"=>ph,"ch"=>ch,"ψh"=>ϕh],
            )
        end
    end
end
#
#=================================================#
# - Main
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
cD0(x,t) = 0.0
cD0(t::Real) = x -> cD0(x,t)
# - cr = 1.0
cD1(x,t) = 1.0
cD1(t::Real) = x -> cD1(x,t)
#
#=================================================#
# Time discretization
θ  = 0.5
t0 = 0.0
tF = 1.0
dt = 0.1
#
#=================================================#
# Finite Element Spaces
V = TestFESpace(reffe=:RaviartThomas,
                order=order,
                valuetype=VectorValue{2,Float64},
                conformity=:HDiv,
                model=model,
                dirichlet_tags=[7,1,2,3,4,5,6])
Q = TestFESpace(reffe=:QLagrangian,
                order=order,
                valuetype=Float64,
                conformity=:L2,
                model=model)
W = FESpace(    reffe=:Lagrangian,
                order=order,
                valuetype=Float64,
                conformity=:H1,
                model=model,
                dirichlet_tags=[1,3,7,2,4,8])
Ξ = TestFESpace(reffe=:Lagrangian,
                order=order,
                valuetype=Float64,
                conformity=:H1,
                model=model,
                dirichlet_tags=[1])
#
U  = TrialFESpace(V,[uin,unp,unp,unp,unp,unp,unp])
P  = TrialFESpace(Q)
C  = TransientTrialFESpace(W,[cD0,cD0,cD0,cD1,cD1,cD1])
Φ  = TrialFESpace(Ξ,[0.0])
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
const ϕ = 0.35
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
const α_L = 0.001
const α_T = 0.001
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
function res_transport(u,c,ct,w)
   w*ϕ*ρ(c)*ct + w*ρ(c)*u*∇(c) + ∇(w)*(ρ(c)*D(u)*∇(c))
end
# Jacobian dr/du
function jac_transport(u,du,c,ct,dc,w)
   w*ϕ*dρ(dc)*ct + w*ρ(c)*u*∇(dc) + ∇(w)*(ρ(c)*D(u)*∇(dc)) + w*ρ(c)*du*∇(c) + ∇(w)*(ρ(c)*dD(u,du)*∇(c)) + w*dρ(dc)*u*∇(c) + ∇(w)*(dρ(dc)*D(u)*∇(c))
end
# Jacobian tdr/du
function jac_transport_t(c,dct,w)
   w*ϕ*ρ(c)*dct
end
#
# Electrostatics
const σ_s   = 0.01
const σ_m   = 1.0
const n_src = 2
const xsrc  = (0.4,1.0,0.6,1.0)
const ge    = 0
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
# Residual ∂r
function ∂res_electrostatics(c,ψ,ξ)
   ξ*σ(c,ψ) - ξ*ge
end
# Jacobian ∂dr/du
function ∂jac_electrostatics(c,ψ,dc,dψ,ξ)
   ξ*σ(c,dψ) + ξ*σ(dc,ψ)
end
#
#=================================================#
# Global system
function res(t, x, xt, y)
  u,   p,   c,   ψ   = x
  ut,  pt,  ct,  ψt  = xt
  v,   q,   w,   ξ   = y
  res_darcy(u,p,c,v,q) + res_transport(u,c,ct,w) + res_electrostatics(c,ψ,ξ)
end
function jac(t, x, xt, dx, y)
  u,   p,   c,   ψ   = x
  ut,  pt,  ct,  ψt  = xt
  du,  dp,  dc,  dψ  = dx
  v,   q,   w,   ξ   = y
  jac_darcy(u,du,p,dp,c,dc,v,q) + jac_transport(u,du,c,ct,dc,w) + jac_electrostatics(c,ψ,dc,dψ,ξ)
end
function jac_t(t, x, xt, dxt, y)
  u,    p,    c,    ψ    = x
  dut,  dpt,  dct,  dψt  = dxt
  v,    q,    w,    ξ    = y
  jac_transport_t(c,dct,w)
end
# Boundary system
function ∂res(x, y)
  u,  p,  c,  ψ  = x
  v,  q,  w,  ξ  = y
  ∂res_electrostatics(c,ψ,ξ)
end
function ∂jac(x, dx, y)
  u,  p,  c,  ψ  = x
  du, dp, dc, dψ = dx
  v,  q,  w,  ξ  = y
  ∂jac_electrostatics(c,ψ,dc,dψ,ξ)
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
#=================================================#
# Here we create the triangulation and the quadrature for the robin condition
#
robinntags = [5,6,7]
btrian_r = BoundaryTriangulation(model,)
bquad_r = CellQuadrature(btrian_r,degree)
#
#=================================================#
# Set the system
t_Ω = FETerm(res,jac,jac_t,trian,quad)
t_∂Ω = FETerm(∂res,∂jac,btrian_r,bquad_r)
t_ΓN = FESource(b_ΓN,btrian,bquad)
op = TransientFEOperator(X,Y,t_Ω,t_∂Ω,t_ΓN)
#
#=================================================#
# Set solver
#using LineSearches: BackTracking
nls = NLSolver(show_trace=true,
               method=:newton,)
#               linesearch = BackTracking(),)
#
#=================================================#
# Initial Conditions
# Set based on Steady state solution
Y0 = MultiFieldFESpace([V, Q])
X0 = MultiFieldFESpace([U, P])
#
function res_darcy0(x,y)
   u,   p   = x
   v,   q   = y
   c        = 0.0
   q*ρ(c)*(∇*u)  + ( v*κ(px,u) ) - (∇*v)*p  + v*(1/ρ0*(Δρ*c)*∇z)
end
# Jacobian dr/du
function jac_darcy0(x, dx, y)
   u,   p   = x
   du,  dp  = dx
   v,   q   = y
   c        = 0.0
   q*ρ(c)*(∇*du) + ( v*κ(px,du) ) - (∇*v)*dp
end
function b0_ΓN(y)
  v,  q  = y
  #
  - inner(v*nb,pD1)
end
#
t0_Ω  = FETerm(res_darcy0,jac_darcy0,trian,quad)
t0_ΓN = FESource(b0_ΓN,btrian,bquad)
op0   = FEOperator(X0,Y0,t0_Ω,t0_ΓN)
#
solver0 = FESolver(nls)
xh0 = solve(solver0,op0)
uh0, ph0 = xh0
writevtk(trian,"Initial_conditions",cellfields=["uh"=>uh0,"ph"=>ph0])
# Set based on funtions
c(x,t) = 0.0
c(t::Real) = x -> c(x,t)
ψ(x,t) = 0.0
ψ(t::Real) = x -> ψ(x,t)
C0 = C(0.0)
Φ0 = Φ(0.0)
ch0 = interpolate_everywhere(C0,c(0.0))
ψh0 = interpolate_everywhere(Φ0,ψ(0.0))
#
C0  = TrialFESpace(W,[cD0(t0),cD0(t0),cD0(t0),cD1(t0),cD1(t0),cD1(t0)])
X00 = MultiFieldFESpace([U, P, C0, Φ])
sol_t0 = Gridap.MultiField.MultiFieldFEFunction(X00,[uh0,ph0,ch0,ψh0])
#=================================================#
# Initialize Paraview files
folderName = "Results"
fileName = "fields"
if !isdir(folderName)
    mkdir(folderName)
end
filePath = join([folderName, fileName], "/")
#
# Print Initial conditon
println("writeStokes")
#writePVD(filePath, trian, [(sol_t0, 0.0)])
#
#=================================================#
# Solve system
odes   = ThetaMethod(nls, dt, θ)
solver = TransientFESolver(odes)
sol_t  = solve(solver, op, xh0, t0, tF)
#
#=================================================#
# Output transient solution to Paraview
println("writeNavierStokes 1")
#writePVD(filePath, trian, sol_t, append=true)
#=================================================#
