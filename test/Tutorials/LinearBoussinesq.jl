# # Linearised Boussinesq equations on the cubed sphere manifold (intrinsic formulation)
# The evolution of fluid velocity, density and buoyancy is:
#
# This example solves the linearised Boussinesq equations.
# The evolution of the prognostic variables, $\widetilde{\boldsymbol{u}}$, $\widetilde{\varphi}$, and $\widetilde{b}$ is:
#
# ```math
# \begin{align*}
# \partial_t \widetilde{\boldsymbol{u}} 	+ \omega (\widetilde{\boldsymbol{k}}\times \widetilde{\boldsymbol{u}} )
# + \nabla_{\mathcal{S}} \widetilde{\varphi}  - \widetilde{b}  \widetilde{\boldsymbol{k}}
#  &=  0  \quad \text{in} \quad \mathcal{S}\times[0,t),
# \\
# \partial_t \widetilde{\varphi}  + c^2 \nabla_{\mathcal{S}} \cdot \widetilde{\boldsymbol{u}}
# &= 0 	\quad  \text{on } \mathcal{S}\times (0,t),	\\
# \partial_t \widetilde{b} + N^2 \widetilde{\boldsymbol{u}} \cdot \widetilde{\boldsymbol{k}} 	&= 0 \quad  \text{on } \mathcal{S}\times (0,t),
# \end{align*}
# ```
#
# where $\widetilde{\boldsymbol{k}}$ is the outward pointing normal vector to the surface

# ## Set up
# First load all required packages. In this example, we will use a distributed model. So we also initialise MPI.
using GridapGeosciences
using Gridap
using GridapDistributed
using GridapSolvers
using PartitionedArrays
using MPI

MPI.Init()
ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))

# ## Discrete model
# To obtain a refined 3D parametric model, we pass $\ell$ levels of refinement to the vertical and horizontal:
ℓ = 2
radius,thickness = 1.0, 0.19
coarse_mesh = CubedSphereWithThicknessMesh(radius,thickness)
octree3_model = AtlasOctreeDistributedDiscreteModel(ranks, coarse_mesh, ℓ; manifold_style=IntrinsicManifold())


# ## Triangulation
order = 1
Ω = Triangulation(octree3_model)
dΩ = Measure(Ω,4*(order+1))

# ## Finite element spaces
# Define the finite element spaces, where the Hdiv space has no-flux boundary conditions:
Q = TestFESpace(octree3_model, ReferenceFE(lagrangian,Float64,order); conformity=:L2)
P = TrialFESpace(Q)

V = TestFESpace(octree3_model, ReferenceFE(raviart_thomas,Float64,order);
    conformity=:HDiv,dirichlet_tags=["bottom_boundary",  "top_boundary"])
U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))

W = TestFESpace(octree3_model, ReferenceFE(lagrangian,Float64,order); conformity=:L2)
B = TrialFESpace(W)

Y = MultiFieldFESpace([V, Q, W])
X = MultiFieldFESpace([U, P, B])

# ## Initial conditions
# Define the initial conditions as analytic julia functions that take the forward
# map of the three dimensional cubed sphere:
function φ₀(x)
  0.0
end

function b₀(x)
  θ,ϕ,_ = xyz2θϕr(x)
  θc = 2*π/3
  ϕc = 0.0
  d = 0.095
  ζ = 0.38
  k = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - radius
  q = acos(sin(ϕc)*sin(ϕ) + cos(ϕc)*cos(ϕ)*cos(θ-θc))
  s = d^2/(d^2 + q^2)
  s*sin(2*π*k/ζ)
end

function u₀(x)
  u_0 = 0.058
  u_0*VectorValue(-x[2],x[1],0.0)
end

function ω(x)
  _,ϕ,_ = xyz2θϕr(x)
  Ωr = 0.01
  2*Ωr*sin(ϕ)
end

# ## Weak forms
# To define the weak forms, we extract the metric information, and define the
# radial unit normal vector from the first parametric coordinate direction (1,0,0):
ambient_map_cf = AmbientMapCellField(Ω)
covariant_basis_cf = transpose∘∇(ambient_map_cf)
g = MetricCellField(Ω)
meas = MeasureCellField(Ω)
n3D = CellField(VectorValue(1.0,0.0,0.0),Ω)
ff = Operation(sqrt)(n3D ⋅ (InvMetricCellField(Ω) ⋅ n3D))
normal_3D_cf = n3D/ff

# Define the associated cell fields and interpolate the initial condition
# into the finite element space:
h_cf = φ₀∘ambient_map_cf
u_cf = meas*((pinvJ∘covariant_basis_cf)⋅(u₀∘ambient_map_cf))
b_cf = b₀∘ambient_map_cf
omega_cf = ω∘ambient_map_cf

xh0 = interpolate_everywhere([u_cf,h_cf,b_cf],X);

# Mass term:
mass(t, (dtu,dtp,dtb), (v,q,r)) = ( ∫( (v⋅ (g⋅ dtu) )*(1/meas) )dΩ
                                  + ∫( (q*dtp)*meas )dΩ
                                  + ∫( (r*dtb)*meas )dΩ )
# Velocity residual:
resu(t,(u,p,b),(v,q,r)) = (
   ∫( omega_cf*( normal_3D_cf ×( g⋅u*(1/meas)  ) )⋅(g⋅v)*(1/meas)  )dΩ
  - ∫( p*(∇⋅v) )dΩ
  - ∫( b*(normal_3D_cf⋅v)  )dΩ
                          )

# Pressure residual:
c = 1.0
resp(t,(u,p,b),(v,q,r)) = ∫( c^2*(q*(∇⋅u)) )dΩ

# Buoyancy residual:
N = 1.48
resb(t,(u,p,b),(v,q,r)) = ∫( N^2*r*(normal_3D_cf⋅u)  )dΩ

# Define the full transient system, and the transient operator:
res(t,(u,p,b),(v,q,r)) = resu(t,(u,p,b),(v,q,r)) + resp(t,(u,p,b),(v,q,r)) + resb(t,(u,p,b),(v,q,r))
jac(t,(u,p,b),(du,dp,db),(v,q,r)) = resu(t,(du,dp,db),(v,q,r)) + resp(t,(du,dp,db),(v,q,r)) + resb(t,(du,dp,db),(v,q,r))
jac_t(t,(u,p,b),(dut,dpt,dbt),(v,q,r)) =  mass(t, (dut,dpt,dbt), (v,q,r))
opT = TransientSemilinearFEOperator(mass, res, (jac,jac_t), X, Y; constant_mass=true)

# Transient parameters:
t0 = 0.0
tF = 1.0
nsteps = 100
dt = tF/nsteps

# Solve with SSP RK 3
ls = LUSolver()
nls = GridapSolvers.NonlinearSolvers.NewtonSolver(ls;verbose=i_am_main(ranks))
solver = ThetaMethod(nls, dt, 0.5)
solT = solve(solver, opT, t0, tF, xh0)

# ## Post processing
# Iterate and visualise the solution
mkpath("output_path/results")

uh,ph,bh = xh0;
uproj = covariant_basis_cf⋅(1/meas*uh)
writevtk_with_cell_geomap(AmbientMapCellField(Ω),Ω,
    "output_path/results/results_0",
    cellfields=["uh"=>uproj, "ph"=>ph, "bh"=>bh],append=false);


it = iterate(solT)
while !isnothing(it)
  data, state = it
  t, xh = data

  println("t = $t")

  uh,ph,bh = xh;
  uproj = covariant_basis_cf⋅(1/meas*uh)

  writevtk_with_cell_geomap(AmbientMapCellField(Ω),Ω,
    "output_path/results/results_$t",
    cellfields=["uh"=>uproj, "ph"=>ph, "bh"=>bh],append=false)
  it = iterate(solT, state)
end
