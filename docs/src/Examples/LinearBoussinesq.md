```@meta
EditURL = "../../../test/Examples/LinearBoussinesq.jl"
```

# Linearised Boussinesq equations on the cubed sphere manifold
The evolution of fluid velocity, density and bouyancy is:

This example solves the shallow water equations.
The evolution of the prognostic variables, $\widetilde{\boldsymbol{u}}$ and $\widetilde{\varphi}$ is:

```math
\begin{align*}
\partial_t \widetilde{\boldsymbol{u}} 	+ \omega (\widetilde{\boldsymbol{k}}\times \widetilde{\boldsymbol{u}} )
+ \nabla_{\mathcal{S}} \widetilde{\varphi}  - \widetilde{b}  \widetilde{\boldsymbol{k}}
 &=  0  \quad \text{in} \quad \mathcal{S}\times[0,t),
\\
\partial_t \widetilde{\varphi}  + c^2 \nabla_{\mathcal{S}} \cdot \widetilde{\boldsymbol{u}}
&= 0 	\quad  \text{on } \mathcal{S}\times (0,t),	\\
\partial_t \widetilde{b} + N^2 \widetilde{\boldsymbol{u}} \cdot \widetilde{\boldsymbol{k}} 	&= 0 \quad  \text{on } \mathcal{S}\times (0,t),
\end{align*}
```

where $\widetilde{\boldsymbol{k}}$ is the outward pointing normal vector to the surface

## Set up
First load all required pacakges. In this example, we will use a distributed model. So we also initialise MPI.

````julia 
using GridapGeosciences
using Gridap
using GridapDistributed
using GridapP4est
using GridapSolvers
using PartitionedArrays
using MPI

MPI.Init()
ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))
````

## Discrete model
To obtain a refined 3D parametric model, we pass $\ell$ levels of refinement to the vertical and horizontal:

````julia 
ℓ = 2
radius,thickness = 1.0, 0.19
octree3_model = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks,radius,thickness;
                  num_horizontal_uniform_refinements=ℓ,
                  num_vertical_uniform_refinements=ℓ);
model = octree3_model.parametric_dmodel
````

## Triangulation

````julia 
order = 1
Ω = Triangulation(model)
dΩ = Measure(Ω,4*(order+1))
````

## Finite element spaces
Define the finite element spaces, where the Hdiv space has no-flux boundary conditions:

````julia 
Q = TestFESpace(model, ReferenceFE(lagrangian,Float64,order); conformity=:L2)
P = TrialFESpace(Q)

V = TestFESpace(model, ReferenceFE(raviart_thomas,Float64,order);
    conformity=:HDiv,dirichlet_tags=["bottom_boundary",  "top_boundary"])
U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))

W = TestFESpace(model, ReferenceFE(lagrangian,Float64,order); conformity=:L2)
B = TrialFESpace(W)

Y = MultiFieldFESpace([V, Q, W])
X = MultiFieldFESpace([U, P, B])
````

## Initial conditions
Define the initial conditions as analytic julia functions that take the forward
map of the three dimensional cubed sphere:

````julia 
function φ₀(forward_map)
  function _p(α)
    0.0
  end
end

function b₀(forward_map)
  function b(α)
    x = forward_map(α)
    θ,ϕ,r = xyz2θϕr(x)

    θc = 2*π/3
    ϕc = 0.0
    d = 0.095
    ζ = 0.38

    k = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - radius

    q = acos( sin(ϕc)*sin(ϕ) + cos(ϕc)*cos(ϕ)*cos(θ-θc)    )
    s = d^2/(d^2 + q^2)
    s*sin( 2*π*k/ζ  )
  end
end

function u₀(forward_map)
  function u(α)
    x = forward_map(α)
    u_0 = 0.058
    u_0*VectorValue(-x[2],x[1],0.0)
  end
end


function ω(forward_map)
  function w(α)
    x = forward_map(α)
    θ,ϕ,r = xyz2θϕr(x)
    Ωr = 0.01
    2*Ωr*sin(ϕ)
  end
end
````

Define the associated ParametricCellField, and interpolate the initial condition
into the finite element space

````julia 
h_cf = ParametricCellField(φ₀,Ω)
u_cf = ParametricCellField(piola(u₀),Ω)
b_cf = ParametricCellField(b₀,Ω)
omega_cf = ParametricCellField(ω,Ω)

xh0 = interpolate([u_cf,h_cf,b_cf],X)

function Gridap.CellData.get_triangulation(a::GridapDistributed.DistributedMultiFieldCellField)
  trians = map(get_triangulation,a.field_fe_fun)
````

@check all(map(t -> t === first(trians), trians))

````julia 
  return first(trians)
end
````

## Weak forms
To define the weak forms, we extract the metric information, and define the
pushforward of the surface normal vector:

````julia 
g = ParametricCellField(metric,Ω)
meas = ParametricCellField(sqrtg,Ω)
covariant_basis_cf = ParametricCellField(covariant_basis,Ω)
_area_meas(p) = x->  forward_jacobian(p,x) ⋅ (inv_metric(p,x) ⋅ VectorValue(1,0,0))
area_meas(p) = x-> norm(_area_meas(p)(x))
normal_3D(p) = x-> (1/area_meas(p)(x) )*VectorValue(1,0,0)
normal_3D_cf = ParametricCellField(normal_3D,Ω)
````

Mass term:

````julia 
mass(t, (dtu,dtp,dtb), (v,q,r)) = ( ∫( (v⋅ (g⋅ dtu) )*(1/meas) )dΩ
                                  + ∫( (q*dtp)*meas )dΩ
                                  + ∫( (r*dtb)*meas )dΩ )
````

Velocity residual:

````julia 
resu(t,(u,p,b),(v,q,r)) = (
   ∫( omega_cf*( normal_3D_cf ×( g⋅u*(1/meas)  ) )⋅(g⋅v)*(1/meas)  )dΩ
  - ∫( p*(∇⋅v) )dΩ
  - ∫( b*(normal_3D_cf⋅v)  )dΩ
                          )
````

Pressure residual:

````julia 
c = 1.0
resp(t,(u,p,b),(v,q,r)) = ∫( c^2*(q*(∇⋅u)) )dΩ
````

Bouyancy residual:

````julia 
N = 1.48
resb(t,(u,p,b),(v,q,r)) = ∫( N^2*r*(normal_3D_cf⋅u)  )dΩ
````

Define the full transient system, and the transient operator:

````julia 
res(t,(u,p,b),(v,q,r)) = resu(t,(u,p,b),(v,q,r)) + resp(t,(u,p,b),(v,q,r)) + resb(t,(u,p,b),(v,q,r))
jac(t,(u,p,b),(du,dp,db),(v,q,r)) = resu(t,(du,dp,db),(v,q,r)) + resp(t,(du,dp,db),(v,q,r)) + resb(t,(du,dp,db),(v,q,r))
jac_t(t,(u,p,b),(dut,dpt,dbt),(v,q,r)) =  mass(t, (dut,dpt,dbt), (v,q,r))
opT = TransientSemilinearFEOperator(mass, res, (jac,jac_t), X, Y; constant_mass=true)
````

Transient parameters:

````julia 
t0 = 0.0
tF = 1.0
nsteps = 100
dt = tF/nsteps
````

Solve with SSP RK 3

````julia 
ls = LUSolver()
nls = GridapSolvers.NonlinearSolvers.NewtonSolver(ls;verbose=i_am_main(ranks))
solver = ThetaMethod(nls, dt, 0.5)
solT = solve(solver, opT, t0, tF, xh0)
````

## Post processing
Iterate and visualise the solution

````julia 
mkpath("output_path/results")

uh,ph,bh = xh0
uproj = covariant_basis_cf⋅(1/meas*uh)
writevtk_with_cell_geomap(geo_map_func(Ω),Ω,
    "output_path/results/results_0",
    cellfields=["uh"=>uproj, "ph"=>ph, "bh"=>bh],append=false)


it = iterate(solT)
while !isnothing(it)
  data, state = it
  t, xh = data

  println("t = $t")

  uh,ph,bh = xh
  uproj = covariant_basis_cf⋅(1/meas*uh)

  writevtk_with_cell_geomap(geo_map_func(Ω),Ω,
    "output_path/results/results_$t",
    cellfields=["uh"=>uproj, "ph"=>ph, "bh"=>bh],append=false)
  it = iterate(solT, state)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

