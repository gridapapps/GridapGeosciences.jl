```@meta
EditURL = "../../../test/Examples/AmbientAdvectionSUPG.jl"
```

# Advection with SUPG stabilisation on the cubed sphere manifold

This example solves the scalar transport equation, given by

```math
\begin{align*}
\partial_t \widetilde{u} + \nabla_{\gamma} \cdot (\widetilde{\boldsymbol{\beta}} \widetilde{u} ) &= 0 \quad \text{in} \quad \gamma,
\end{align*}
```

where $\gamma$ is the cubed sphere manifold, $\widetilde{u}: \gamma \rightarrow \mathbb{R}$
is a scalar valued functions defined in the ambient space of the manifold,
$\widetilde{\boldsymbol{\beta}}\in T_p \gamma $ is a velocity field defined in the
tangent space of the manifold,  and
$\nabla_{\gamma}\cdot$ is the surface divergence operator.

We use a SUPG method to solve in the ambient space of the cubed sphere.
For more information about the SUPG method for scalar transport, refer to
[Brookes \& Huges 1982](https://doi.org/10.1016/0045-7825(82)90071-8).

The weak formulation in the ambient space is: find $\widetilde{u}_h \in \mathbb{V} \subset H^1(\gamma)$ such that $\forall \widetilde{v}_h \in \mathbb{V}$
```math
\begin{align*}
a(\widetilde{u}_h,\widetilde{v}_h) &+ s(\widetilde{u}_h,\widetilde{v}_h) = 0  , \\
a(\widetilde{u}_h,\widetilde{v}_h) &= \int_{\gamma} \partial_t \widetilde{u}_h \widetilde{v}_h
+ \int_{\gamma} \widetilde{\beta} \cdot \nabla_{\gamma} \widetilde{u}_h ~\widetilde{v}_h    \\
s(\widetilde{u}_h,\widetilde{v}_h) 	&= \int_{\gamma} \partial_t \widetilde{u}_h (\beta \cdot \nabla_{\gamma} \widetilde{v}_h)
+ \int_{\gamma} \widetilde{\beta} \cdot \nabla_{\gamma} \widetilde{u}_h ~(\beta \cdot \nabla_{\gamma} \widetilde{v}_h )   \\
\end{align*}
```

## Set up
First load all required pacakges. In this example, we will use a serial model

````julia 
using GridapGeosciences
using Gridap
using GridapSolvers
````

## Discrete Model
To obtain a refined 2D ambient model, we pass $\ell$ levels of refinement:

````julia 
radius = 1.0
ℓ = 2
model = CubedSphereAmbientDiscreteModel(radius;num_initial_uniform_refinements=ℓ)
````

## Triangulation and FE spaces
Triangulate the model and extract finite element spaces in the typical way:

````julia 
order = 1
Ω = Triangulation(model)
dΩ = Measure(Ω,2*(order+1))
Q = TestFESpace(model, ReferenceFE(lagrangian,Float64,order); conformity=:H1)
P = TransientTrialFESpace(Q)
````

## Initial conditions and velocity field
The velocity field is a solid body rotation, that can be converted to a CellField
in the standard way:

````julia 
vX(x) = VectorValue(-x[2],x[1],0.0)
vel =  CellField(vX,Ω)
````

The initial condition is a gaussian bump, defined as an analytic function:

````julia 
u(x) = exp(-(x[2]^2 + x[3]^2))
````

We convert this initial condition to a CellField, and interpolate it into the
finite element space.

````julia 
u_cf = CellField(u,Ω)
uh0 = interpolate_everywhere(u_cf, P(0.0))
````

## Transient weak form
To define the transient weak form, we follow the metholody of transient problems
in here, refer [here](https://gridap.github.io/Tutorials/dev/pages/t017_transient_linear/)

````julia 
a_mass_Ω(dtu,v) = ∫( (dtu*v) )dΩ
a_mass_s(dtu,v) = ∫( (dtu*(vel⋅∇(v))) )dΩ
a_Ω(u,v) = ∫( ((vel⋅∇(u))*v ) )dΩ
a_s(u,v) =  ∫( ((vel⋅∇(u))*(vel⋅∇(v)) ) )dΩ

a_mass(t,dtu,v) = a_mass_Ω(dtu,v) + τ*a_mass_s(dtu,v)
res(t,u,v) =  a_Ω(u,v) + τ*a_s(u,v)
jac(t,u,du,v) = a_Ω(du,v) + τ*a_s(du,v)
jac_t(t,u,dtu,v) = a_mass_Ω(dtu,v) + τ*a_mass_s(dtu,v)
opT = TransientSemilinearFEOperator(a_mass, res, (jac,jac_t), P, Q, constant_mass=true)
````

The transient parameters are:

````julia 
t0 = 0.0
tF = 2*π
nsteps = 100
dt = tF/nsteps
τ = 0.5*dt
````

The transient solution is obtained using a Runge Kutta method:

````julia 
ls = LUSolver()
nls = GridapSolvers.NonlinearSolvers.NewtonSolver(ls;rtol=1.e-12,verbose=true)
solver = RungeKutta(nls, ls, dt, :SDIRK_Crouzeix_3_4)
solT = solve(solver, opT, t0, tF, uh0)
````

## Post processing
Iterate the solution, and visualise the solution

````julia 
mkpath("output_path/results")
createpvd("output_path/results") do pvd
  pvd[0] = createvtk(Ω, "output_path/results/results_0" * ".vtu", cellfields=["u" => uh0],append=false)
  for (t, uh) in solT
    println("t = $t")
    pvd[t] = createvtk(Ω, "output_path/results/results_$t" * ".vtu", cellfields=["u" => uh],append=false)
  end
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

