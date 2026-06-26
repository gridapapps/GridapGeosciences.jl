# # Advection with upwinding stabilisation on the cubed sphere manifold
#
# This example solves the scalar transport equation, given by
#
# ```math
# \begin{align*}
# \partial_t \widetilde{u} + \nabla_{\gamma} \cdot (\widetilde{\boldsymbol{\beta}} \widetilde{u} ) &= 0 \quad \text{in} \quad \gamma,
# \end{align*}
# ```
#
# where $\gamma$ is the cubed sphere manifold, $\widetilde{u}: \gamma \rightarrow \mathbb{R}$
# is a scalar valued functions defined in the ambient space of the manifold,
# $\widetilde{\boldsymbol{\beta}}\in T_p \gamma $ is a velocity field defined in the
# tangent space of the manifold,  and
# $\nabla_{\gamma}\cdot$ is the surface divergence operator.
#
# We use a discontinuous Galerkin method with upwinding to solve in the ambient space of the cubed sphere.
# For more information about the upwinding method for scalar transport, refer to
# [Brezzi et al. 2004](https://doi.org/10.1142/S0218202504003866).
#
#
# The weak formulation in the ambient space is: find $\widetilde{u}_h \in \mathbb{V} \subset L^2(\gamma)$ such that $\forall \widetilde{v}_h \in \mathbb{V}$
# ```math
# \begin{align*}
# a(\widetilde{u}_h,\widetilde{v}_h) &+ s(\widetilde{u}_h,\widetilde{v}_h) = 0  , \\
# a(\widetilde{u}_h,\widetilde{v}_h) &= \int_{\gamma} \partial_t \widetilde{u}_h \widetilde{v}_h
# - \int_{\gamma} \widetilde{u}_h \widetilde{\beta} \cdot \nabla_{\gamma} \widetilde{v}_h
# +  \int_{\widetilde{\mathcal{E}}_0} \frac{1}{2} 	[ \widetilde{u}_h \widetilde{\beta} \cdot \widetilde{n} ][ \widetilde{v}_h ] 		ds \\
# s({u}_h,{v}_h) 	&=\int_{\widetilde{\mathcal{E}}_0} \frac{1}{2}  |\widetilde{\beta}\cdot \widetilde{n}^+| [  \widetilde{u}_h ][   \widetilde{v}_h  ] ds
# \end{align*}
# ```
# where $[a] = a^+ - a^-$ for any scalar $a$.

# ## Set up
# First load all required packages. In this example, we will use a serial model
using GridapGeosciences
using Gridap
using GridapSolvers


# ## Discrete Model
# To obtain a refined 2D ambient model, we first define the coarse mesh and
# then apply $\ell$ levels of refinement:
radius = 1.0
ℓ = 2
coarse_mesh = CubedSphereMesh(radius)
model = AtlasDiscreteModel(coarse_mesh,ℓ,manifold_style=ExtrinsicManifold())

# ## Triangulation and FE spaces
# Triangulate the model, both volume and skeleton
order = 1
degree = 2*(order+1)
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ,degree)
n_Λ = get_normal_vector(Λ)


# ## Finite element spaces
# Extract the finite element spaces in the typical way:
Q = TestFESpace(model, ReferenceFE(lagrangian,Float64,order); conformity=:L2)
P = TransientTrialFESpace(Q)

# ## Initial conditions and velocity field
# The velocity field is a solid body rotation, that can be converted to a CellField
# in the standard way:
vX(x) = VectorValue(-x[2],x[1],0.0)
vel =  CellField(vX,Ω)

# The initial condition is a gaussian bump, defined as an analytic function:
u(x) = exp(-(x[2]^2 + x[3]^2))
# We convert this initial condition to a CellField, and interpolate it into the
# finite element space.
u_cf = CellField(u,Ω)
uh0 = interpolate_everywhere(u_cf, P(0.0))


# ## Transient weak form
# To define the transient weak form, we follow the methodology of transient problems
# in here, refer [here](https://gridap.github.io/Tutorials/dev/pages/t017_transient_linear/)
a_mass(t,dtu,v) = ∫( (dtu*v)  )dΩ
a_Ω(u,v) =   ∫( -(u*(∇(v)⋅vel) )  )dΩ
a_s1(u,v) = ∫( mean(vel*u)⋅jump(v*n_Λ )  )dΛ
upwind = abs((vel⋅ n_Λ).plus)/2
a_s2(u,v) = ∫(  upwind*jump(u*n_Λ )⋅jump(v*n_Λ )   )dΛ

res(t,u,v) =  a_Ω(u,v) + a_s1(u,v) + a_s2(u,v)
jac(t,u,du,v) = a_Ω(du,v) + a_s1(du,v) + a_s2(du,v)
jac_t(t,u,dtu,v) = ∫( (dtu*v) )dΩ
opT = TransientSemilinearFEOperator(a_mass, res, (jac,jac_t), P, Q, constant_mass=true)

# The transient parameters are:
t0 = 0.0
tF = 2*π
nsteps = 100
dt = tF/nsteps
τ = 0.5*dt

# The transient solution is obtained using a Runge Kutta method:
ls = LUSolver()
nls = GridapSolvers.NonlinearSolvers.NewtonSolver(ls;rtol=1.e-12,verbose=true)
solver = RungeKutta(nls, ls, dt, :SDIRK_Crouzeix_3_4)
solT = solve(solver, opT, t0, tF, uh0)

# ## Post processing
# Iterate the solution, and visualise the solution
mkpath("output_path/results")
createpvd("output_path/results") do pvd
  pvd[0] = createvtk(Ω, "output_path/results/results_0" * ".vtu", cellfields=["u" => uh0],append=false)
  for (t, uh) in solT
    println("t = $t")
    pvd[t] = createvtk(Ω, "output_path/results/results_$t" * ".vtu", cellfields=["u" => uh],append=false)
  end
end
