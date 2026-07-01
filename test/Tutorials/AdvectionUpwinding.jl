# # Advection with upwinding stabilisation on the cubed sphere manifold (intrinsic formulation)
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
# We use a discontinuous Galerkin method with upwinding to solve in the parametric space of the cubed sphere.
# For more information about the upwinding method for scalar transport, refer to
# [Brezzi et al. 2004](https://doi.org/10.1142/S0218202504003866).
#
# The weak formulation in the parametric space is: find $u_h \in \mathbb{V} \subset L^2(\mathcal{V})$ such that
# ```math
# \begin{align*}
# a(u_h,v_h) &+ b(u_h,v_h) + s(u_h,v_h) = 0  , \\
# a(u_h,v_h) &= \int_{\mathcal{V}} \partial_t {u}_h {v}_h ~\sqrt{g}
# - \int_{\mathcal{V}} {u}_h \boldsymbol{\beta} \cdot \mathbf{grad}~v_h    ~\sqrt{g} \\
# b({u}_h,{v}_h) &= \int_{{\mathcal{E}}_0} \{\! \!\{ J \boldsymbol{\beta}{u}_h \}\! \!\} \cdot
# \lbrack\!\lbrack {v}_h,  J g^{-1}\boldsymbol{n} \rbrack\!\rbrack  ~\sqrt{g} , \\
# s({u}_h,{v}_h) 	&= \int_{{\mathcal{E}}_0} \frac{1}{2} |\boldsymbol{\beta}\cdot \boldsymbol{n}^+|
# \lbrack\!\lbrack {u}_h, \sigma^* [\boldsymbol{n}]\rbrack\!\rbrack \cdot \lbrack\!\lbrack {v}_h, \sigma^*[\boldsymbol{n}] \rbrack\!\rbrack \sqrt{g}   ,
# \end{align*}
# ```
# $\forall v_h \in \mathbb{V}$ where $\mathcal{V}$ is the parametric space,
# $\boldsymbol{n}\in \mathbb{R}^n$ is the normal vector in the parametric space,
# $\boldsymbol{\beta}\in \mathbb{R}^n$ is a known vector field,
# $J$ is the Jacobian associated to the geometrical map $\sigma: \mathcal{V} \rightarrow \gamma$,
# $\sigma^*$ is the pushforward operator,
# and $\sqrt{g} = (\det{g})^{1/2}$ is the measure.


# ## Set up
# First load all required packages. In this example, we use a serial model, and the
# basic LU and Newton solvers provided in Gridap and GridapSolvers.
using GridapGeosciences
using Gridap
using GridapSolvers

# ## Discrete model

# To obtain a refined parametric model, we first define the coarse parametric model, and
# then apply $\ell$ levels of refinement:
ℓ = 3
radius = 1.0
coarse_mesh = CubedSphereMesh(radius)
model = AtlasDiscreteModel(coarse_mesh,ℓ,manifold_style=IntrinsicManifold())


# ## Triangulation
# This test requires both volume and skeleton triangulations.
# For more information about skeleton triangulations in Gridap, refer to
# [Tutorial 6](https://gridap.github.io/Tutorials/dev/pages/t006_dg_discretization/).
#
# The volume triangulation and associated panel ids can be extracted as per usual:
Ω = Triangulation(model)

# The skeleton triangulation, skeleton normal vector and skeleton panel ids are:
Λ = SkeletonTriangulation(model)
n_Λ = get_normal_vector(Λ)

# Note, $n_{\Lambda}$ is the skeleton normal vector in the parametric space.
# To obtain the normal vector in the ambient space, we can pushforward $n_{\Lambda}$
# and plot the result on $\Lambda$:
n_ambient = pushforward_normal(Λ)
cellfields = ["amb_n_plus"=>n_ambient.plus, "amb_n_minus"=>n_ambient.minus, "amb_n_total"=>n_ambient.minus+n_ambient.plus ]
writevtk_with_cell_geomap(AmbientMapCellField(Λ),Λ,"ambient_skeleton_normal",cellfields=cellfields,append=false)


# ## FE Spaces
# Now that we have a discrete model, we define trial and test spaces using Gridap's high level API:
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(model, reffe; conformity=:L2)
P = TrialFESpace(Q)

# ## Initial condition
# The initial condition and velocity field is
# ```math
# \begin{align*}
# \widetilde{u} &= \exp( -(y^2 + z^2) ) \\
# \widetilde{\boldsymbol{\beta}} &= (-y,x,0)
# \end{align*}
# ```
# This is defined as a function of the forward map as follows:
function uₓ(x)
    exp(-(x[2]^2 + x[3]^2))
end

function βₓ(x)
  VectorValue(-x[2],x[1],0)
end

# Then converted into a panelwise cellfield, where we extract the contravariant components
# for the velocity:
ambient_map_cf = AmbientMapCellField(Ω)
u = uₓ∘ambient_map_cf
β = (pinvJ∘transpose∘∇(ambient_map_cf))⋅(βₓ∘ambient_map_cf)
# ## Weak form
# The weak form is written as a transient problem using  using Gridap's high level API.
# For more information about transient problems in Gridap, refer to
# [Tutorial 17](https://gridap.github.io/Tutorials/dev/pages/t017_transient_linear/#transient_linear.jl-1).
# We use an increased degree of quadrature to exactly approximate the geometrical map included in the weak form.
function my_mean( Bu_n::Gridap.Geometry.SkeletonPair)
  plus  = ( Bu_n.plus)
  minus = ( Bu_n.minus)
  0.5*( plus - minus  )
end

meas = MeasureCellField(Ω)
meas_skel = MeasureCellField(Λ)
upwind = 0.5*abs((β⋅n_Λ).plus)
dΩ = Measure(Ω,4*order)
dΛ = Measure(Λ,4*order)

a_Ω(u,v) = ∫(-(u*(∇(v)⋅β))*meas)dΩ
b_Ω(u,v) = ∫(my_mean((β*u)⋅n_Λ)*jump(v)*meas_skel.plus)dΛ
s_Ω(u,v) = ∫(upwind*jump(u)*jump(v)*meas_skel.plus)dΛ

mass(t,dtu,v) = ∫((dtu*v)*meas)dΩ
res(t,u,v) =  a_Ω(u,v) + b_Ω(u,v) + s_Ω(u,v)
jac(t,u,du,v) = a_Ω(du,v) + b_Ω(du,v) + s_Ω(du,v)
jac_t(t,u,dtu,v) = ∫( (dtu*v)*meas )dΩ

# ## Transient problem
# The transient operator is defined as follows:
t₀ = 0.0
tF = 2*π
nsteps = 100
dt = tF/nsteps
uh₀ = interpolate_everywhere(u, P)
opT = TransientSemilinearFEOperator(mass, res, (jac,jac_t), P, Q)
# The transient operator is solved using a Runge Kutta method
tableau = :SDIRK_Crouzeix_3_4
ls = LUSolver()
nls = GridapSolvers.NonlinearSolvers.NewtonSolver(ls;rtol=1.e-12)

solver = RungeKutta(nls, ls, dt, tableau)
solT = solve(solver, opT, t₀, tF, uh₀)

# ## Post processing
# The transient solution is post-processed and inspected in Paraview:
mkpath("transient_sol/results")
createpvd("transient_sol/results") do pvd
  pvd[0] = createvtk_with_cell_geomap(AmbientMapCellField(Ω), Ω, "transient_sol/results/results_0" * ".vtu",
           cellfields=["u"=>uh₀],append=false)
  for (t, uh) in solT
    println("t = $t")
    pvd[t] = createvtk_with_cell_geomap(AmbientMapCellField(Ω),Ω, "transient_sol/results/results_$t" * ".vtu",
           cellfields=["u"=>uh],append=false)
  end
end
