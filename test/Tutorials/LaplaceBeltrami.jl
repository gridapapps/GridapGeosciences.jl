# # Laplace Beltrami equation on the cubed sphere manifold (intrinsic formulation)
#
# This example solves the Laplace Beltrami equation, given by
#
# ```math
# \begin{align*}
# -\Delta_{\gamma} \widetilde{u} &= \widetilde{f} \quad \text{in} \quad \gamma,
# \end{align*}
# ```
#
# where $\gamma$ is the cubed sphere manifold, $\widetilde{u}, \widetilde{f}: \gamma \rightarrow \mathbb{R}$
# are scalar valued functions, defined in the ambient space of the manifold, and
# $\Delta_{\gamma}$ is the Laplace Beltrami operator.
#
# We use $H^1$ scalar finite-elements, to solve in the parametric space of the cubed sphere.
# The weak formulation in the parametric space is: find $u \in H^1(\mathcal{V})$ such that
# ```math
# \begin{align*}
# \int_{\mathcal{V}} \mathbf{grad}~u \cdot (g^{-1} \mathbf{grad}~ v )~\sqrt{g} &= \int_{\mathcal{V}} f v ~\sqrt{g} \qquad   \forall v \in H^1(\mathcal{V})
# \end{align*}
# ```
# where $\mathcal{V}$ is the parametric space $f: \mathcal{V} \rightarrow \mathbb{R}$,
# $g$ is the Riemannian metric associated to the geometrical map $\sigma: \mathcal{V} \rightarrow \gamma$,
# and $\sqrt{g} = (\det{g})^{1/2}$ is the measure.


# ## Set up
# First load all required packages. In this example, we will use a serial model, and the
#  basic LU solver provided in Gridap.
using GridapGeosciences
using Gridap

# ## Discrete model

# To obtain a refined parametric model, we first define the coarse parametric model, and
# then apply $\ell$ levels of refinement:
ℓ = 3
radius = 1.0
coarse_mesh = CubedSphereMesh(radius)
model = AtlasDiscreteModel(coarse_mesh,ℓ,manifold_style=IntrinsicManifold())

# Each cell is assigned a panel identifier, $p$, which is extracted as a cellwise array.
# Using the panel ids, we can visualise the triangulation in the ambient space of the sphere
# or in latitude-longitude by passing a cellwise array of geometrical maps to writevtk_with_cell_geomap:
Ω = Triangulation(model)
writevtk_with_cell_geomap(AmbientMapCellField(Ω),Ω,"sphere_model",append=false)
writevtk_with_cell_geomap(LatLonMapCellField(Ω),Ω,"latlon_model",append=false)

# ## FE Spaces
# Now that we have a discrete model, we define trial and test spaces using Gridap's high level API.
# To remove the kernel, we use the zeromean constraint in the definition of the FE space.
order = 1
V = TestFESpace(model, ReferenceFE(lagrangian,Float64,order); conformity=:H1, constraint=:zeromean)
U = TrialFESpace(V)

# ## Manufactured solution
# We consider the method of manufactured solutions for analytic solution, $\widetilde{u} = xyz$.
# This is defined as a function of the forward map, as follows:
# First we define a function that given a forward map, returns a function
# that takes coordinates in parametric space as input, and returns the value of the function.
function uₓ(x)
  x[1]*x[2]*x[3]
end

ambient_map_cf = AmbientMapCellField(Ω)

# The function is composed with the ambient map to obtain a cell field in terms of ambient coordinates:
u_cf = uₓ∘ambient_map_cf

# The corresponding rhs forcing function uses the surface Laplacian operator Δs,
# which takes an ambient function and a triangulation and returns a cell field:
slap_cf = Δs(uₓ,Ω)
rhs = -slap_cf

# ## Weak form
# To define the weak form, we first obtain panelwise cellfields of the inverse metric and meas,
# and then write the bilinear and linear forms using Gridap's high level API.
# We use an increased degree of quadrature to exactly approximate the geometrical map included in the weak form.
invg = InvMetricCellField(Ω)
meas = MeasureCellField(Ω)
dΩ = Measure(Ω,6*order)
poisson_biform(u,v) = ∫((gradient(v)⋅(invg⋅gradient(u)))*meas )dΩ
poisson_liform(v) = ∫((rhs*v)*meas)dΩ

# ## FE problem
# Now we can build the FE operator that represents the Laplace Beltrami equation,
# and solve using LU factorisation
op = AffineFEOperator(poisson_biform,poisson_liform,U,V)
uh = solve(LUSolver(),op)

# The $L^2$ norm of the error between the exact and numerical solutions is computed as
e = u_cf-uh
el2 = sqrt(sum(∫((e⋅e)*meas)dΩ))

# ## Post processing
# The solution can be visualised in the ambient space by passing a
# cell-wise array of geometrical maps to our writevtk_with_cell_geomap function
writevtk_with_cell_geomap(AmbientMapCellField(Ω),Ω,"laplace_beltrami",cellfields=["u"=>u_cf,"uh"=>uh,"eu"=>e],append=false)
