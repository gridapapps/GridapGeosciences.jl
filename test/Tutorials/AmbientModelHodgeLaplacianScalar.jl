# # Scalar Hodge Laplacian on the cubed sphere manifold
#
# This example solves the scalar Hodge Laplacian (or mixed scalar Poisson).
# Consider the scalar Poisson (or Laplace Beltrami) $-\Delta_{\gamma} \widetilde{\varphi} = \widetilde{f}$
# The mixed form is
#
# ```math
# \begin{align*}
# \widetilde{\boldsymbol{u}} + \nabla_{\gamma}\varphi &= 0 \quad \text{in} \quad \gamma, \\
# \nabla_{\gamma}\cdot \widetilde{\boldsymbol{u}}  &= \widetilde{f} \quad \text{in} \quad \gamma,
# \end{align*}
# ```
#
# where $\gamma$ is the cubed sphere manifold, $\widetilde{\varphi}, \widetilde{f}: \gamma \rightarrow \mathbb{R}$
# are scalar valued functions defined in the ambient space of the manifold,
# $\widetilde{\boldsymbol{u}} = - \nabla_{\gamma} \widetilde{\varphi} \in T_p \gamma $ is a vector valued functions
# in the  tangent space of the manifold,
# $\nabla_{\gamma}$ is the surface gradient operator, and
# $\nabla_{\gamma}\cdot$ is the surface divergence operator.
#
# We use $H(\mathrm{div})$ vector and $L^2$ scalar finite-elements, to solve in the ambient space of the cubed sphere.
# The weak formulation in the ambient space is: find $\widetilde{\boldsymbol{u}} \in H(\mathrm{div},\gamma)$,
# $\widetilde{\varphi} \in L^2(\gamma)$ such that
# ```math
# \begin{align*}
# \int_{\gamma} \boldsymbol{u}\cdot \boldsymbol{v}
# - \int_{\gamma} \varphi \nabla_{\gamma} \cdot \boldsymbol{v}
#    &= 0
# \qquad   \forall \boldsymbol{v} \in H(\mathrm{div},\gamma) \\
# \int_{\gamma} \psi \nabla_{\gamma} \cdot \boldsymbol{u}
#    &= \int_{\gamma} f \psi
# \qquad   \forall \psi \in L^2(\gamma)

# \end{align*}
# ```
# where $f = - \Delta_{\gamma} \widetilde{\varphi}$ is the surface Laplacian that
# manufactures the solution.


# ## Set up
# First load all required pacakges. In this example, we will use a serial model.
using GridapGeosciences
using Gridap

# ## Discrete model

# To obtain a refined ambient model, we pass $\ell$ levels of refinement:
ℓ = 3
radius = 1.0
coarse_mesh = CubedSphereMesh(radius)
model = AtlasDiscreteModel(coarse_mesh,ℓ,manifold_style=ExtrinsicManifold())

# We can visualise the triangulation using the typical visualise tools in Gridap:
Ω = Triangulation(model)
writevtk(Ω,"sphere_model",append=false)

# ## FE Spaces
# Now that we have a discrete model, we define trial and test spaces using Gridap's high level API.
order = 1
Q = TestFESpace(Ω, ReferenceFE(lagrangian,Float64,order); conformity=:L2, constraint=:zeromean)
P = TrialFESpace(Q)
V = TestFESpace(Ω, ReferenceFE(raviart_thomas,Float64,order); conformity=:HDiv)
U = TrialFESpace(V)

# The assoicated multifields are:
Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])


# ## Manufactured solution
# We consider the method of manufactured solutions for analytic solution, $\widetilde{\varphi} = xyz$.
# This is defined as a regular Julia function that takes a point in ambient space
# and returns a scalar:
φₓ(x) = x[1]*x[2]*x[3]

# The cooresponding rhs forcing function is defined panelwise, using the AmbientCellField.
# Similar to ParametricCellField, AmbientCellField returns an GenericCellField object, where the cell_field is an
# array of cell-wise functions. However, the acutal input function of AmbientCellField is defined
# differently  to CellField, where the user passes a function that takes points
# in physical space and returns the function evaluated in physical space.
# To manufacture the solution, we use the ambient_surflap, which computes the surface
# Laplacian operator for ambient functions
u_cf = ∇s(φₓ,Ω)
slap_cf = Δs(φₓ,Ω)
rhs_cf = -slap_cf


# ## Weak form
# To define the weak form, we use Gridap's high level API, as follows:
dΩ = Measure(Ω,6*order)
biform_u((u,p),(v,q)) = ∫( u⋅v )dΩ - ∫( p*(∇⋅v) )dΩ
biform_p((u,p),(v,q)) = ∫( q*(∇⋅u) )dΩ

biform((u,p),(v,q)) = biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))
liform((v,q)) = ∫( (rhs_cf*q) )dΩ


# ## FE problem
# Now we can build the FE operator and solve using the LU solver:
op = AffineFEOperator(biform,liform,X,Y)
uh,ph = solve(LUSolver(),op)

# For the pressure and velocity, the $L^2$ norm of the error between the exact
# and numerical soltuions is computed as (recall $\widetilde{\boldsymbol{u}} = - \nabla_\gamma \widetilde{\varphi}$)
ep = φₓ - ph
el2_p = sqrt(sum(∫( ep*ep  )dΩ))

eu = uh - (- u_cf )
el2_u = sqrt(sum(∫( eu⋅eu  )dΩ))

# ## Post processing
# The solution can be visualised in the ambient space using Gridap's visualisation
# functionality:
writevtk(Ω,"hodge_laplacian_scalar",
        cellfields=["p"=>φₓ,"ph"=>ph,"ep"=>ep,
                "uamb"=>u_cf,"uamb_h"=>uh, "eu"=>eu],
        append=false)
