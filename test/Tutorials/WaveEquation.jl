# # Linear wave equation on the cubed sphere manifold
#
# This example solves the linear wave equation, given by
#
# ```math
# \begin{align*}
# \widetilde{\boldsymbol{u}} + \nabla_{\gamma} \varphi &= \widetilde{\boldsymbol{f}}_1 \quad \text{in} \quad \gamma, \\
# \widetilde{\varphi} + \nabla_{\gamma}\cdot \widetilde{\boldsymbol{u}}  &= \widetilde{f}_2 \quad \text{in} \quad \gamma,
# \end{align*}
# ```
#
# where $\gamma$ is the cubed sphere manifold, $\widetilde{\varphi}, \widetilde{f}_2: \gamma \rightarrow \mathbb{R}$
# are scalar valued functions defined in the ambient space of the manifold,
# $\widetilde{\boldsymbol{u}}, \widetilde{\boldsymbol{f}}_1 \in T_p \gamma $ are vector valued functions defined in the
# tangent space of the manifold,
# $\nabla_{\gamma}$ is the surface gradient operator, and
# $\nabla_{\gamma}\cdot$ is the surface divergence operator.
#
# We use $H(\mathrm{div})$ vector and $L^2$ scalar finite-elements, to solve in the parametric space of the cubed sphere.
# The weak formulation in the parametric space is: find $\boldsymbol{u} \in H(\mathrm{div},\mathcal{V})$, $\varphi \in L^2(\mathcal{V})$ such that
# ```math
# \begin{align*}
# \int_{\mathcal{V}} \boldsymbol{u}\cdot(g \boldsymbol{v}) \frac{1}{\sqrt{g}}
# - \int_{\mathcal{V}} \varphi \nabla\cdot \boldsymbol{v}
#    &= \int_{\mathcal{V}} \boldsymbol{f}_1\cdot(g \boldsymbol{v}) \frac{1}{\sqrt{g}}
#       - \int_{\mathcal{\partial\mathcal{V}}} \varphi_{bc} \boldsymbol{v}\cdot\boldsymbol{k}~\mathrm{d}S
# \qquad   \forall \boldsymbol{v} \in H(\mathrm{div},\mathcal{V}) \\
# \int_{\mathcal{V}} \varphi \psi \sqrt{g}
# + \int_{\mathcal{V}} \psi \nabla\cdot \boldsymbol{u}
#    &= \int_{\mathcal{V}} f_2 \psi \sqrt{g}
# \qquad   \forall \psi \in L^2(\mathcal{V})

# \end{align*}
# ```
# where $\mathcal{V}$ is the parametric space,  $f_1\in \mathbb{R}^n$, $f_2: \mathcal{V} \rightarrow \mathbb{R}$,
# $\partial \mathcal{V}$ is the boundary, $\boldsymbol{k}$ is the normal to the boundary,
# and $\varphi_{bc}$ is the boundary condition,
# $g$ is the Riemannian metric associated to the geometrical map $\sigma: \mathcal{V} \rightarrow \gamma$,
# and $\sqrt{g} = (\det{g})^{1/2}$ is the measure.


# ## Set up
# First load all required packages. In this example, we will use a distributed model. So we also initialise MPI.
using GridapGeosciences
using Gridap
using PartitionedArrays
using MPI

MPI.Init()
ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))

# ## Discrete model

# To obtain a refined 3D parametric model, we pass $\ell$ levels of refinement:
ℓ = 2
radius,thickness = 1.0, 0.19
coarse_mesh = CubedSphereWithThicknessMesh(radius,thickness)
octree3_model = AtlasOctreeDistributedDiscreteModel(ranks, coarse_mesh, ℓ; manifold_style=IntrinsicManifold())
model = get_atlas_model(octree3_model)

# We can visualise the triangulation in the ambient space of the 3D cubed sphere
# by passing a cellwise array of geometrical maps to writevtk_with_cell_geomap:
Ω = Triangulation(model)
writevtk_with_cell_geomap(AmbientMapCellField(Ω),Ω,"sphere_model",append=false)

# The 3D cubed sphere model has tags associated to the bottom, top and intermediate
# boundary cells. We can visualise each component of the model by passing the appropriate tag
# to the BoundaryTriangulation constructor
Γ_top = BoundaryTriangulation(model,tags=["top_boundary"])
Γ_bottom = BoundaryTriangulation(model,tags=["bottom_boundary"])
Γ_intermediate = BoundaryTriangulation(model,tags=["intermediate_boundary"])

writevtk_with_cell_geomap(AmbientMapCellField(Γ_bottom),Γ_bottom,"boundary_bottom",append=false)
writevtk_with_cell_geomap(AmbientMapCellField(Γ_top),Γ_top,"boundary_top",append=false)
writevtk_with_cell_geomap(AmbientMapCellField(Γ_intermediate),Γ_intermediate,"boundary_intermediate",append=false)

# In this test, we have non-homogeneous boundary conditions. So we need to create
# a boundary trangulation, and include the appropriate boundary term that arises
# from integration by parts.
Γ = BoundaryTriangulation(model;tags=["bottom_boundary","top_boundary"])
nΓ = get_normal_vector(Γ)


# ## FE Spaces
# Now that we have a discrete model, we define trial and test spaces using Gridap's high level API.
order = 1
Q = TestFESpace(Ω, ReferenceFE(lagrangian,Float64,order); conformity=:L2)
P = TrialFESpace(Q)
V = TestFESpace(Ω, ReferenceFE(raviart_thomas,Float64,order); conformity=:HDiv)
U = TrialFESpace(V)

# The associated multifields are:
Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

# ## Manufactured solution
# We consider the method of manufactured solutions for analytic solutions defined in ambient space:
# ```math
# \begin{align*}
# \widetilde{\boldsymbol{u}} &= (-y,x,0)\\
# \widetilde{\varphi} &= \exp( -(y^2 + z^2) )
# \end{align*}
# ```
# These analytic solutions are defined as a function of the forward map, as follows:

function u(x)
  VectorValue(-x[2],x[1],0.0)
end

function φ(x)
  exp(-(x[2]^2+x[3]^2))
end

ambient_map_cf = AmbientMapCellField(Ω)
covariant_basis_cf = transpose∘∇(ambient_map_cf)
meas = MeasureCellField(Ω)
g = MetricCellField(Ω)

# Each cell is assigned a panel identifier, $p$, which is extracted as a cellwise array.
# This is used to generate a panelwise cellfield of the analytic functions, where
# we extra the contravariant Piola component for the velocity:
u_cf = meas*(pinvJ∘covariant_basis_cf)⋅u
phi_cf = φ∘ambient_map_cf

# Interpolate the exact solution into the FE space
p_int = interpolate(phi_cf,P)
u_int = interpolate(u_cf,U)

# ## Weak form
# The weak form is written as a multifield problem using Gridap's high level API.
# We use an increased degree of quadrature to exactly approximate the geometrical map included in the weak form.
degree = 4*(order+1)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

biform_u((u,p),(v,q)) = ∫( (u⋅ (g⋅v))*(1/meas) )dΩ - ∫( p*(∇⋅v) )dΩ
biform_p((u,p),(v,q)) = ∫( (p*q)*meas )dΩ + ∫( q*(∇⋅u) )dΩ
biform((u,p),(v,q)) = biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))

# The corresponding rhs forcing function is manufactured as follows, where
# we assume regularity to IBP:
liform((v,q)) = ( ∫( (u_int⋅ (g⋅v))*(1/meas) )dΩ + ∫( gradient(p_int)⋅v )dΩ
                + ∫( (p_int*q)*meas )dΩ + ∫( q*(∇⋅u_int) )dΩ
                - ∫( (v⋅nΓ)*p_int )dΓ
                )


# ## FE problem
# Now we can build the FE operator that represents the linearised wave equation.
op = AffineFEOperator(biform,liform,X,Y)

# We solve using a LU solver. One can also solve using MUMPS or another iterative solver from GridapSolvers/GridapPETSc
ls = LUSolver()
A = get_matrix(op)
b = get_vector(op)
ns = numerical_setup(symbolic_setup(ls,A),A)
x = Gridap.Algebra.allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)
xh = FEFunction(X,x)
uh,ph = xh

# For the depth, the $L^2$ norm of the error between the exact and numerical solutions is computed as
ep = phi_cf - ph
ep_l2 = sqrt(sum(∫((ep⋅ep)*meas)dΩ))

# For the velocity, the parametric velocity field is pushed to the ambient space
# via the contraviant Piola map. Then the $L^2$ norm of the error between
# the exact and numerical solutions is computed as
uh_proj = covariant_basis_cf ⋅ (1.0/meas * uh)
u_proj_cf = covariant_basis_cf ⋅ (1.0/meas *u_cf )
eu = u_cf - uh
eu_l2 = sqrt(sum(∫( eu⋅(g⋅eu)*(1.0/meas) )dΩ))


# ## Post processing
# The solution can be visualised in the ambient space by passing a
# cell-wise array of geometrical maps to our writevtk_with_cell_geomap function
writevtk_with_cell_geomap(AmbientMapCellField(Ω),Ω,"wave_equation",
       cellfields=["p"=>phi_cf,"ph"=>ph,"ep"=>ep,
               "uamb"=>u_proj_cf,"uamb_h"=>uh_proj, "eu"=>eu],
       append=false)
