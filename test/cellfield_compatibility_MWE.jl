
"""
Solve the Darcy problem on periodic meshes
̃u + ∇ᵧ ̃p  = ̃f₁
    ∇ᵧ⋅ ̃u = ̃f₂
"""

using GridapGeosciences
using Gridap

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver


function get_patch_smoothers(sh,biform,qdegree)
  nlevs = num_levels(sh)
  smoothers = map(view(sh,1:nlevs-1)) do shl
    model = get_model(shl)
    ptopo = Gridap.Geometry.PatchTopology(ReferenceFE{0},model)
    space = get_fe_space(shl)
    Ω  = Gridap.Geometry.PatchTriangulation(model,ptopo)
    dΩ = Measure(Ω,qdegree)
    ap = (u,v) -> biform(u,v,dΩ)
    solver = PatchBasedSmoothers.PatchSolver(
      ptopo, space, space, ap;
      assembly = :star,
      collect_factorizations = true,
      is_nonlinear = false
    )
    return RichardsonSmoother(solver,10,0.2)
  end
  return smoothers
end


function get_block_jacobi_smoothers(sh)
  nlevs = num_levels(sh)
  smoothers = map(view(sh,1:nlevs-1)) do shl
    model = get_model(shl)
    ptopo = Gridap.Geometry.PatchTopology(ReferenceFE{0},model)
    space = get_fe_space(shl)
    solver = PatchBasedSmoothers.BlockJacobiSolver(space, ptopo; assembly=:star)
    return RichardsonSmoother(solver,10,0.2)
  end
  return smoothers
end


function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end


uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)
pX(x) = x[3]

coarse_mesh = CubedSphereMesh(1.0)
n_ref = 0
atlas_model = AtlasDiscreteModel(coarse_mesh,n_ref,manifold_style=IntrinsicManifold())
p_fe = 1
γ = 1
_i_am_main = true


fmodel = refine(atlas_model)
mh = ModelHierarchy([fmodel,atlas_model])
num_levels(mh)

model = get_model(mh,1)
num_cells(model)
Ω_atlas = Triangulation(model)
qdegree = 6*(p_fe+1)
dΩ = Measure(Ω_atlas,qdegree)

tests_u = TestFESpace(mh,ReferenceFE(raviart_thomas,Float64,p_fe);conformity=:Hdiv);
trials_u = TrialFESpace(tests_u);
U = get_fe_space(trials_u,1)
V = get_fe_space(tests_u,1)
Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,p_fe);conformity=:L2)

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

ambient_map_cf = AmbientMapCellField(Ω_atlas)
metric_cf = MetricCellField(Ω_atlas)
meas_cf = MeasureCellField(Ω_atlas)
covariant_basis_cf = transpose∘∇(ambient_map_cf)

biform_u(u,v,dΩ) = ( ∫( (u⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ
                   + ∫(γ*(divergence(u)*divergence(v))*(1.0/meas_cf) )dΩ
                    )
# biform_u(u,v,dΩ) = ∫( (u⋅ (v)) )dΩ + ∫(γ*(divergence(u)*divergence(v)) )dΩ

biform((u,p),(v,q),dΩ) = ( biform_u(u,v,dΩ)
                          - ∫(divergence(v)*p)dΩ
                          + ∫(divergence(u)*q)dΩ
                          )

## Manufactured solution
p_cf = pX∘ambient_map_cf
u_cf = meas_cf*((pinvJ∘covariant_basis_cf)⋅(uX∘ambient_map_cf))

_f = ∇s(pX,Ω_atlas)  # returns contravariant components
f2 = divs(uX,Ω_atlas)
liform((v,q),dΩ) = ( ∫( (u_cf⋅(metric_cf⋅v ))*(1.0/meas_cf)  )dΩ
                    + ∫( (_f⋅(metric_cf⋅v ))  )dΩ ### ∇p⋅v
                    + ∫( γ*divergence(v)*f2  )dΩ
                    + ∫( (q*f2)*meas_cf )dΩ
                  )

a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)
op = AffineFEOperator(a,l,X,Y)
A, b = get_matrix(op), get_vector(op);

#### solvers
biforms = map(mhl -> get_bilinear_form(mhl,biform_u,qdegree),mh)
# smoothers = get_patch_smoothers(tests_u,biform_u,qdegree)
smoothers = get_block_jacobi_smoothers(tests_u)

prolongations = setup_prolongation_operators(tests_u,qdegree;mode=:residual)
restrictions = setup_restriction_operators(
  tests_u,qdegree;mode=:residual,solver=CGSolver(JacobiLinearSolver())
)

gmg = GMGLinearSolver(
  trials_u,tests_u,biforms,
  prolongations,restrictions,
  pre_smoothers=smoothers,
  post_smoothers=smoothers,
  coarsest_solver=LUSolver(),
  maxiter=20,mode=:preconditioner,verbose=_i_am_main,
  atol=1.0e-14, rtol=1.0e-08
)

##### solvers for the blocks of the preconditioner
solver_u = gmg
solver_p = LUSolver()

#### preconditioner
bblocks  = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫( (-1.0/γ)*(p*q)*(meas_cf))dΩ,Q,Q)]
coeffs = [1.0 1.0;
          0.0 1.0]

P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)

##### Preconditioned external solver
ls = FGMRESSolver(20,P;maxiter=1000,atol=1e-14,rtol=1.e-10,verbose=true)
ns = numerical_setup(symbolic_setup(ls,A),A)

x = Gridap.Arrays.allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)
