
"""
Solve the Darcy problem on cubed sphere manifold, using augmented lagrangian
method
̃u + ∇ᵧ ̃p  = ̃f₁
    ∇ᵧ⋅ ̃u = ̃f₂
"""

module DarcyGMG

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Geometry
using GridapGeosciences
using GridapP4est
using Test

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
    metric_cf_lev = MetricCellField(Ω)
    meas_cf_lev = MeasureCellField(Ω)
    ap = (u,v) ->  biform(u,v,dΩ,metric_cf_lev,meas_cf_lev)
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

function get_jacobi_smoothers(mh)
  nlevs = num_levels(mh)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),nlevs-1)
  level_parts = view(get_level_parts(mh),1:nlevs-1)
  return HierarchicalArray(smoothers,level_parts)
end


function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  metric_cf_lev = MetricCellField(Ω)
  meas_cf_lev = MeasureCellField(Ω)
  return (u,v) -> biform(u,v,dΩ,metric_cf_lev,meas_cf_lev)
end

uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)
pX(x) = x[3]


function darcy_gmg_manifold(atlas_model,
  p_fe::Int,dir::String,uX::Function,pX::Function,γ=1.0,smoother_type=:patch,return_vtk=false;_i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc; smoother = $(smoother_type)")

  # fmodel = refine(atlas_model)
  # mh = ModelHierarchy([fmodel,atlas_model])
  mh = ModelHierarchy(atlas_model,1)

  model = get_model(mh,1)
  Ω_atlas = Triangulation(model)
  qdegree = 6*(p_fe+1)
  dΩ = Measure(Ω_atlas,qdegree)
  dΩ_error = Measure(Ω_atlas,2*qdegree)

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

  ## Manufactured solution
  p_cf = pX∘ambient_map_cf
  u_cf = meas_cf*((pinvJ∘covariant_basis_cf)⋅(uX∘ambient_map_cf))

  biform_u(u,v,dΩ,metric,meas) = ( ∫( (u⋅ (metric⋅v))*(1.0/meas) )dΩ
                                 + ∫(γ*(divergence(u)*divergence(v))*(1.0/meas) )dΩ )

  biform((u,p),(v,q),dΩ) = ( biform_u(u,v,dΩ,metric_cf,meas_cf)
                            - ∫(divergence(v)*p)dΩ
                            + ∫(divergence(u)*q)dΩ
                            )

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

  smoothers = if smoother_type == :block
     get_block_jacobi_smoothers(tests_u)
  elseif smoother_type == :patch
     get_patch_smoothers(tests_u,biform_u,qdegree)
  else
    error("Unknown smoother type")
  end
  # smoothers =  get_jacobi_smoothers(mh) #Jacobi will always fail

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
    maxiter=10,mode=:preconditioner,verbose=_i_am_main,
    atol=1.0e-14, rtol=1.0e-08
  )

  ##### solvers for the blocks of the preconditioner
  solver_u = gmg
  solver_p = CGSolver(JacobiLinearSolver();maxiter=1000,atol=1e-14,rtol=1.e-8,verbose=1,name="CG_pressure")


  #### preconditioner
  bblocks  = [LinearSystemBlock() LinearSystemBlock();
              LinearSystemBlock() BiformBlock((p,q) -> ∫( (-1.0/γ)*(p*q)*(meas_cf))dΩ,Q,Q)]
  coeffs = [1.0 1.0;
            0.0 1.0]

  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)

  ##### Preconditioned external solver
  ls = FGMRESSolver(20,P;maxiter=1000,atol=1e-14,rtol=1.e-10,verbose=true)
  ns = numerical_setup(symbolic_setup(ls,A),A)

  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)

  xh = FEFunction(X,x)
  uh,ph = xh

  eu = uh-u_cf
  eu_l2 = sqrt(sum(∫( (eu⋅(metric_cf ⋅ eu))*(1.0/meas_cf) )dΩ_error )) #

  ep = ph-p_cf
  ep_l2 = sqrt(sum(∫( (ep⋅ep)*meas_cf )dΩ_error ))

  lvl = nref(model)
  uh_proj = covariant_basis_cf ⋅ (1.0/meas_cf*uh)
  u_proj = covariant_basis_cf ⋅ (1.0/meas_cf*u_cf)
  if return_vtk
    writevtk_with_cell_geomap(LatLonMapCellField(Ω_atlas),Ω_atlas,dir*"/Darcy_manifold_nref$(lvl)_p$(p_fe)",
          cellfields= ["uh"=>uh_proj, "ph"=>ph, "eu"=>uh_proj-u_proj, "ep"=>ep, "u_ex"=>u_proj,"p_ex"=>p_cf],
          append=false)
  end

  return eu_l2, ep_l2, false

end


################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[1],smoother_type=:patch,_i_am_main=true)
  dir = @__DIR__
  γ = 1
  p_convergence_auto_test(ps,models,darcy_gmg_manifold,dir,uX,pX,γ,smoother_type;_i_am_main=_i_am_main)
end



end # module
