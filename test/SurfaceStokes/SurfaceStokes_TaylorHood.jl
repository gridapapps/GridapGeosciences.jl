""" Solve the vector surface Stokes + Laplacian in 2D
  -ν Δᵧ(̃u) + α ̃u + ∇ᵧ ̃p = ̃f
                  ∇ᵧ⋅̃u  = 0
where Δᵧ(̃u) = ∇ᵧ(∇ᵧ⋅̃u) - ∇ᵧ^⟂(∇ᵧ^⟂ ⋅ ̃u)
The velocity is in H1 vector-valued lagrangian, order_u ≥ 2
The pressure is in H1 continuous lagrangian, order_p = order_u - 1
Using the Taylor--Hood pair
"""

module SurfaceStokes_TaylorHood

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Geometry
using GridapGeosciences
import GridapGeosciences.Geometry: get_cell_ambient_maps
import GridapGeosciences.CellData: deriv_det, deriv_sqrt, cpAB
using GridapP4est
using Test

import Gridap.Fields: grad2curl

## Need to use velocity field that is tangent, but not divergence free. Thus, the
## incompressibility condition div u = 0 does not hold, and the Stokes system
## must be solved.
## The pressure field is zeromean
uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)
pX(x) = x[3]


function surface_stokes(atlas_model,
  p_fe::Int,dir::String,uX::Function,pX::Function,ls=LUSolver(),return_vtk=false,jumpCalcs=false;
  _i_am_main=true)
  α = 1.0
  ν = 1.0

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)
 _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  @check p_fe >= 2 "/n # Taylor hood pair requires p≥2 "

  degree = 6*(p_fe+1)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,3*degree)

  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))

  ## Expanded product rule to help with weak form
  # div ( √g u) = div(u)*√g + u⋅gradient(√g)
  divg_product_rule(u) = divergence(u)*meas_cf + u⋅grad_meas_cf

  ## Expand product rule of divergence( (R⋅g)⋅u)
  # See https://github.com/gridapapps/GridapGeosciences.jl/pull/60#issuecomment-5216501843
  grad_metric_cf = gradient(metric_cf)
  # [i,j] = Σ_k v^k ∂g_jk/∂x_i   (product-rule term from differentiating g⋅v)
  _cpvB(v,B) = Gridap.TensorValues.contracted_product(Val(1), v, permutedims(B,(3,1,2)))
  function divergenceRgu(v)
    dgv = Operation(_cpvB)(v,grad_metric_cf) + ∇(v)⋅metric_cf   # = ∇(g⋅v)
    -1.0*Operation(grad2curl)(dgv) # div(perp(w)) = -∂w₂/∂x₁ + ∂w₁/∂x₂ = -(∂w₂/∂x₁ - ∂w₁/∂x₂) = -grad2curl(w)
  end

  ## Manufactured solution
  p_cf = pX∘ambient_map_cf
  u_contra_cf = (pinvJ∘covariant_basis_cf)⋅(uX∘ambient_map_cf)

  @check sum(∫(p_cf*meas_cf)dΩ) < 1e-14 "Function must be zero mean to solve with zeromean FE space!" # check zeromean

  sigma_cf = divs(uX,Ω_atlas) # div u to add to pressure equation
  rhs_curl_cf = vecΔs_2D(uX, Ω_atlas)
  rhs = -1.0*ν*rhs_curl_cf + α*u_contra_cf + ∇s(pX,Ω_atlas) # returns contravariant components
  # rhs = -1.0*ν*rhs_curl_cf + α*u_int + inv_metric_cf⋅(gradient(p_int))

  ## FE spaces: Taylor hood pair --> Q2/Q1 continuous
  reffe_u  = ReferenceFE(lagrangian,VectorValue{2, Float64},p_fe)
  reffe_p = ReferenceFE(lagrangian,Float64,p_fe-1)

  V = TestFESpace(Ω_atlas, reffe_u; conformity=:H1)
  U = TrialFESpace(V)

  Q = TestFESpace(Ω_atlas, reffe_p; conformity=:H1,constraint=:zeromean)
  P = TrialFESpace(Q)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # Interpolate the exact solution into the FE space
  p_int = interpolate(p_cf,P)
  u_int = interpolate(u_contra_cf,U)

  # ## Weak form
  biform_curl((u,p),(v,q)) = ( ∫( ν*(divg_product_rule(u)*divg_product_rule(v))*(1/meas_cf) )dΩ
                            +  ∫( ν*(divergenceRgu(u)*divergenceRgu(v))*(1/meas_cf)  )dΩ
                            )

  biform_u((u,p),(v,q)) = ∫( α*((u⋅(metric_cf⋅v))*meas_cf)  )dΩ - ∫( p*divg_product_rule(v)  )dΩ

  biform_p((u,p),(v,q)) = ∫( q*divg_product_rule(u)  )dΩ

  biform((u,p),(v,q)) = biform_curl((u,p),(v,q)) + biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))


  liform((v,q)) = ∫( (rhs⋅(metric_cf⋅v))*meas_cf  )dΩ + ∫( (sigma_cf*q)*meas_cf )dΩ

  # ## FE problem
  op = AffineFEOperator(biform,liform,X,Y)
  A = get_matrix(op)
  b = get_vector(op)
  ns = numerical_setup(symbolic_setup(ls,A),A)
  x = Gridap.Algebra.allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  xh = FEFunction(X,x)
  uh,ph = xh

  # For the depth, the $L^2$ norm of the error between the exact and numerical solutions is computed as
  ep = p_int - ph
  ep_l2 = sqrt(sum(∫((ep⋅ep)*meas_cf)dΩ_error))

  # For the velocity, the $L^2$ norm of the error between
  # the exact and numerical solutions is computed as
  eu = u_int - uh
  eu_l2 = sqrt(sum(∫( eu⋅(metric_cf⋅eu)*(meas_cf) )dΩ_error))
  eu_h1 = sqrt(sum(∫( eu⋅(metric_cf⋅eu)*(meas_cf) + (gradient(eu)⊙(inv_metric_cf⋅gradient(eu)))*meas_cf  )dΩ_error))

  ## To help with plotting, split the terms of the H1 norm
  mass_term =  eu_l2 #sqrt(sum(∫( eu⋅(metric_cf⋅eu)*(meas_cf)   )dΩ_error))
  grad_term = sqrt(sum(∫( (gradient(eu)⊙(inv_metric_cf⋅gradient(eu)))*meas_cf  )dΩ_error))

  ######### START JUMP CALCS -- for publication
  if jumpCalcs ## only if serial model
    _i_am_main && println("computing jump calcs")
    u_ambient = covariant_basis_cf⋅uh
    eu = u_contra_cf - uh
    eu_ambient = covariant_basis_cf⋅eu

    ## 1. restrict the ambient solution to plus and minus side
    u_tilde_plus = (eu_ambient).plus
    u_tilde_minus = (eu_ambient).minus

    ## 2. Make a skeleton triangulation of the interface of charts

    # 2a. get a mask that is the interface of charts
    topo = get_grid_topology(atlas_model)
    Dc = num_cell_dims(topo)
    e2c = Gridap.Geometry.get_faces(topo,1,Dc)
    panel_ids = get_cell_ambient_maps(atlas_model.model).ptrs

    mask = zeros(num_facets(atlas_model))
    for (i,edge) in enumerate(e2c)
      pid_1 = panel_ids[edge[1]]
      pid_2 = panel_ids[edge[2]]
      if pid_1 != pid_2
        mask[i] = 1
      end
    end

    # 2b. Skeleton triangulation:  for simplicity, extract the trian out of the adapted trian
    skel = SkeletonTriangulation(atlas_model,Bool.(mask))
    Λ = skel.trian

    ## 3. Compute the pushforward of the skeleton normal vector. Then restrict to plus and minus sides
    n_tilde = pushforward_reference_normal(Λ)
    n_tilde_plus = n_tilde.plus
    n_tilde_minus = n_tilde.minus

    ## 4. Compute the jump term [u ⊗ n] as per https://doi.org/10.1007/s10915-008-9261-1
    ## This is the outer product of two vectors. So returns a A_ij = v_i*w_j
    jump = u_tilde_plus ⊗ n_tilde_plus + u_tilde_minus ⊗ n_tilde_minus

    ## 5. Evaluate jump norm at quadrature points. Use ⊙ to take the double contraction
    dΛ = Measure(Λ,3*degree)
    area_skel = pullback_area_form(Λ)
    γ = 1.0
    dxx = dx(atlas_model)
    jump_norm_scaled = 1.0/sqrt(dxx) * sqrt(sum(  1/dxx*∫( (jump⊙jump)*(meas_cf.plus * area_skel.plus) )dΛ))

    ## 6. Save the solution using DrWatson (needs to be installed locally)
    # n = num_cells(atlas_model)/6
    # n_ref = lvl
    # output = @strdict eu_l2 eu_h1 ep_l2 n p_fe n_ref Dc dxx mass_term grad_term jump_norm_scaled
    # safesave(datadir(dir*"/convergence", ("stokes_nref$(n_ref)_p$(p_fe)_D$Dc.jld2")), output)
    ######### END JUMP CALCS -- for publication

  end

 _i_am_main && println("eu = $(eu_l2), ep = $(ep_l2), euh1 = $eu_h1")


  if return_vtk
      panel_cfs = [p_int,ph,ep, covariant_basis_cf⋅ u_int, covariant_basis_cf⋅ uh,eu]
      labels = ["p","ph", "ep", "u","uh","eu"]
      cellfields = map((x,y) -> x=>y, labels,panel_cfs)
      writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/stokes_nref$(lvl)_p$p_fe",
          cellfields=cellfields,append=false)
  end

  return eu_h1, ep_l2, false

end

################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[2],_i_am_main=true,jumpCalcs=false)
  # Taylor hood pair requires p≥2
  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,surface_stokes,dir,uX,pX,ls,jumpCalcs;_i_am_main=_i_am_main)
end


end
