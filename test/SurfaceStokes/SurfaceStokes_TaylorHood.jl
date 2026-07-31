""" Solve the vector surface Stokes + Laplacian in 2D
  -ν Δᵧ(̃u) + α ̃u + ∇ᵧ ̃p = ̃f
                  ∇ᵧ⋅̃u  = 0
where Δᵧ(̃u) = ∇ᵧ(∇ᵧ⋅̃u) - ∇ᵧ^⟂(∇ᵧ^⟂ ⋅ ̃u)
The velocity is in H1 vector-valued lagrangian, order_u ≥ 2
The pressure is in H1 continuous lagrangian, order_p = order_u - 1
Using the Taylor--Hood pair
"""

using GridapGeosciences
using Gridap
using Gridap.Helpers

import GridapGeosciences.CellData: deriv_det, deriv_sqrt, cpAB

include("operator.jl")

## Need to use velocity field that is tangent, but not divergence free. Thus, the
## incompressibility condition div u = 0 does not hold, and the Stokes system
## must be solved.
## The pressure field is zeromean
uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)
pX(x) = x[3]


function surface_stokes(atlas_model,
  p_fe::Int,dir::String,uX::Function,pX::Function,ls=LUSolver(),return_vtk=false;
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
  dΩ_error = Measure(Ω_atlas,2*degree)

  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))

  perp_metric_cf = PerpMetric(Ω_atlas)

  ## Expanded product rule to help with weak form
  # div ( √g u) = div(u)*√g + u⋅gradient(√g)
  divg_product_rule(u) = divergence(u)*meas_cf + u⋅grad_meas_cf

  ## Manufactured solution
  p_cf = pX∘ambient_map_cf
  u_contra_cf = (pinvJ∘covariant_basis_cf)⋅(uX∘ambient_map_cf)

  sum(∫( p_cf  )dΩ) # check zeromean

  sigma_cf = divs(uX,Ω_atlas) # div u to add to pressure equation
  rhs_curl_cf = vecΔs_2D(uX, Ω_atlas)
  rhs = -1.0*ν*rhs_curl_cf + α*u_contra_cf + ∇s_contra(pX,Ω_atlas)
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
                            +  ∫( -1.0*ν*(divergence(perp_metric_cf⋅u)*divergence(perp_metric_cf⋅v))*(1/meas_cf)  )dΩ
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

 _i_am_main && println("eu = $(eu_l2), ep = $(ep_l2)")

  if return_vtk
      panel_cfs = [p_int,ph,ep, covariant_basis_cf⋅ u_int, covariant_basis_cf⋅ uh,eu]
      labels = ["p","ph", "ep", "u","uh","eu"]
      cellfields = map((x,y) -> x=>y, labels,panel_cfs)
      writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/stokes_nref$(lvl)_p$p_fe",
          cellfields=cellfields,append=false)
  end

  return eu_l2, ep_l2, false

end


function main(models::AbstractArray;ps=[2,3,4],_i_am_main=true)
  # Taylor hood pair requires p≥2
  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,surface_stokes,dir,uX,pX,ls;_i_am_main=_i_am_main)
end

# n_ref_lvls = 4
# radius = 1.0
# models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
# main(models)



# using Plots
# eu_p2 = [ 0.03164362085970014,0.003991140793862124,0.0002901060711779985,3.970002349520792e-5 ]
# ep_p2 = [0.16461510525371303,  0.044523953987363946, 0.011474435127885615,  0.0028947604104046128]

# eu_p3 = [0.0025600600613447477,0.0002053004908208434, 1.498851700549086e-5,1.7593630367403369e-6]
# ep_p3 = [ 0.0065320141924923025, 0.0007983345802019229,  6.837372482786403e-5,  6.174521768640946e-6]

# eu_p4 = [0.0004348104561119693,2.5925309783872616e-5,1.062248276532864e-6,3.9506215758074916e-8]
# ep_p4 = [0.0011135355236547846, 6.705782594548784e-5, 2.9727560214463023e-6, 2.142191533797376e-7]



# xplot = [2,4,8,16]
# zz = 2e-1xplot.^(-2)
# ww = 3e-2xplot.^(-3)
# qq = 1e-3xplot.^(-4)

# plot()

# plot!(xplot, eu_p2, color=:blue, lw=2,marker=:circle,label="velocity: p=2")
# plot!(xplot, ep_p2,color=:blue,ls=:dash, lw=2,marker=:square, label="pressure: p=1")
# plot!(xplot,zz,lw=2,color=:blue,label="lvl^2")

# plot!(xplot, eu_p3, color=:orange, lw=2,marker=:circle,label="velocity: p=3")
# plot!(xplot, ep_p3, color=:orange, ls=:dash, lw=2,marker=:square, label="pressure: p=2")
# plot!(xplot,ww,lw=2,color=:orange,label="lvl^3")

# plot!(xplot, eu_p4, color=:green, lw=2,marker=:circle,label="velocity: p=4")
# plot!(xplot, ep_p4, color=:green, ls=:dash, lw=2,marker=:square, label="pressure: p=3")
# plot!(xplot,qq,lw=2,color=:green,label="lvl^4")


# plot!(shape=:auto,
#     xaxis=:log2,yaxis=:log10,
#     xlabel="lvl",
#     ylabel="L^2 error",
#     framestyle = :box,
#     legend_columns=3,
#     legend=:bottomleft
#     )
# xs = xplot#2.0.^(log2.(ns))
# xl = map(x->string(Int(log2((x)))),xs)
# plot!(xticks = (xs, xl))
# savefig(joinpath(@__DIR__,"convergence_stokes.png"))
