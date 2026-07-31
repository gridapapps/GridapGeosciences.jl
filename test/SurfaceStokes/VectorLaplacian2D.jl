""" Solve the vector Laplacian in 2D in primial form using H1 vector-valued elements
-Δᵧ(̃u) = ̃f
where Δᵧ(̃u) = ∇ᵧ(∇ᵧ⋅̃u) - ∇ᵧ^⟂(∇ᵧ^⟂ ⋅ ̃u)
"""

using GridapGeosciences
using Gridap
using Gridap.Helpers

import GridapGeosciences.CellData: deriv_det, deriv_sqrt, cpAB

include("operator.jl")

uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)


function vector_laplacian2d(atlas_model,
  p_fe::Int,dir::String,uX::Function,ls=LUSolver(),return_vtk=false;
  _i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)
 _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  degree = 6*(p_fe+1)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))

  perp_metric_cf = PerpMetric(Ω_atlas) # R*g

  ## Expanded product rule to help with weak form
  # div ( √g u) = div(u)*√g + u⋅gradient(√g)
  divg_product_rule(u) = divergence(u)*meas_cf + u⋅grad_meas_cf

  u_contra_cf = (pinvJ∘covariant_basis_cf)⋅(uX∘ambient_map_cf)

  ## FE spaces: Taylor hood pair --> Q2/Q1 continuous
  reffe_u  = ReferenceFE(lagrangian,VectorValue{2, Float64},p_fe)

  V = TestFESpace(Ω_atlas, reffe_u; conformity=:H1)
  U = TrialFESpace(V)

  # Interpolate the exact solution into the FE space
  u_int = interpolate(u_contra_cf,U)

  rhs_curl_cf = vecΔs_2D(uX, Ω_atlas)
  rhs = -1.0*rhs_curl_cf

  # ## Weak form
  biform(u,v) = ( ∫( (divg_product_rule(u)*divg_product_rule(v))*(1/meas_cf) )dΩ
               +  ∫(  -1.0*(divergence(perp_metric_cf⋅u)*divergence(perp_metric_cf⋅v))*(1/meas_cf)  )dΩ
                          )
  liform(v) = ∫( (rhs⋅(metric_cf⋅v))*meas_cf  )dΩ


  # ## FE problem
  op = AffineFEOperator(biform,liform,U,V)
  A = get_matrix(op)
  b = get_vector(op)
  ns = numerical_setup(symbolic_setup(ls,A),A)
  x = Gridap.Algebra.allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  uh = FEFunction(U,x)

  eu = u_int - uh
  eu_l2 = sqrt(sum(∫( eu⋅(metric_cf⋅eu)*(meas_cf) )dΩ_error))

 _i_am_main && println("eu = $(eu_l2)")

  if return_vtk
      panel_cfs = [covariant_basis_cf⋅ u_int, covariant_basis_cf⋅ uh,eu]
      labels = ["u","uh","eu"]
      cellfields = map((x,y) -> x=>y, labels,panel_cfs)
      writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/surface_laplacian_nref$(lvl)_p$p_fe",
          cellfields=cellfields,append=false)
  end

  return eu_l2, false, false

end


function main(models::AbstractArray;ps=[1,2,3],_i_am_main=true)
  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,vector_laplacian2d,dir,uX,ls;_i_am_main=_i_am_main)
end

# n_ref_lvls = 4
# radius = 1.0
# models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
# main(models)




# using Plots
# e_p1 = [0.0838810603261452, 0.021195507901009464, 0.005517045873698801, 0.0016305133693349221]
# e_p2 = [0.02808268995395562, 0.0036914369043910982, 0.000275514518401057, 2.6693363759415473e-5]
# e_p3 = [0.002301214919325417, 0.00019751766551319804, 1.4756926516761667e-5, 9.86219929353771e-7]

# xplot = [2,4,8,16]
# zz = 2e-1xplot.^(-2)
# ww = 0.75e-1xplot.^(-3)
# qq = 9e-2xplot.^(-4)

# plot()

# plot!(xplot, e_p1,  lw=2,marker=:circle,label="p=1")
# plot!(xplot, e_p2,  lw=2,marker=:square, label="p=2")
# plot!(xplot, e_p3,  lw=2,marker=:diamond,label="p=3")

# plot!(xplot,zz,lw=2,color=:blue,label="lvl^2")
# plot!(xplot,ww,lw=2,color=:orange,label="lvl^3")
# plot!(xplot,qq,lw=2,color=:green,label="lvl^4")
# plot!(shape=:auto,
#     xaxis=:log2,yaxis=:log10,
#     xlabel="lvl",
#     ylabel="L^2 error",
#     framestyle = :box,
#     legend_columns=3,
#     )
# xs = xplot#2.0.^(log2.(ns))
# xl = map(x->string(Int(log2((x)))),xs)
# plot!(xticks = (xs, xl))
# savefig(joinpath(@__DIR__,"convergence_vec_lap_H1.png"))
