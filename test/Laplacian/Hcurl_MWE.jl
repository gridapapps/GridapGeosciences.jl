using Gridap
using GridapDistributed
using GridapGeosciences
using GridapP4est
using PartitionedArrays, MPI
using Gridap.CellData

transpose_jacobian(p) = x -> transpose(forward_jacobian(p)(x))

covar_v_3D(vecX::Function,m) = αβ -> transpose_jacobian(m)(αβ) ⋅ vecX(m)(αβ)
covar_v_3D(vecX::Function) = m -> covar_v_3D(vecX,m)

function uX(forward_map)
  function _u(γαβ)
    xyz = forward_map(γαβ)
    # VectorValue(-xyz[2],xyz[1],0.0)

    r = sqrt(xyz[1]^2 + xyz[2]^2 + xyz[3]^2)
    f = 2.0*xyz[3]/r
    n = normal_vec(xyz)
    f*n
  end
end


dir = @__DIR__
radius = 1.0
thickness = 0.19

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

dmodel = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks,
                                                    radius,thickness;
                                                    num_horizontal_uniform_refinements=0,
                                                    num_vertical_uniform_refinements=0)
dpanel_model = dmodel.parametric_dmodel

# extract serial model
panel_model = dpanel_model.models.item
p_fe = 1
Ω_panel = Triangulation(panel_model)

## FE spaces
R = TestFESpace(panel_model, ReferenceFE(nedelec,Float64,p_fe);conformity=:Hcurl)
H = TrialFESpace(R)

## metric information
inv_metric_cf = ParametricCellField(inv_metric,Ω_panel)
metric_cf = ParametricCellField(metric,Ω_panel)
meas_cf = ParametricCellField(sqrtg,Ω_panel)
covariant_basis_cf = ParametricCellField(covariant_basis,Ω_panel)

u_cov_cf = ParametricCellField(covar_v_3D(uX),Ω_panel)
u_int = interpolate(u_cov_cf,H)

curl_func(p,x) = curl(covar_v_3D(uX)(p))(x)
curl_func(p) = x -> curl_func(p,x)
curl_cf = ParametricCellField(curl_func,Ω_panel)

q1 = covariant_basis_cf ⋅ (curl(u_int)/meas_cf)
q2 = covariant_basis_cf ⋅ (curl_cf/meas_cf)

ref_pts = get_cell_ref_coordinates(panel_model)
cmap = get_cell_map(get_grid(panel_model))
param_pts = lazy_map(evaluate,cmap,ref_pts)

# check domain style
DomainStyle(q1) ## Referencce
DomainStyle(q2) ## Physical -> parametric domain

# evaluate on points
q1_x = lazy_map(evaluate,get_data(q1),ref_pts)./1
q2_x = lazy_map(evaluate,get_data(q2),param_pts)./1

q1_x[1]
q2_x[1]

### Plottting
cellfields = ["curlu"=> q1,
              "curlu_cf"=> q2,
              "diff_curl" =>q1-q2,
              "u"=>covariant_basis_cf ⋅ (inv_metric_cf⋅u_int),
              "u_cf"=>covariant_basis_cf ⋅ (inv_metric_cf⋅u_cov_cf),
              ]
writevtk_with_cell_geomap(latlon_geo_map_func(Ω_panel),Ω_panel,dir*"/sol",
        cellfields=cellfields,
        append=false)
