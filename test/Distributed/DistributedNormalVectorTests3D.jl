"""
In this module, test the mapping of (1,0,0) as the radial normal vector in
ambient space
"""

module DistributedNormalTests3D

using Gridap
using GridapGeosciences
using GridapP4est
using Test


function main(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  radius,thickness = 1.0, 0.19

  o3model = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks,radius,thickness;
  num_horizontal_uniform_refinements=2, num_vertical_uniform_refinements=2);
  panel_model = o3model.parametric_dmodel

  Ω_panel =  Triangulation(panel_model)
  dΩ = Measure(Ω_panel,4)

  ## the normal in parametric space (γ,α,β) is (1,0,0)
  n3D_panel = CellField(VectorValue(1.0,0.0,0.0),Ω_panel)
  J_cf = ParametricCellField(forward_jacobian,Ω_panel)
  inv_cf = ParametricCellField(inv_metric,Ω_panel)

  ## map the normal from parametric space -> ambient space
  _n_mapped = J_cf ⋅ (inv_cf  ⋅ n3D_panel )
  ff = Operation(sqrt)(  n3D_panel   ⋅ (inv_cf⋅ n3D_panel )  )
  n_mapped = _n_mapped/ff

  ## the unit surface normal is given by the position vector
  vX = panel_to_cartesian(normal_vec)
  norm_vec_cf = ParametricCellField(vX,Ω_panel)

  metric_cf = ParametricCellField(metric,Ω_panel)
  _e = norm_vec_cf-n_mapped
  e = sum(∫( _e⋅(metric_cf⋅_e ) )dΩ)
  @test e < 1e-12

  # if return_vtk
  #   lvl = nref(panel_model)
  #   panel_cfs = [ n_mapped,norm_vec_cf,norm_vec_cf-n_mapped]
  #   labels = ["n_mapped", "n_vec", "diff"]
  #   cellfields = map((x,y) -> x=>y, labels,panel_cfs)
  #   writevtk_with_cell_geomap(AmbientMapCellField(Ω_panel),dir*"/ambient_model_nref$(lvl)",cellfields=cellfields,append=false)
  # end

end

end # module
