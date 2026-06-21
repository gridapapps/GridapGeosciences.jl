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
  coarse_mesh = CubedSphereWithThicknessMesh(radius,thickness)
  
  num_refinements = 2
  o3model = AtlasOctreeDistributedDiscreteModel(ranks, 
                                                coarse_mesh, 
                                                num_refinements; 
                                                manifold_style=IntrinsicManifold())
  atlas_model = get_atlas_model(o3model)

  Ω_panel =  Triangulation(atlas_model)
  dΩ = Measure(Ω_panel,4)

  ## the normal in parametric space (γ,α,β) is (1,0,0)
  n3D_panel = CellField(VectorValue(1.0,0.0,0.0),Ω_panel)
  ambient_map_cf = AmbientMapCellField(Ω_panel)
  J_cf = transpose∘∇(ambient_map_cf)
  inv_cf = InvMetricCellField(Ω_panel)

  ## map the normal from parametric space -> ambient space
  _n_mapped = J_cf ⋅ (inv_cf  ⋅ n3D_panel )
  ff = Operation(sqrt)(  n3D_panel   ⋅ (inv_cf⋅ n3D_panel )  )
  n_mapped = _n_mapped/ff

  ## the unit surface normal is given by the position vector
  norm_vec_cf = normal_vec∘ambient_map_cf

  metric_cf = MetricCellField(Ω_panel)
  _e = norm_vec_cf-n_mapped
  e = sum(∫( _e⋅(metric_cf⋅_e ) )dΩ)
  @test e < 1e-12

  # if return_vtk
  #   lvl = nref(panel_model)
  #   panel_cfs = [ n_mapped,norm_vec_cf,norm_vec_cf-n_mapped]
  #   labels = ["n_mapped", "n_vec", "diff"]
  #   cellfields = map((x,y) -> x=>y, labels,panel_cfs)
  #   writevtk_with_cell_geomap(ambient_map_cf,dir*"/ambient_model_nref$(lvl)",cellfields=cellfields,append=false)
  # end

end

end # module
