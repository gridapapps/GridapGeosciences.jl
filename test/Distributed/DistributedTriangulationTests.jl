"""
Test the construction of meshes by evaluating the cellmaps
"""

module DistributedTriangulationTests

using Gridap
using GridapGeosciences
using GridapDistributed
using GridapP4est
using Test

################################################################################
## Test the evaluation of cmaps on DistributedTriangulations
## i.e. are all the cellmaps there
################################################################################

function test_triangulation(trian::GridapDistributed.DistributedTriangulation)
  map(trian.trians) do trian
    cmap = get_cell_map(trian)
    pts = get_cell_ref_coordinates(trian)
    lazy_map(evaluate,cmap,pts)
    @test true
  end
end

function main(distribute,nprocs)
  test_distributedParametricDiscreteModel(distribute,nprocs)
  test_ParametricOctreeDistributedDiscreteModel(distribute,nprocs)
  test_Parametric3DOctreeDistributedDiscreteModel(distribute,nprocs)

  test_distributedAmbientDiscreteModel(distribute,nprocs)
  test_AmbientOctreeDistributedDiscreteModel(distribute,nprocs)
  test_Ambient3DOctreeDistributedDiscreteModel(distribute,nprocs)
end

function test_distributedParametricDiscreteModel(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  # i_am_main(ranks) && println("--test CubedSphereParametricDistributedDiscreteModel")

  n_ref_lvls = 2
  radius = 1.0
  dmodels = get_distributed_intrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)

  model = dmodels[2]

  trian = Triangulation(model)
  test_triangulation(trian)

  btrian = BoundaryTriangulation(model)
  test_triangulation(btrian)

  strian = SkeletonTriangulation(model)
  test_triangulation(strian)

  @test true
end


function test_ParametricOctreeDistributedDiscreteModel(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  # i_am_main(ranks) && println("--test CubedSphere2DParametricOctreeDistributedDiscreteModel")

  n_ref_lvls = 2
  radius = 1.0
  omodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=n_ref_lvls)
  model = omodel.parametric_dmodel

  trian = Triangulation(model)
  test_triangulation(trian)

  btrian = BoundaryTriangulation(model)
  test_triangulation(btrian)

  strian = SkeletonTriangulation(model)
  test_triangulation(strian)

  @test true
end


function test_Parametric3DOctreeDistributedDiscreteModel(distribute,nprocs)

  ranks = distribute(LinearIndices((nprocs,)))

  radius,thickness = 1.0, 0.19
  # i_am_main(ranks) && println("--test 3D CubedSphere3DParametricOctreeDistributedDiscreteModel")
  n_ref_lvls = 2
  o3model = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks,radius,thickness;
  num_horizontal_uniform_refinements=n_ref_lvls, num_vertical_uniform_refinements=n_ref_lvls);
  panel_model = o3model.parametric_dmodel

  trian = Triangulation(panel_model)
  test_triangulation(trian)

  strian = SkeletonTriangulation(panel_model)
  test_triangulation(strian)



  tags = ["bottom_boundary"]
  Γ = BoundaryTriangulation(panel_model,tags=tags)
  ## TO DO: replace with AmbientMapCellField
  cell_geo_map = geo_map_func(get_forward_map_generator(panel_model),get_panel_ids(Γ))
  test_triangulation(Γ)
  # writevtk_with_cell_geomap(cell_geo_map,Γ,dir*"/boundary_bottom",append=false)

  tags = ["top_boundary"]
  Γ = BoundaryTriangulation(panel_model,tags=tags)
  Γ.trians.item_ref[].parent.glue.face_to_bgface
  ## TO DO: replace with AmbientMapCellField
  cell_geo_map = geo_map_func(get_forward_map_generator(panel_model),get_panel_ids(Γ))
  test_triangulation(Γ)
  # writevtk_with_cell_geomap(cell_geo_map,Γ,dir*"/boundary_top",append=false)

  tags = ["intermediate_boundary"]
  Γ = BoundaryTriangulation(panel_model,tags=tags)
  ## TO DO: replace with AmbientMapCellField
  cell_geo_map = geo_map_func(get_forward_map_generator(panel_model),get_panel_ids(Γ))
  test_triangulation(Γ)
  # writevtk_with_cell_geomap(cell_geo_map,Γ,dir*"/boundary_intermediate",append=false)

  @test true
end



function test_distributedAmbientDiscreteModel(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  n_ref_lvls = 2
  radius = 1.0
  dmodels = get_distributed_extrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)

  model = dmodels[2]

  trian = Triangulation(model)
  test_triangulation(trian)

  btrian = BoundaryTriangulation(model)
  test_triangulation(btrian)

  strian = SkeletonTriangulation(model)
  test_triangulation(strian)

  @test true
end






function test_AmbientOctreeDistributedDiscreteModel(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  # i_am_main(ranks) && println("--test CubedSphere2DAmbientOctreeDistributedDiscreteModel")

  n_ref_lvls = 2
  radius = 1.0
  omodel = CubedSphere2DAmbientOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=n_ref_lvls)
  model = omodel.ambient_dmodel

  trian = Triangulation(model)
  test_triangulation(trian)

  btrian = BoundaryTriangulation(model)
  test_triangulation(btrian)

  strian = SkeletonTriangulation(model)
  test_triangulation(strian)

  @test true
end


function test_Ambient3DOctreeDistributedDiscreteModel(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  radius,thickness = 1.0, 0.19
  n_ref_lvls = 2

  o3model = CubedSphere3DAmbientOctreeDistributedDiscreteModel(ranks,radius,thickness;
  num_horizontal_uniform_refinements=n_ref_lvls, num_vertical_uniform_refinements=n_ref_lvls);
  ambient_model = o3model.ambient_dmodel

  trian = Triangulation(ambient_model)
  test_triangulation(trian)

  strian = SkeletonTriangulation(ambient_model)
  test_triangulation(strian)
  # writevtk(strian,dir*"/skel",append=false)

  tags = ["bottom_boundary"]
  Γ = BoundaryTriangulation(ambient_model,tags=tags)
  test_triangulation(Γ)
  # writevtk(Γ,dir*"/boundary_bottom",append=false)

  tags = ["top_boundary"]
  Γ = BoundaryTriangulation(ambient_model,tags=tags)
  test_triangulation(Γ)
  # writevtk(Γ,dir*"/boundary_top",append=false)

  tags = ["intermediate_boundary"]
  Γ = BoundaryTriangulation(ambient_model,tags=tags)
  test_triangulation(Γ)
  # writevtk(Γ,dir*"/boundary_intermediate",append=false)

  @test true

end

end ## module
