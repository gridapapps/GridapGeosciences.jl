"""
In this module, we test the panel ids from the BodyFittedTriangulation trian are
equivalent to the panel ids from the model.
We also test the length of the panel ids is equvialent to the number of cells.
Do this for:
  - CubedSphere2DParametricDistributedDiscreteModel
  - CubedSphere2DParametricOctreeDistributedDiscreteModel
  - CubedSphere3DParametricOctreeDistributedDiscreteModel

  - CubedSphere2DAmbientDistributedDiscreteModel
  - CubedSphere2DAmbientOctreeDistributedDiscreteModel
  - CubedSphere3DAmbientOctreeDistributedDiscreteModel
"""

module DistributedPanelIdsTests

using Gridap
using GridapGeosciences
using GridapDistributed
using PartitionedArrays
using GridapP4est
using Test

function test_distributed_panel_ids(dpanel_model)
  trian = Triangulation(dpanel_model)
  gids = get_cell_gids(dpanel_model)
  map(get_panel_ids(dpanel_model), get_panel_ids(trian), local_views(dpanel_model)) do p1, p2, model
    @test p1 == p2
    @test length(p1) == num_cells(model)
  end

  map(get_owned_panel_ids(dpanel_model),get_panel_ids(trian),partition(gids)) do p1, p2, cid
    owned_cells = own_to_local(cid)
    _p2 = p2[owned_cells]
    @test p1 == _p2
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

  test_distributed_panel_ids(dmodels[1])

  @test true
end


function test_ParametricOctreeDistributedDiscreteModel(distribute,nprocs)

  ranks = distribute(LinearIndices((nprocs,)))

  # i_am_main(ranks) && println("--test CubedSphere2DParametricOctreeDistributedDiscreteModel")

  radius = 1.0

  # level 0
  omodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=0)
  panel_model = omodel.parametric_dmodel
  test_distributed_panel_ids(panel_model)

  # Ω_panel = Triangulation(panel_model)
  # writevtk_with_cell_geomap(geo_map_func(Ω_panel),"model0",append=false)


  # level 1
  omodel, = adapt_model(ranks,omodel)
  panel_model = omodel.parametric_dmodel
  test_distributed_panel_ids(panel_model)

  _omodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=1)
  _panel_model = _omodel.parametric_dmodel
  test_distributed_panel_ids(_panel_model)


  # level 2
  omodel, = adapt_model(ranks,omodel)
  panel_model = omodel.parametric_dmodel
  test_distributed_panel_ids(panel_model)

  _omodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=2)
  _panel_model = _omodel.parametric_dmodel
  test_distributed_panel_ids(_panel_model)

  @test true
end




function test_Parametric3DOctreeDistributedDiscreteModel(distribute,nprocs)

  ranks = distribute(LinearIndices((nprocs,)))

  # i_am_main(ranks) && println("--test 3D CubedSphere3DParametricOctreeDistributedDiscreteModel")

  radius,thickness = 1.0, 0.19

  # level 0
  o3model = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks, radius,thickness;
        num_horizontal_uniform_refinements=0, num_vertical_uniform_refinements=0);
  panel_model = o3model.parametric_dmodel
  test_distributed_panel_ids(panel_model)

  # level 1
  o3model, = adapt_model(ranks,o3model)
  panel_model = o3model.parametric_dmodel
  test_distributed_panel_ids(panel_model)

  _o3model = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks,radius,thickness;
       num_horizontal_uniform_refinements=1, num_vertical_uniform_refinements=1);
  _panel_model = _o3model.parametric_dmodel
  test_distributed_panel_ids(_panel_model)


  # level 2
  o3model, = adapt_model(ranks,o3model)
  panel_model = o3model.parametric_dmodel
  test_distributed_panel_ids(panel_model)

  _o3model = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks,radius,thickness;
      num_horizontal_uniform_refinements=2, num_vertical_uniform_refinements=2);
  _panel_model = _o3model.parametric_dmodel
  test_distributed_panel_ids(_panel_model)

  @test true
end



function test_distributedAmbientDiscreteModel(distribute,nprocs)

  ranks = distribute(LinearIndices((nprocs,)))

  n_ref_lvls = 2
  radius = 1.0
  dmodels = get_distributed_extrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)

  test_distributed_panel_ids(dmodels[1])

  @test true
end

function test_AmbientOctreeDistributedDiscreteModel(distribute,nprocs)

  ranks = distribute(LinearIndices((nprocs,)))
  radius = 1.0

  # level 0
  omodel = CubedSphere2DAmbientOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=0)
  ambient_model = omodel.ambient_dmodel
  test_distributed_panel_ids(ambient_model)


  # level 1
  omodel, = adapt_model(ranks,omodel)
  ambient_model = omodel.ambient_dmodel
  test_distributed_panel_ids(ambient_model)

  _omodel = CubedSphere2DAmbientOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=1)
  _ambient_model = _omodel.ambient_dmodel
  test_distributed_panel_ids(_ambient_model)


  # level 2
  omodel, = adapt_model(ranks,omodel)
  ambient_model = omodel.ambient_dmodel
  test_distributed_panel_ids(ambient_model)

  _omodel = CubedSphere2DAmbientOctreeDistributedDiscreteModel(ranks, radius; num_initial_uniform_refinements=2)
  _ambient_model = _omodel.ambient_dmodel
  test_distributed_panel_ids(_ambient_model)

  @test true

end


function test_Ambient3DOctreeDistributedDiscreteModel(distribute,nprocs)

  ranks = distribute(LinearIndices((nprocs,)))
  radius,thickness = 1.0, 0.19

  # level 0
  omodel = CubedSphere3DAmbientOctreeDistributedDiscreteModel(ranks, radius,thickness;
    num_horizontal_uniform_refinements=0, num_vertical_uniform_refinements=0);
  ambient_model = omodel.ambient_dmodel
  test_distributed_panel_ids(ambient_model)


  # level 1
  omodel, = adapt_model(ranks,omodel)
  ambient_model = omodel.ambient_dmodel
  test_distributed_panel_ids(ambient_model)

  _omodel = CubedSphere3DAmbientOctreeDistributedDiscreteModel(ranks, radius, thickness;
    num_horizontal_uniform_refinements=1, num_vertical_uniform_refinements=1)
  _ambient_model = _omodel.ambient_dmodel
  test_distributed_panel_ids(_ambient_model)


  # level 2
  omodel, = adapt_model(ranks,omodel)
  ambient_model = omodel.ambient_dmodel
  test_distributed_panel_ids(ambient_model)

  _omodel = CubedSphere3DAmbientOctreeDistributedDiscreteModel(ranks, radius, thickness;
  num_horizontal_uniform_refinements=2, num_vertical_uniform_refinements=2)
  _ambient_model = _omodel.ambient_dmodel
  test_distributed_panel_ids(_ambient_model)

end








end # module
