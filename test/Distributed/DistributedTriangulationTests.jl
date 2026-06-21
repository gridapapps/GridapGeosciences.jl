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
  test_DistributedDiscreteModel(distribute,nprocs,IntrinsicManifold())
  test_OctreeDistributedDiscreteModel(distribute,nprocs,IntrinsicManifold())
  test_3DDistributedDiscreteModel(distribute,nprocs,IntrinsicManifold())
  test_3DOctreeDistributedDiscreteModel(distribute,nprocs,IntrinsicManifold())
end

function _test_2D_trians(atlas_model)
  trian = Triangulation(atlas_model)
  test_triangulation(trian)
  btrian = BoundaryTriangulation(atlas_model)
  test_triangulation(btrian)
  strian = SkeletonTriangulation(atlas_model)
  test_triangulation(strian)
end

function test_DistributedDiscreteModel(distribute,nprocs,manifold_style)
  ranks = distribute(LinearIndices((nprocs,)))
  n_ref_lvls = 2
  radius = 1.0
  coarse_mesh = CubedSphereMesh(radius)
  atlas_model = AtlasDiscreteModel(ranks,
                                   coarse_mesh,
                                   n_ref_lvls;
                                   manifold_style=manifold_style)
  _test_2D_trians(atlas_model)
  @test true
end

function test_OctreeDistributedDiscreteModel(distribute,nprocs,manifold_style)
  ranks = distribute(LinearIndices((nprocs,)))
  n_ref_lvls = 2
  radius = 1.0
  coarse_mesh = CubedSphereMesh(radius)
  o3model = AtlasOctreeDistributedDiscreteModel(ranks, 
                                               coarse_mesh,
                                               n_ref_lvls; 
                                               manifold_style=manifold_style)
  atlas_model = get_atlas_model(o3model)
  _test_2D_trians(atlas_model)
  @test true
end

function _test_3D_trians(atlas_model)
    trian = Triangulation(atlas_model)
  test_triangulation(trian)

  strian = SkeletonTriangulation(atlas_model)
  test_triangulation(strian)

  tags = ["bottom_boundary"]
  Γ = BoundaryTriangulation(atlas_model,tags=tags)
  cell_geo_map = AmbientMapCellField(Γ)
  test_triangulation(Γ)
  # writevtk_with_cell_geomap(cell_geo_map,Γ,dir*"/boundary_bottom",append=false)

  tags = ["top_boundary"]
  Γ = BoundaryTriangulation(atlas_model,tags=tags)
  Γ.trians.item_ref[].parent.glue.face_to_bgface
  cell_geo_map = AmbientMapCellField(Γ)
  test_triangulation(Γ)
  # writevtk_with_cell_geomap(cell_geo_map,Γ,dir*"/boundary_top",append=false)

  tags = ["intermediate_boundary"]
  Γ = BoundaryTriangulation(atlas_model,tags=tags)
  cell_geo_map = AmbientMapCellField(Γ)
  test_triangulation(Γ)
  # writevtk_with_cell_geomap(cell_geo_map,Γ,dir*"/boundary_intermediate",append=false)
end 

function test_3DDistributedDiscreteModel(distribute,nprocs,manifold_style)
  ranks = distribute(LinearIndices((nprocs,)))
  radius,thickness = 1.0, 0.19
  coarse_mesh = CubedSphereWithThicknessMesh(radius,thickness)
  n_ref_lvls = 2
  atlas_model = AtlasDiscreteModel(ranks,
                                   coarse_mesh,
                                   n_ref_lvls;
                                   manifold_style=manifold_style);
  _test_3D_trians(atlas_model)
  @test true
end

function test_3DOctreeDistributedDiscreteModel(distribute,nprocs,manifold_style)
  ranks = distribute(LinearIndices((nprocs,)))
  radius,thickness = 1.0, 0.19
  coarse_mesh = CubedSphereWithThicknessMesh(radius,thickness)
  n_ref_lvls = 2
  o3model = AtlasOctreeDistributedDiscreteModel(ranks,
                                                coarse_mesh,
                                                n_ref_lvls;
                                                manifold_style=manifold_style);
  atlas_model = get_atlas_model(o3model)
  _test_3D_trians(atlas_model)
  @test true
end

end ## module
