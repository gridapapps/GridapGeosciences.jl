"""
CubedSphereAmbientOctreeDistributedDiscreteModel

Generic struct that holds an omodel of the topology and a dmodel of the ambient space
Dispatch through the ambient model
"""
struct CubedSphereAmbientOctreeDistributedDiscreteModel{Dc,Dp,
  A<:OctreeDistributedDiscreteModel{Dc,Dc}, #### Assuming!!!
  B<:CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,<:CubedSphereAmbientDiscreteModel}} <: GridapDistributed.DistributedDiscreteModel{Dc,Dp}
  octree_dmodel::A
  ambient_dmodel::B
end


function CubedSphere2DAmbientOctreeDistributedDiscreteModel(
  ranks::AbstractArray,
  radius::Real;
  num_initial_uniform_refinements=0)

  omodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(
    ranks,radius;num_initial_uniform_refinements=num_initial_uniform_refinements)

  ambient_dmodel = CubedSphereAmbientDistributedDiscreteModel(omodel.parametric_dmodel)
  A = typeof(omodel.octree_dmodel)
  B = typeof(ambient_dmodel)
  Dc = num_cell_dims(ambient_dmodel)
  Dp = num_point_dims(ambient_dmodel)
  CubedSphereAmbientOctreeDistributedDiscreteModel{Dc,Dp,A,B}(omodel.octree_dmodel,ambient_dmodel)
end

function CubedSphere3DAmbientOctreeDistributedDiscreteModel(
  ranks::AbstractArray,
  radius::Real,thickness::Real;
  num_horizontal_uniform_refinements=0,
  num_vertical_uniform_refinements=0
  )

  omodel = CubedSphere3DParametricOctreeDistributedDiscreteModel(
    ranks,radius,thickness;
    num_horizontal_uniform_refinements=num_horizontal_uniform_refinements,
    num_vertical_uniform_refinements=num_vertical_uniform_refinements)

  ambient_dmodel = CubedSphereAmbientDistributedDiscreteModel(omodel.parametric_dmodel)

  A = typeof(omodel.octree_dmodel)
  B = typeof(ambient_dmodel)
  Dc = num_cell_dims(ambient_dmodel)
  Dp = num_point_dims(ambient_dmodel)

  CubedSphereAmbientOctreeDistributedDiscreteModel{Dc,Dp,A,B}(omodel.octree_dmodel,ambient_dmodel)
end

get_radius(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel) = get_radius(dmodel.ambient_dmodel)
get_thickness(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel) = get_thickness(dmodel.ambient_dmodel)
get_panel_ids(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel) = get_panel_ids(dmodel.ambient_dmodel)
get_owned_panel_ids(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel) = get_owned_panel_ids(dmodel.ambient_dmodel)
get_forward_map_generator(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel) = get_forward_map_generator(dmodel.ambient_dmodel)
get_parametric_model(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel) = get_parametric_model(dmodel.ambient_dmodel)

function get_parametric_model(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel{2,Dp}) where Dp
  octree_dmodel = dmodel.octree_dmodel
  amodel = dmodel.ambient_dmodel
  generic_dmodel = get_parametric_model(amodel)
  CubedSphere2DParametricOctreeDistributedDiscreteModel(octree_dmodel, generic_dmodel)
end

function get_parametric_model(dmodel::CubedSphereAmbientOctreeDistributedDiscreteModel{3,Dp}) where Dp
  octree_dmodel = dmodel.octree_dmodel
  amodel = dmodel.ambient_dmodel
  generic_dmodel = get_parametric_model(amodel)
  CubedSphere3DParametricOctreeDistributedDiscreteModel(octree_dmodel, generic_dmodel)
end
