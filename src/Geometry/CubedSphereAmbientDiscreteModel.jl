"""
CubedSphereAmbientDiscreteModel

is a discrete model of the ambient space.
Constructed by composig the cmap of the panel_model with the forward_map
The panel_model is kept for the purposes of dispatching
"""

struct CubedSphereAmbientDiscreteModel{Dc,Dp,Tp,B,P<:ParametricModels} <: DiscreteModel{Dc,Dp}
  grid::UnstructuredGrid{Dc,Dp,Tp,B}
  grid_topology::UnstructuredGridTopology{Dc,Dp,Tp,B}
  face_labeling::FaceLabeling
  panel_model::P
end

Geometry.get_grid(model::CubedSphereAmbientDiscreteModel) = model.grid
Geometry.get_grid_topology(model::CubedSphereAmbientDiscreteModel) = model.grid_topology
Geometry.get_face_labeling(model::CubedSphereAmbientDiscreteModel) = model.face_labeling
get_panel_ids(model::CubedSphereAmbientDiscreteModel) = get_panel_ids(model.panel_model)
Geometry.get_cell_map(model::CubedSphereAmbientDiscreteModel) = model.grid.cell_map
get_forward_map_generator(model::CubedSphereAmbientDiscreteModel) = get_forward_map_generator(model.panel_model)
get_parametric_model(model::CubedSphereAmbientDiscreteModel) = model.panel_model
get_parametric_model(model::AdaptedDiscreteModel{Dc,Dp,<:CubedSphereAmbientDiscreteModel}) where {Dc,Dp} = get_parametric_model(model.model)

const AmbientModels{Dc,Dp} = Union{CubedSphereAmbientDiscreteModel{Dc,Dp},AdaptedDiscreteModel{Dc,Dp,<:CubedSphereAmbientDiscreteModel}}

function get_radius(model::AmbientModels{Dc,Dp}) where {Dc,Dp}
  get_radius(get_parametric_model(model))
end

function get_thickness(model::AmbientModels{Dc,Dp}) where {Dc,Dp}
  get_thickness(get_parametric_model(model))
end

## TO DO: generalise to 3D...
function get_inverse_map_generator(model::AmbientModels{2,Dp}) where {Dp}
  radius = get_radius(model)
  InverseMap2DGenerator(radius)
end

function get_inverse_map_generator(model::AmbientModels{3,Dp}) where {Dp}
  radius = get_radius(model)
  thickness = get_thickness(model)
  InverseMap3DGenerator(radius,thickness)
end

function CubedSphereAmbientDiscreteModel(
  radius::Real;
  num_initial_uniform_refinements=0)


  if num_initial_uniform_refinements == 0
    return coarse_ambient_model(radius)
  end

  models = get_extrinsic_cubed_sphere_refined_models(num_initial_uniform_refinements,radius)
  models[1]
end

function CubedSphereAmbientDiscreteModel(model::AdaptedDiscreteModel{Dc,Dp,<:CubedSphereParametricDiscreteModel}) where {Dc,Dp}
  ref_model = CubedSphereAmbientDiscreteModel(model.model)
  # AdaptedDiscreteModel(ref_model,model,model.glue)
end

function CubedSphereAmbientDiscreteModel(panel_model::CubedSphereParametricDiscreteModel)

  panel_grid = get_grid(panel_model)
  panel_topo = get_grid_topology(panel_model)
  labels = get_face_labeling(panel_model)

  ## map: reffe -> alpha,beta
  cmap = get_cell_map(panel_grid)

  ## map: alpha,beta -> manifold
  fwd_map_generator = get_forward_map_generator(panel_model)
  panel_ids = get_panel_ids(panel_model)
  fwd_map =  geo_map_func(fwd_map_generator,panel_ids)

  ## map: reffe -> manifold
  geo_cmap = lazy_map(∘,fwd_map,cmap)

  ref_pts = get_cell_ref_coordinates(panel_grid)
  ambient_cell_coords = lazy_map(evaluate,geo_cmap,ref_pts)

  eT = eltype(testitem(ambient_cell_coords))
  panel_nodes = Gridap.Geometry.num_nodes(panel_grid)
  ambient_nodes = fill(zero(eT),panel_nodes)

  # Now create proper nodes from the cell-wise array of coords
  # We can do this for the ambient model because the nodes are defined uniquely
  # in ambient space
  cell_to_nodes = get_cell_node_ids(panel_grid)
  Gridap.FESpaces._free_and_dirichlet_values_fill!(
    ambient_nodes, eT[],
    array_cache(ambient_cell_coords),
    array_cache(cell_to_nodes),
    ambient_cell_coords,
    cell_to_nodes,
    eachindex(cell_to_nodes)
  )


  ## the ambient_grid has the bespoke panel_2_ambient_cmap
  ambient_grid = Gridap.Geometry.UnstructuredGrid(ambient_nodes,get_cell_node_ids(panel_grid),get_reffes(panel_grid),get_cell_type(panel_grid),OrientationStyle(panel_grid),
                      nothing,geo_cmap)
  ambient_topo = UnstructuredGridTopology(ambient_nodes,get_cell_node_ids(panel_grid),get_cell_type(panel_topo),get_polytopes(panel_topo),OrientationStyle(panel_topo))
  # ambient_labels = FaceLabeling(ambient_topo)

  CubedSphereAmbientDiscreteModel(ambient_grid,ambient_topo,labels,panel_model)

end

function coarse_ambient_model(radius)
  panel_model0 = coarse_parametric_model(radius)
  CubedSphereAmbientDiscreteModel(panel_model0)
end


function get_surface_normal(trian::AdaptedTriangulation)
  get_surface_normal(trian.trian)
end



