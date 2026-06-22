function coarse_parametric_model(radius)
  cube_model = coarse_cube_model()
  panel_model = parametric_model(cube_model,radius)
  return panel_model
end

# TO-DO: decide better name (???)
function setup_panel_cmaps(cube_cmaps, panel_ids)
  # I removed MatMultField .... I do not think we will needed it. 
  # This is why I commented it out.  
  # cube2panel_map = lazy_map(p->MatMultField( A_cube2panel[p] ), panel_ids)
  # panel_cmaps = lazy_map(∘,cube2panel_map,cube_cmaps)
end

function parametric_model(cube_model, radius)
  Dc = num_cell_dims(cube_model)
  Dp = num_point_dims(cube_model)

  @check Dp == Dc+1

  panel_ids = get_panel_ids(cube_model)
  cube_grid = get_grid(cube_model)
  cube_topo = get_grid_topology(cube_model)
  cube_nodes = get_node_coordinates(cube_grid)

  ## map the cube to the parametric domain
  cube_cmaps  = get_cell_map(cube_grid)
  panel_cmaps = setup_panel_cmaps(cube_cmaps, panel_ids)

  ## create the panel model
  ## to correctly trigger Dc=2,Dp=2, need to have 2D nodes.
  ## the panel_nodes are just junk 2D nodes, never used by Gridap
  ## the panel_grid has the bespoke panel_cmap
  ## the panel_topo is the same as the cube, but with the 2D nodes so that Dp=2
  ## the panel_labels are from the panel_topo
  panel_nodes = map(x->Point(x[2],x[3]),cube_nodes) # these are just junk nodes, never used
  panel_grid = Geometry.UnstructuredGrid(panel_nodes,get_cell_node_ids(cube_grid),get_reffes(cube_grid),get_cell_type(cube_grid),OrientationStyle(cube_grid),
                      nothing,panel_cmaps)
  panel_topo = UnstructuredGridTopology(panel_nodes,get_cell_node_ids(cube_grid),get_cell_type(cube_topo),get_polytopes(cube_topo),OrientationStyle(cube_topo))
  panel_labels = FaceLabeling(panel_topo)

  panel_model = CubedSphere2DParametricDiscreteModel(panel_grid,panel_topo,panel_labels,panel_ids,radius)

  return panel_model
end


struct CubedSphereParametricDiscreteModel{Dc,Dp,Tp,B,Tf<:Map} <: DiscreteModel{Dc,Dp}
  grid::UnstructuredGrid{Dc,Dp,Tp,B}
  grid_topology::UnstructuredGridTopology{Dc,Dp,Tp,B}
  face_labeling::FaceLabeling
  panel_ids::AbstractArray{Int}
  forward_map_generator::Tf
end

const CubedSphere2DParametricDiscreteModel{Tp,B,Tf} = CubedSphereParametricDiscreteModel{2,2,Tp,B,Tf}
const CubedSphere3DParametricDiscreteModel{Tp,B,Tf} = CubedSphereParametricDiscreteModel{3,3,Tp,B,Tf}


function CubedSphere2DParametricDiscreteModel(grid::UnstructuredGrid{2,2},
                                 grid_topology::UnstructuredGridTopology{2,2},
                                 face_labeling::FaceLabeling,
                                 panel_ids::AbstractArray{Int},
                                 radius::Real)
  forward_map_generator = ForwardMap2DGenerator(radius)
  return CubedSphereParametricDiscreteModel(grid, grid_topology, face_labeling, panel_ids, forward_map_generator)
end

function CubedSphere3DParametricDiscreteModel(grid::UnstructuredGrid{3,3},
                                 grid_topology::UnstructuredGridTopology{3,3},
                                 face_labeling::FaceLabeling,
                                 panel_ids::AbstractArray{Int},
                                 radius::Real,
                                 thickness::Real)
  forward_map_generator = ForwardMap3DGenerator(radius, thickness)
  return CubedSphereParametricDiscreteModel(grid, grid_topology, face_labeling, panel_ids, forward_map_generator)
end

Geometry.get_grid(model::CubedSphereParametricDiscreteModel) = model.grid
Geometry.get_grid_topology(model::CubedSphereParametricDiscreteModel) = model.grid_topology
Geometry.get_face_labeling(model::CubedSphereParametricDiscreteModel) = model.face_labeling
get_panel_ids(model::CubedSphereParametricDiscreteModel) = model.panel_ids
Geometry.get_cell_map(model::CubedSphereParametricDiscreteModel) = model.grid.cell_map
get_forward_map_generator(model::CubedSphereParametricDiscreteModel) = model.forward_map_generator
get_forward_map_generator(model::AdaptedDiscreteModel{Dc,Dp,<:CubedSphereParametricDiscreteModel}) where {Dc,Dp} = model.model.forward_map_generator

function get_radius(model::Union{<:CubedSphereParametricDiscreteModel,AdaptedDiscreteModel{Dc,Dp,<:CubedSphereParametricDiscreteModel}}) where {Dc,Dp}
  generator = get_forward_map_generator(model)
  return generator.radius
end

function get_thickness(model::Union{CubedSphere2DParametricDiscreteModel,AdaptedDiscreteModel{2,2,<:CubedSphere2DParametricDiscreteModel}})
  @notimplemented """\n
  The model is two dimensional, get_thickness not defined.
  """
end

function get_thickness(model::Union{CubedSphere3DParametricDiscreteModel,AdaptedDiscreteModel{3,3,<:CubedSphere3DParametricDiscreteModel}})
  generator = get_forward_map_generator(model)
  return generator.thickness
end



"""
CubedSphere2DParametricDiscreteModel

General constructor to match inputs of CubedSphere2DParametricOctreeDistributedDiscreteModel
"""

function CubedSphere2DParametricDiscreteModel(
  radius::Real;
  num_initial_uniform_refinements=0)

  if num_initial_uniform_refinements == 0
    return coarse_parametric_model(radius)
  end

  models = generate_refined_models(num_initial_uniform_refinements, CubedSphereMesh(radius), IntrinsicManifold())
  models[1]
end

"""
CubedSphere3DParametricDiscreteModel

General constructor to match inputs of CubedSphere3DParametricOctreeDistributedDiscreteModel
Returns not implemented error
"""

function CubedSphere3DParametricDiscreteModel(
  radius::Real,thickness::Real;
  num_horizontal_uniform_refinements=0,
  num_vertical_uniform_refinements=0)

  @notimplemented """\n
  No serial 3D cubed sphere available, use the MPI version
  """
end


const ParametricModels{Dc,Dp} = Union{CubedSphereParametricDiscreteModel{Dc,Dp},AdaptedDiscreteModel{Dc,Dp,<:CubedSphereParametricDiscreteModel}}
