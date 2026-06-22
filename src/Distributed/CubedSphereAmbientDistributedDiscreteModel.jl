################################################################################
#### CubedSphereAmbientDistributedDiscreteModel
#### Has serial CubedSphereAmbientDiscreteModel underneath
#### The provided panel_ids are owned+ghost. Return only owned+ghost in get_panel_ids
#### Return owned in get_owned_panel_ids
#### Implemenet the interface of GridapDistributed.GenericDistributedDiscreteModel
################################################################################

const CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,T} = GridapDistributed.GenericDistributedDiscreteModel{Dc,Dp,<:AbstractArray{T}} where T<:AmbientModels


function CubedSphereAmbientDistributedDiscreteModel(
  dpanel_model::CubedSphereParametricDistributedDiscreteModel
)
  gids = get_cell_gids(dpanel_model)

  ambient_models = map(local_views(dpanel_model)) do panel_model
    CubedSphereAmbientDiscreteModel(panel_model)
   end
  return GridapDistributed.GenericDistributedDiscreteModel(ambient_models,gids)
end

## these are local+ghost panel ids
## for dmodels where the local model is CubedSphereAmbientDiscreteModel
function get_panel_ids(dmodel::CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,T}) where {Dc,Dp,T}
  return map(get_panel_ids,local_views(dmodel))
end

# return owned here to assist with triangulations
function get_owned_panel_ids(dmodel::CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,T}) where {Dc,Dp,T}
  gids = get_cell_gids(dmodel)
  dpanel_ids = get_panel_ids(dmodel)

  panel_ids = map(dpanel_ids,partition(gids)) do pids, cids
    owned_cells = own_to_local(cids)
    return pids[owned_cells]
  end
  return panel_ids

end

function get_forward_map_generator(dmodel::CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,T}) where {Dc,Dp,T}
  return map(get_forward_map_generator,local_views(dmodel))
end

function get_radius(dmodel::CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,T}) where {Dc,Dp,T}
  radii =  map(get_radius,local_views(dmodel))
  radius = zero(eltype(radii))
  map(radii) do r
    radius = r
  end
  return radius
end


function get_thickness(dmodel::CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,T}) where {Dc,Dp,T}
  Ts =  map(get_thickness,local_views(dmodel))
  thickness = zero(eltype(Ts))
  map(Ts) do t
    thickness = t
  end
  return thickness
end

function get_parametric_model(dmodel::CubedSphereAmbientDistributedDiscreteModel{Dc,Dp,T}) where {Dc,Dp,T}
  gids = get_cell_gids(dmodel)
  models = map(local_views(dmodel)) do model
    get_parametric_model(model)
   end
  return GridapDistributed.GenericDistributedDiscreteModel(models,gids)
end





"""
CubedSphereAmbientDistributedDiscreteModel

General constructor to match inputs of CubedSphere2DParametricOctreeDistributedDiscreteModel
Note num_initial_uniform_refinements=1 as default since we want to call Gridap.Adaptivity.get_model,
which is not defined for the coarsest model (i.e. not adapted model)
"""
function CubedSphereAmbientDistributedDiscreteModel(
  ranks::AbstractArray,
  radius::Real;
  num_initial_uniform_refinements=1)
  nprocs = length(ranks)

  dmodels = generate_distributed_refined_models(ranks,nprocs,num_initial_uniform_refinements,radius)
  dpanel_model = dmodels[1]
  CubedSphereAmbientDistributedDiscreteModel(dpanel_model)
end

