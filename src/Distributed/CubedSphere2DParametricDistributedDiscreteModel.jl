################################################################################
#### CubedSphereParametricDistributedDiscreteModel
#### Has serial CubedSphereParametricDiscreteModel underneath
#### The provided panel_ids are owned+ghost. Return only owned+ghost in get_panel_ids
#### Return owned in get_owned_panel_ids
#### Implemenet the interface of GridapDistributed.GenericDistributedDiscreteModel
################################################################################
# const CubedSphereParametricDistributedDiscreteModel{Dc,Dp,T} = GridapDistributed.GenericDistributedDiscreteModel{Dc,Dp,<:AbstractArray{T}} where T<:Union{<:CubedSphereParametricDiscreteModel{Dc,Dp},<:AdaptedDiscreteModel{Dc,Dp}}

const CubedSphere2DParametricDistributedDiscreteModel{T} = GridapDistributed.GenericDistributedDiscreteModel{2,2,<:AbstractArray{T}} where T<:Union{<:CubedSphere2DParametricDiscreteModel,<:AdaptedDiscreteModel{2,2}}
const CubedSphere3DParametricDistributedDiscreteModel{T} = GridapDistributed.GenericDistributedDiscreteModel{3,3,<:AbstractArray{T}} where T<:Union{<:CubedSphere3DParametricDiscreteModel,<:AdaptedDiscreteModel{3,3}}

const CubedSphereParametricDistributedDiscreteModel{T} = Union{CubedSphere2DParametricDistributedDiscreteModel{T},CubedSphere3DParametricDistributedDiscreteModel{T}}

# # Note the dmodel passed in does not have CubedSphereParametricDiscreteModel as local models
# # Thus, we need to explicitly pass the radius to the constructor as well
# # I am unsure where to put the radius, store in metadata for now
# function CubedSphere2DParametricDistributedDiscreteModel(
#   model::GridapDistributed.DistributedDiscreteModel,
#   dpanel_ids::AbstractArray,
#   radius::Real
# )
#   gids = get_cell_gids(model)

#   models = map(local_views(model),dpanel_ids,partition(gids)) do model, pids, cids
#     _grid = get_grid(model)

#     ## construct the new grid by hand, so that specific cmaps are given
#     nodes = collect1d(get_node_coordinates(_grid)) # these are just junk nodes, never used
#     ctype = collect1d(get_cell_type(_grid))
#     cmaps = collect1d(get_cell_map(_grid))
#     grid = Gridap.Geometry.UnstructuredGrid(nodes,get_cell_node_ids(_grid),
#               get_reffes(_grid),ctype,OrientationStyle(_grid),
#                         nothing,cmaps)

#     topo = UnstructuredGridTopology(get_grid_topology(model))
#     labels = FaceLabeling(topo)

#     # extract the owned panel ids
#     # owned_cells = own_to_local(cids)
#     # panel_ids = pids[owned_cells]
#     CubedSphere2DParametricDiscreteModel(grid,topo,labels,pids,radius)
#    end
#   return GridapDistributed.GenericDistributedDiscreteModel(models,gids)
# end

# ## these are local+ghost panel ids
# ## for dmodels where the local model is CubedSphereParametricDiscreteModel
# function get_panel_ids(dmodel::CubedSphereParametricDistributedDiscreteModel{<:CubedSphereParametricDiscreteModel{Dc,Dp}}) where {Dc,Dp}
#   return map(get_panel_ids,local_views(dmodel))
# end

# ## these are local+ghost panel ids
# ## for omodels where the local model is AdaptedDiscreteModel
# function get_panel_ids(dmodel::CubedSphereParametricDistributedDiscreteModel{<:AdaptedDiscreteModel{Dc,Dp}}) where {Dc,Dp}
#   panel_ids = map(local_views(dmodel)) do lmodel
#     get_panel_ids(lmodel.model)
#   end
#   return panel_ids
# end

# # return owned here to assist with triangulations
# function get_owned_panel_ids(dmodel::CubedSphereParametricDistributedDiscreteModel{T}) where {T}
#   gids = get_cell_gids(dmodel)
#   dpanel_ids = get_panel_ids(dmodel)

#   panel_ids = map(dpanel_ids,partition(gids)) do pids, cids
#     owned_cells = own_to_local(cids)
#     return pids[owned_cells]
#   end
#   return panel_ids

# end

# function get_forward_map_generator(dmodel::CubedSphereParametricDistributedDiscreteModel{T}) where {T}
#   return map(get_forward_map_generator,local_views(dmodel))
# end

# function get_radius(dmodel::CubedSphereParametricDistributedDiscreteModel{T}) where {T}
#   radii =  map(get_radius,local_views(dmodel))
#   radius = zero(eltype(radii))
#   map(radii) do r
#     radius = r
#   end
#   return radius
# end

# function get_thickness(dmodel::CubedSphere2DParametricDistributedDiscreteModel{T}) where {T}
#   @notimplemented """\n
#   The model is two dimensional, get_thickness not defined.
#   """
# end

# function get_thickness(dmodel::CubedSphere3DParametricDistributedDiscreteModel{T}) where {T}
#   Ts =  map(get_thickness,local_views(dmodel))
#   thickness = zero(eltype(Ts))
#   map(Ts) do t
#     thickness = t
#   end
#   return thickness
# end





# """
# CubedSphere2DParametricDistributedDiscreteModel

# General constructor to match inputs of CubedSphere2DParametricOctreeDistributedDiscreteModel
# Note num_initial_uniform_refinements=1 as default since we want to call Gridap.Adaptivity.get_model,
# which is not defined for the coarsest model (i.e. not adapted model)
# """

# function CubedSphere2DParametricDistributedDiscreteModel(
#   ranks::AbstractArray,
#   radius::Real;
#   num_initial_uniform_refinements=1)
#   nprocs = length(ranks)


#   dmodels = generate_distributed_refined_models(ranks,nprocs,num_initial_uniform_refinements,radius)
#   dmodels[1]
# end

# """
# CubedSphere3DParametricDistributedDiscreteModel

# General constructor to match inputs of CubedSphere3DParametricOctreeDistributedDiscreteModel
# Returns not implemented error
# """

# function CubedSphere3DParametricDistributedDiscreteModel(
#   ranks::AbstractArray,
#   radius::Real,thickness::Real;
#   num_horizontal_uniform_refinements=0,
#   num_vertical_uniform_refinements=0)

#   @notimplemented """\n
#   No distributed 3D cubed sphere available, use CubedSphere3DParametricOctreeDistributedDiscreteModel
#   """
# end


# #### The distributed panel ids are extracted from the serial. This includes both
# #### owned+ghost panel_ids.
# #### The owned panel_ids are extracted by determining the owned cell
# function distributed_panel_ids(dmodel,spanel_ids::AbstractArray{Int})
#   gids = get_cell_gids(dmodel)

#   dpanel_ids = map(partition(gids)) do ids
#     lid_to_gid = local_to_global(ids)
#     return spanel_ids[lid_to_gid]
#   end

#   owned_panel_ids = map(dpanel_ids,partition(gids)) do panel_ids, cids
#     owned_cells = own_to_local(cids)
#     return panel_ids[owned_cells]
#   end

#   return dpanel_ids, owned_panel_ids
# end


# ### This is Jordi's function from GridapDistributed/test/AdaptivityTests.jl
# function DistributedAdaptivityGlue(serial_glue,parent,child)
#   glue = map(partition(get_cell_gids(parent)),partition(get_cell_gids(child))) do parent_gids, child_gids
#     old_g2l = global_to_local(parent_gids)
#     old_l2g = local_to_global(parent_gids)
#     new_l2g = local_to_global(child_gids)

#     n2o_cell_map  = lazy_map(Reindex(old_g2l),serial_glue.n2o_faces_map[3][new_l2g])
#     n2o_faces_map = [Int64[],Int64[],collect(n2o_cell_map)]
#     n2o_cell_to_child_id = serial_glue.n2o_cell_to_child_id[new_l2g]
#     rrules = serial_glue.refinement_rules[old_l2g]
#     Gridap.Adaptivity.AdaptivityGlue(n2o_faces_map,n2o_cell_to_child_id,rrules)
#   end
#   return glue
# end




# """
# generate_distributed_refined_models
# returns an array of refined serial models where
#   models[1] == most refined model
#   models[end] == coarsest model
# """

# function generate_distributed_refined_models(ranks,nprocs,n_ref_lvls::Int,radius::Real,coarse_s_model=false)
#   s_models  = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls,radius,coarse_s_model)
#   dmodels, dpanel_ids, owned_panel_ids = generate_distributed_refined_models(ranks,nprocs,s_models)
#   dmodels
# end

# function generate_distributed_refined_models(ranks,nprocs,s_models::Vector{<:DiscreteModel})
#   # get refined models in serial
#   spanel_ids = map(m->get_panel_ids(m),s_models)
#   s_model_coarse = s_models[end]
#   radii = map(m->get_radius(m),s_models)

#   # extract the models and glues in arrays
#   models = map(m->Adaptivity.get_model(m),s_models[1:end-1])
#   glues = map(m->get_adaptivity_glue(m),s_models[1:end-1])
#   if typeof(s_model_coarse) <: AdaptedDiscreteModel
#     push!(models,Adaptivity.get_model(s_model_coarse))
#   else
#     push!(models,s_model_coarse)
#   end

#   # partition the processors
#   part_to_cells = [PartitionedArrays.local_range(rank,nprocs,num_cells(s_model_coarse)) for rank in 1:nprocs]

#   # store the partition for each model
#   cell_to_part = Vector{Vector{Int32}}(undef,length(models))

#   # get the coarse partition
#   coarse_cell_to_part = zeros(Int32,num_cells(s_model_coarse))
#   for (rank, cells) in enumerate(part_to_cells)
#     coarse_cell_to_part[cells] .= rank
#   end
#   cell_to_part[end] = coarse_cell_to_part

#   # get the refine partition based on glue
#   for level in length(models)-1:-1:1
#     n2o_cells = glues[level].n2o_faces_map[3]
#     cell_to_part[level] = cell_to_part[level+1][n2o_cells]
#   end

#   # construct array of distributed models, distributed panel ids (all + owned only)
#   dmodels = Vector{CubedSphere2DParametricDistributedDiscreteModel}(undef,length(models))
#   dpanel_ids = Vector{AbstractArray{Vector{Int}}}(undef,length(models))
#   owned_panel_ids = Vector{AbstractArray{Vector{Int}}}(undef,length(models))

#   # the coarsest model is the last in the list
#   coarse_dmodel = DiscreteModel(ranks,models[end],cell_to_part[end])
#   dpanel_ids[end],owned_panel_ids[end] = distributed_panel_ids(coarse_dmodel,spanel_ids[end])
#   dmodels[end] = CubedSphere2DParametricDistributedDiscreteModel(coarse_dmodel,dpanel_ids[end],radii[end])

#   # loop backwards through refinement levels
#   for level in length(models)-1:-1:1
#     child = DiscreteModel(ranks,models[level],cell_to_part[level])
#     parent = dmodels[level+1]
#     glue = DistributedAdaptivityGlue(glues[level],parent,child)
#     dpanel_ids[level],owned_panel_ids[level] = distributed_panel_ids(child,spanel_ids[level])

#     dmodels[level] = CubedSphere2DParametricDistributedDiscreteModel(child, dpanel_ids[level],radii[level])
#   end
#   return dmodels, dpanel_ids, owned_panel_ids
# end
