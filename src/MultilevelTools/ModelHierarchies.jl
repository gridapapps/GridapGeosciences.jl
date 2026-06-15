function GridapSolvers.MultilevelTools.ModelHierarchy(
    coarse_model::CubedSphere2DParametricOctreeDistributedDiscreteModel,
    n_ref_lvls::Int)
  println("gmg lvls = $(n_ref_lvls)")

  ranks = get_parts(coarse_model.parametric_dmodel)

  models = Vector{GridapDistributed.DistributedDiscreteModel}(undef,n_ref_lvls+1)
  # glues = Vector{MPIArray}(undef,n_ref_lvls)

  models[n_ref_lvls+1] = coarse_model.parametric_dmodel

  model = coarse_model
  for n in n_ref_lvls:-1:1
    model, glue = adapt_model(ranks,model)
    models[n] = model.parametric_dmodel
    # glues[n] = glue
  end

  return GridapSolvers.MultilevelTools.ModelHierarchy(models)
end

function GridapSolvers.MultilevelTools.ModelHierarchy(
  coarse_model::AtlasDiscreteModel,
  n_ref_lvls::Int)
  println("gmg lvls = $(n_ref_lvls)")

  models = Vector{DiscreteModel}(undef,n_ref_lvls+1)
  models[n_ref_lvls+1] = coarse_model

  model = coarse_model
  for n in n_ref_lvls:-1:1
    model = Gridap.Adaptivity.refine(model)
    models[n] = model
  end
  return GridapSolvers.MultilevelTools.ModelHierarchy(models)
end

# function GridapSolvers.MultilevelTools.ModelHierarchy(models::Vector{<:GridapDistributed.DistributedDiscreteModel},glues::Vector{<:MPIArray})
#   println("my model hierarhcy")
#   nlevs = length(models)

#   ranks = get_parts(models[1])
#   @check all(m -> length(get_parts(m)) === length(ranks), models) "Models have different communicators."

#   level_parts = fill(ranks,nlevs)
#   meshes = Vector{GridapSolvers.MultilevelTools.ModelHierarchyLevel}(undef,nlevs)
#   for lev in 1:nlevs-1
#     glue  = glues[lev]
#     model = models[lev]
#     meshes[lev] = GridapSolvers.MultilevelTools.ModelHierarchyLevel(lev,model,glue,nothing,nothing)
#   end
#   meshes[nlevs] = GridapSolvers.MultilevelTools.ModelHierarchyLevel(nlevs,models[nlevs],nothing,nothing,nothing)
#   return GridapSolvers.MultilevelTools.HierarchicalArray(meshes,level_parts)
# end

function adapt_model(ranks,
  model::CubedSphere2DParametricOctreeDistributedDiscreteModel)
  cell_partition=get_cell_gids(model.octree_dmodel)
  ref_flags=map(ranks,partition(cell_partition)) do rank,indices
      flags=zeros(Cint,length(indices))
      flags.=refine_flag
  end
  ref_model, adaptivity_glue = Gridap.Adaptivity.adapt(model,ref_flags)
  return ref_model, adaptivity_glue
end

function adapt_model(ranks,
  model::CubedSphere3DParametricOctreeDistributedDiscreteModel)
  model_vertically_refined = vertically_uniformly_refine(model)
  model_uniformly_refined = horizontally_uniformly_refine(model_vertically_refined)

  return model_uniformly_refined, false
end


function adapt_model(ranks,
  model::CubedSphereAmbientOctreeDistributedDiscreteModel)

  pmodel = get_parametric_model(model)

  ref_pmodel, = adapt_model(ranks,pmodel)
  ambient_dmodel = CubedSphereAmbientDistributedDiscreteModel(ref_pmodel.parametric_dmodel)
  ref_model = CubedSphereAmbientOctreeDistributedDiscreteModel(ref_pmodel.octree_dmodel,ambient_dmodel)

  return ref_model, false

end
