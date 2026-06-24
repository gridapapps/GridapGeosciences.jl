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

function GridapSolvers.MultilevelTools.ModelHierarchy(
    coarse_model::AtlasOctreeDistributedDiscreteModel,
    n_ref_lvls::Int)
  println("gmg lvls = $(n_ref_lvls)")

  models = Vector{GridapDistributed.DistributedDiscreteModel}(undef, n_ref_lvls+1)
  models[n_ref_lvls+1] = get_atlas_model(coarse_model)

  model = coarse_model
  for n in n_ref_lvls:-1:1
    model, _ = Gridap.Adaptivity.refine(model)
    models[n] = get_atlas_model(model)
  end
  return GridapSolvers.MultilevelTools.ModelHierarchy(models)
end

function adapt_model(ranks, model::AtlasDistributedDiscreteModel)
  Gridap.Adaptivity.refine(model), nothing
end

function adapt_model(ranks, model::AtlasOctreeDistributedDiscreteModel)
  Gridap.Adaptivity.refine(model)
end

function adapt_model(ranks, model::ExtrudedAtlasOctreeDistributedDiscreteModel)
  horizontally_uniformly_refine(vertically_uniformly_refine(model)), nothing
end
