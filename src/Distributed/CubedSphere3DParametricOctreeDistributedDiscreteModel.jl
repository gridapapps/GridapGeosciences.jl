struct CubedSphere3DParametricOctreeDistributedDiscreteModel{A<:OctreeDistributedDiscreteModel{3,3},
                                                  B<:GenericDistributedDiscreteModel{3,3}} <: DistributedDiscreteModel{3,3}

  octree_dmodel::A
  parametric_dmodel::B
end

function get_radius(dmodel::CubedSphere3DParametricOctreeDistributedDiscreteModel)
  return get_radius(dmodel.parametric_dmodel)
end

function get_thickness(dmodel::CubedSphere3DParametricOctreeDistributedDiscreteModel)
  return get_thickness(dmodel.parametric_dmodel)
end

function CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks::AbstractArray,
                                                    radius::Real,thickness::Real;
                                                    num_horizontal_uniform_refinements=0,
                                                    num_vertical_uniform_refinements=0)

    msg = """\n
    For performance reasons, radius and thickness variables must be of the same type.
    Currently the type of radius is $(typeof(radius)), while the type of thickness is
    $(typeof(thickness)).
      """

    @assert typeof(radius) == typeof(thickness) msg

    coarse_model = _create_parametric_octree_dmodel_coarse_model()

    octree_dmodel, cell_wise_vertex_alpha_beta_gamma_coordinates, cell_panels =
            _generate_octree_alpha_beta_gamma_coordinates_and_panels(ranks,
                                                                coarse_model,
                                                                num_horizontal_uniform_refinements,
                                                                num_vertical_uniform_refinements,
                                                                setup_coarse_cell_vertices_alpha_beta_coordinates(),
                                                                collect(1:NPANELS))

    # Build the proc-local ParametricDiscreteModels
    parametric_models = _setup_parametric_models(octree_dmodel,
                                                 cell_wise_vertex_alpha_beta_gamma_coordinates,
                                                 cell_panels,
                                                 radius,
                                                 thickness)

    # Build the GenericDistributedDiscreteModel
    generic_dmodel = GenericDistributedDiscreteModel(parametric_models, get_cell_gids(octree_dmodel.dmodel))
    CubedSphere3DParametricOctreeDistributedDiscreteModel(octree_dmodel, generic_dmodel)
end

function _setup_parametric_models(octree_dmodel::OctreeDistributedDiscreteModel{3,3},
                                 cell_wise_vertex_alpha_beta_gamma_coordinates,
                                 cell_panels,
                                 radius,
                                 thickness)

    map(local_views(octree_dmodel.dmodel),
                    cell_wise_vertex_alpha_beta_gamma_coordinates,
                    cell_panels) do omodel, cell_wise_vertex_alpha_beta_gamma_coordinates, cell_panels

        alpha_beta_gamma_cmap = setup_alpha_beta_gamma_cell_map(cell_wise_vertex_alpha_beta_gamma_coordinates)

        ogrid = get_grid(omodel)
        otopo = get_grid_topology(omodel)
        olabels = Gridap.Geometry.get_face_labeling(omodel)
        panel_grid = UnstructuredGrid(get_node_coordinates(ogrid),
                                     get_cell_node_ids(ogrid),
                                     get_reffes(ogrid),
                                     get_cell_type(ogrid),
                                     OrientationStyle(ogrid),
                                     nothing,
                                     alpha_beta_gamma_cmap)
        CubedSphere3DParametricDiscreteModel(panel_grid,
                                otopo,
                                olabels,
                                cell_panels,
                                radius,
                                thickness)
    end
end



function setup_alpha_beta_gamma_cell_map(cell_vertices_alpha_beta_gamma)
  scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(HEX,Gridap.ReferenceFEs.lagrangian,Float64,1)
  cell_shape_funs =
     FillArrays.Fill( Gridap.ReferenceFEs.get_shapefuns(scalar_reffe), length(cell_vertices_alpha_beta_gamma) )
  lazy_map(linear_combination,cell_vertices_alpha_beta_gamma,cell_shape_funs)
end

function _generate_octree_alpha_beta_gamma_coordinates_and_panels(ranks,
                                                                  coarse_model::DiscreteModel{2,2},
                                                                  num_horizontal_uniform_refinements,
                                                                  num_vertical_uniform_refinements,
                                                                  coarse_cell_wise_vertex_alpha_beta_coordinates,
                                                                  coarse_cell_panel)
   comm = ranks.comm
   Dc=3
   pXest_type = GridapP4est.P6estType()
   pXest_refinement_rule_type = GridapP4est.PXestHorizontalRefinementRuleType()

   extrusion_vector::Vector{Float64}=[0.0,0.0,1.0]

   ptr_pXest_connectivity=GridapP4est.setup_pXest_connectivity(pXest_type,
                                                   coarse_model,
                                                   extrusion_vector)

   ptr_pXest=P4est_wrapper.p6est_new_ext(comm,
                  ptr_pXest_connectivity,
                  Cint(0),
                  Cint(num_horizontal_uniform_refinements), # min_level
                  Cint(num_vertical_uniform_refinements),   # min_zlevel
                  Cint(1),                       # num_zroot
                  Cint(1),                       # fill_uniform
                  Cint(1),                       # data_size
                  C_NULL,                        # init_fn
                  C_NULL)                        # user_pointer


    ptr_pXest_ghost=GridapP4est.setup_pXest_ghost(pXest_type,ptr_pXest)
    ptr_pXest_lnodes=GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type, ptr_pXest, ptr_pXest_ghost)


    dmodel,non_conforming_glue  = GridapP4est.setup_non_conforming_distributed_discrete_model(pXest_type,
                                                    GridapP4est.PXestHorizontalRefinementRuleType(),
                                                    ranks,
                                                    coarse_model,
                                                    ptr_pXest_connectivity,
                                                    ptr_pXest,
                                                    ptr_pXest_ghost,
                                                    ptr_pXest_lnodes;
                                                    grid_and_topology_function=dummy_grid_and_topology_function,
                                                    grid_and_topology_bottom_function=dummy_grid_and_topology_function)

    cell_coordinates, panels=generate_cell_alpha_beta_gamma_coordinates_and_panels(ranks,
                                          coarse_model,
                                          setup_coarse_cell_vertices_alpha_beta_coordinates(),
                                          coarse_cell_panel,
                                          ptr_pXest_connectivity,
                                          ptr_pXest,
                                          ptr_pXest_ghost)

     omodel= GridapP4est.OctreeDistributedDiscreteModel(Dc,
                                                        Dc,
                                                        ranks,
                                                        dmodel,
                                                        non_conforming_glue,
                                                        coarse_model,
                                                        ptr_pXest_connectivity,
                                                        ptr_pXest,
                                                        pXest_type,
                                                        pXest_refinement_rule_type,
                                                        true,
                                                        nothing)

    GridapP4est.pXest_lnodes_destroy(pXest_type,ptr_pXest_lnodes)
    GridapP4est.pXest_ghost_destroy(pXest_type,ptr_pXest_ghost)

    omodel, cell_coordinates, panels
end


function vertically_uniformly_refine(parametric_model::CubedSphere3DParametricOctreeDistributedDiscreteModel)
  ptr_new_pXest = GridapP4est._vertically_uniformly_refine!(parametric_model.octree_dmodel)

  pXest_type = parametric_model.octree_dmodel.pXest_type

  # Extract ghost and lnodes
  ptr_pXest_ghost  = GridapP4est.setup_pXest_ghost(pXest_type, ptr_new_pXest)
  ptr_pXest_lnodes = GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type, ptr_new_pXest, ptr_pXest_ghost)

  pXest_refinement_rule_type = parametric_model.octree_dmodel.pXest_refinement_rule_type
  ranks = parametric_model.octree_dmodel.parts
  coarse_model = parametric_model.octree_dmodel.coarse_model
  ptr_pXest_connectivity = parametric_model.octree_dmodel.ptr_pXest_connectivity

  fmodel,non_conforming_glue  = GridapP4est.setup_non_conforming_distributed_discrete_model(pXest_type,
                                              parametric_model.octree_dmodel.pXest_refinement_rule_type,
                                              ranks,
                                              coarse_model,
                                              ptr_pXest_connectivity,
                                              ptr_new_pXest,
                                              ptr_pXest_ghost,
                                              ptr_pXest_lnodes;
                                              grid_and_topology_function=dummy_grid_and_topology_function,
                                              grid_and_topology_bottom_function=dummy_grid_and_topology_function)

  cell_coordinates, panels=generate_cell_alpha_beta_gamma_coordinates_and_panels(ranks,
                                          coarse_model,
                                          setup_coarse_cell_vertices_alpha_beta_coordinates(),
                                          collect(1:NPANELS),
                                          ptr_pXest_connectivity,
                                          ptr_new_pXest,
                                          ptr_pXest_ghost)

   GridapP4est.pXest_ghost_destroy(pXest_type,ptr_pXest_ghost)
   GridapP4est.pXest_lnodes_destroy(pXest_type,ptr_pXest_lnodes)

   pXest_refinement_rule_type = GridapP4est.PXestVerticalRefinementRuleType()
   _refinement_and_coarsening_flags = map(partition(get_cell_gids(parametric_model.octree_dmodel))) do indices
     flags  = Vector{Cint}(undef,length(local_to_global(indices)))
     flags .= refine_flag
   end

   stride = GridapP4est.pXest_stride_among_children(pXest_type,
                                                    pXest_refinement_rule_type,
                                                    parametric_model.octree_dmodel.ptr_pXest)
   adaptivity_glue = GridapP4est._compute_fine_to_coarse_model_glue(pXest_type,
                                                       pXest_refinement_rule_type,
                                                       ranks,
                                                       parametric_model.octree_dmodel.dmodel,
                                                       fmodel,
                                                       _refinement_and_coarsening_flags,
                                                       stride)
   adaptive_models = map(local_views(parametric_model.octree_dmodel),
                           local_views(fmodel),
                           adaptivity_glue) do model, fmodel, glue
           parent = isa(model,AdaptedDiscreteModel) ? model.model : model
           Gridap.Adaptivity.AdaptedDiscreteModel(fmodel,parent,glue)
   end
   fmodel = GridapDistributed.GenericDistributedDiscreteModel(adaptive_models,get_cell_gids(fmodel))
   ref_model = OctreeDistributedDiscreteModel(3,3,
                                              ranks,
                                              fmodel,
                                              non_conforming_glue,
                                              coarse_model,
                                              ptr_pXest_connectivity,
                                              ptr_new_pXest,
                                              pXest_type,
                                              parametric_model.octree_dmodel.pXest_refinement_rule_type,
                                              false,
                                              parametric_model)

   # Build the proc-local ParametricDiscreteModels
   parametric_models = _setup_parametric_models(ref_model,
                                                cell_coordinates,
                                                panels,
                                                get_radius(parametric_model),
                                                get_thickness(parametric_model))

   adaptive_models = map(parametric_models,
                         local_views(parametric_model.parametric_dmodel),
                         local_views(ref_model.dmodel)) do parametric_model,
                                                           parametric_model_parent,
                                                           octree_dmodel_adapted_model
        parent = isa(parametric_model_parent,AdaptedDiscreteModel) ? parametric_model_parent.model : parametric_model_parent
        Gridap.Adaptivity.AdaptedDiscreteModel(parametric_model,
                                               parent,
                                               get_adaptivity_glue(octree_dmodel_adapted_model))
   end
   generic_dmodel =
      GenericDistributedDiscreteModel(adaptive_models, get_cell_gids(ref_model.dmodel))

   CubedSphere3DParametricOctreeDistributedDiscreteModel(ref_model, generic_dmodel)
end

function horizontally_uniformly_refine(parametric_model::CubedSphere3DParametricOctreeDistributedDiscreteModel)

    num_cols = GridapP4est.num_locally_owned_columns(parametric_model.octree_dmodel)
    _refinement_and_coarsening_flags = map(num_cols) do num_cols
        flags  = Vector{Cint}(undef,num_cols)
        flags .= refine_flag
    end
    ptr_new_pXest = GridapP4est._horizontally_refine_coarsen_balance!(parametric_model.octree_dmodel,
                                                                      _refinement_and_coarsening_flags)


    pXest_type = parametric_model.octree_dmodel.pXest_type
    pXest_refinement_rule_type = parametric_model.octree_dmodel.pXest_refinement_rule_type
    ranks = parametric_model.octree_dmodel.parts
    coarse_model = parametric_model.octree_dmodel.coarse_model
    ptr_pXest_connectivity = parametric_model.octree_dmodel.ptr_pXest_connectivity

    # Extract ghost and lnodes
    ptr_pXest_ghost  = GridapP4est.setup_pXest_ghost(pXest_type, ptr_new_pXest)
    ptr_pXest_lnodes = GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type, ptr_new_pXest, ptr_pXest_ghost)

    fmodel,non_conforming_glue  = GridapP4est.setup_non_conforming_distributed_discrete_model(pXest_type,
                                              parametric_model.octree_dmodel.pXest_refinement_rule_type,
                                              ranks,
                                              coarse_model,
                                              ptr_pXest_connectivity,
                                              ptr_new_pXest,
                                              ptr_pXest_ghost,
                                              ptr_pXest_lnodes;
                                              grid_and_topology_function=dummy_grid_and_topology_function,
                                              grid_and_topology_bottom_function=dummy_grid_and_topology_function)

    cell_coordinates, panels=generate_cell_alpha_beta_gamma_coordinates_and_panels(ranks,
                                          coarse_model,
                                          setup_coarse_cell_vertices_alpha_beta_coordinates(),
                                          collect(1:NPANELS),
                                          ptr_pXest_connectivity,
                                          ptr_new_pXest,
                                          ptr_pXest_ghost)

    GridapP4est.pXest_ghost_destroy(pXest_type,ptr_pXest_ghost)
    GridapP4est.pXest_lnodes_destroy(pXest_type,ptr_pXest_lnodes)

    pXest_refinement_rule_type = GridapP4est.PXestHorizontalRefinementRuleType()

    extruded_ref_coarsen_flags=
       map(partition(get_cell_gids(parametric_model.octree_dmodel.dmodel)),_refinement_and_coarsening_flags) do indices, flags
      similar(flags, length(local_to_global(indices)))
    end

    GridapP4est._extrude_refinement_and_coarsening_flags!(extruded_ref_coarsen_flags,
                                              _refinement_and_coarsening_flags,
                                              parametric_model.octree_dmodel.ptr_pXest,
                                              ptr_new_pXest)

    stride = GridapP4est.pXest_stride_among_children(pXest_type,
                                         pXest_refinement_rule_type,
                                         parametric_model.octree_dmodel.ptr_pXest)

    adaptivity_glue = GridapP4est._compute_fine_to_coarse_model_glue(pXest_type,
                                                       pXest_refinement_rule_type,
                                                       parametric_model.octree_dmodel.parts,
                                                       parametric_model.octree_dmodel.dmodel,
                                                       fmodel,
                                                       extruded_ref_coarsen_flags,
                                                       stride)

    adaptive_models = map(local_views(parametric_model.octree_dmodel),
                           local_views(fmodel),
                           adaptivity_glue) do model, fmodel, glue
           parent = isa(model,AdaptedDiscreteModel) ? model.model : model
           Gridap.Adaptivity.AdaptedDiscreteModel(fmodel,parent,glue)
   end
   fmodel = GridapDistributed.GenericDistributedDiscreteModel(adaptive_models,get_cell_gids(fmodel))
   ref_model = OctreeDistributedDiscreteModel(3,3,
                                              ranks,
                                              fmodel,
                                              non_conforming_glue,
                                              coarse_model,
                                              ptr_pXest_connectivity,
                                              ptr_new_pXest,
                                              pXest_type,
                                              parametric_model.octree_dmodel.pXest_refinement_rule_type,
                                              false,
                                              parametric_model)

   # Build the proc-local ParametricDiscreteModels
   parametric_models = _setup_parametric_models(ref_model,
                                                cell_coordinates,
                                                panels,
                                                get_radius(parametric_model),
                                                get_thickness(parametric_model))

   adaptive_models = map(parametric_models,
                         local_views(parametric_model.parametric_dmodel),
                         local_views(ref_model.dmodel)) do parametric_model,
                                                           parametric_model_parent,
                                                           octree_dmodel_adapted_model
        parent = isa(parametric_model_parent,AdaptedDiscreteModel) ? parametric_model_parent.model : parametric_model_parent
        Gridap.Adaptivity.AdaptedDiscreteModel(parametric_model,
                                               parent,
                                               get_adaptivity_glue(octree_dmodel_adapted_model))
   end
   generic_dmodel =
      GenericDistributedDiscreteModel(adaptive_models, get_cell_gids(ref_model.dmodel))

   CubedSphere3DParametricOctreeDistributedDiscreteModel(ref_model, generic_dmodel)
end



"""
get_3D_octree_refined_models

returns array of dmodels that originate from an 3D omodel
"""
function get_3D_octree_refined_models(ranks,n_ref_lvls::Int,radius::Real,thickness::Real)
  dmodels = Vector{CubedSphere3DParametricDistributedDiscreteModel}(undef,n_ref_lvls)

  for (i,n) in enumerate(n_ref_lvls:-1:1)
    octree3_model = CubedSphere3DParametricOctreeDistributedDiscreteModel(ranks,radius,thickness;
                        num_horizontal_uniform_refinements=n,
                        num_vertical_uniform_refinements=n);
    dmodels[i] = octree3_model.parametric_dmodel
  end

  return dmodels
end
