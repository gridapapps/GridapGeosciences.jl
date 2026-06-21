

# Ideas:
# * GenericDistributedDiscreteModel should be created out of OctreeDistributedDiscreteModel
# * Many methods of CubedSphere2DParametricOctreeDistributedDiscreteModel should be forwarded to the underlying
#   GenericDistributedDiscreteModel
# * At some point, we should create the 3D coordinates of the nodes in the cube surface
# * To this end, we can use a cell-wise, DG-like array, to be transformed into a nodal-like array,
#   generated out of the composition of two maps:
#    - map from the reference cell to the local coordinate system of the panel [-1,1]^2 or [0,1]^2
#    - map from the local coordinate system of the panel to the 3D cube surface



"""
   CubedSphere2DParametricOctreeDistributedDiscreteModel{2,2}
   octree_dmodel manages the topology of the parametric mesh
   parametric_dmodel includes a distributed array of proc-local CubedSphereParametricDiscreteModel{2,2} objects
"""

struct CubedSphere2DParametricOctreeDistributedDiscreteModel{A<:OctreeDistributedDiscreteModel{2,2},
                                               B<:GenericDistributedDiscreteModel{2,2}} <: DistributedDiscreteModel{2,2}

  octree_dmodel::A
  parametric_dmodel::B
end

function get_radius(dmodel::CubedSphere2DParametricOctreeDistributedDiscreteModel)
  return get_radius(dmodel.parametric_dmodel)
end


# TO-THINK: not sure if radius of the cubed sphere is required?
# Right now, in the serial version of the code, we are passing the length of the cube edges
# (see coarse_cube_surface_3D function)
function CubedSphere2DParametricOctreeDistributedDiscreteModel(
      ranks::AbstractArray,
      radius::Real;
      num_initial_uniform_refinements=0)
   coarse_model = _create_parametric_octree_dmodel_coarse_model()
   octree_dmodel, cell_wise_vertex_alpha_beta_coordinates, cell_panels =
           _generate_octree_dmodel_alpha_beta_coordinates_and_panels(ranks,
                                                                   coarse_model,
                                                                   num_initial_uniform_refinements,
                                                                   setup_coarse_cell_vertices_alpha_beta_coordinates(),
                                                                   collect(1:NPANELS))

   # Build the proc-local ParametricDiscreteModels
   parametric_models = _setup_parametric_models(octree_dmodel,
                                                cell_wise_vertex_alpha_beta_coordinates,
                                                cell_panels,
                                                radius)

   # Build the GenericDistributedDiscreteModel
   generic_dmodel = GenericDistributedDiscreteModel(parametric_models, get_cell_gids(octree_dmodel.dmodel))
   CubedSphere2DParametricOctreeDistributedDiscreteModel(octree_dmodel, generic_dmodel)
end

function _setup_parametric_models(octree_dmodel::OctreeDistributedDiscreteModel{2,2},
                                  cell_wise_vertex_alpha_beta_coordinates,
                                  cell_panels,
                                  radius::Real)

   map(local_views(octree_dmodel.dmodel),
       cell_wise_vertex_alpha_beta_coordinates,
       cell_panels) do omodel, cell_wise_vertex_alpha_beta_coordinates, cell_panels

       alpha_beta_cmap = setup_alpha_beta_cell_map(cell_wise_vertex_alpha_beta_coordinates)

       ogrid = get_grid(omodel)
       otopo = get_grid_topology(omodel)
       olabels = Gridap.Geometry.get_face_labeling(omodel)
       panel_grid = UnstructuredGrid(get_node_coordinates(ogrid),
                                     get_cell_node_ids(ogrid),
                                     get_reffes(ogrid),
                                     get_cell_type(ogrid),
                                     OrientationStyle(ogrid),
                                     nothing,
                                     alpha_beta_cmap)

        CubedSphere2DParametricDiscreteModel(panel_grid,
                                otopo,
                                olabels,
                                cell_panels,
                                radius)
   end
end

function _create_parametric_octree_dmodel_coarse_model()
  # 6 panels (cells), 4 corners (vertices) each panel
  cell_node_ids = _CCAM_panel_wise_node_ids()
  node_coordinates = Vector{Point{2,Float64}}(undef,8)
  # These coordinates are junk coordinates, but they have to be unique.
  # This a precondition for the OctreeDistributedDiscreteModel to properly
  # count the vertices of the coarse model
  for i in 1:length(node_coordinates)
    node_coordinates[i]=Point{2,Float64}(Float64(i),Float64(i))
  end
  polytope=QUAD
  scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(polytope,Gridap.ReferenceFEs.lagrangian,Float64,1)
  cell_types=collect(Fill(1,length(cell_node_ids)))
  cell_reffes=[scalar_reffe]
  grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                          cell_node_ids,
                                          cell_reffes,
                                          cell_types,
                                          Gridap.Geometry.NonOriented())
  m=Gridap.Geometry.UnstructuredDiscreteModel(grid)

  # Assign all vertices, edges, and cells of the coarse model to the
  # same entity. A single tag "boundary" is associated to this entity.
  # The name of such tag is not relevant. What it is important is that
  # all faces are associated with a positive entity identifier. This is
  # a precondition for the discrete model grounded on p6est
  face_labeling=Gridap.Geometry.get_face_labeling(m)
  for i=1:length(face_labeling.d_to_dface_to_entity)
    face_labeling.d_to_dface_to_entity[i].=1
  end
  add_tag!(face_labeling, "boundary", [1])

  return m
end

 # This info goes to P4est ...
 function setup_coarse_cell_vertices_alpha_beta_coordinates()
    pts = CUBE_HALF_EDGE.*[Point(-1.0,-1.0),Point(1.0,-1.0),Point(-1.0,1.0),Point(1.0,1.0)]
    _data = fill(pts,NPANELS)
    data = vcat( _data...  )
    ptr = [1, 5, 9, 13, 17, 21, 25]
    Gridap.Arrays.Table(data,ptr)
 end

function setup_alpha_beta_cell_map(cell_vertices_alpha_beta)
  scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(QUAD,Gridap.ReferenceFEs.lagrangian,Float64,1)
  cell_shape_funs = FillArrays.Fill( Gridap.ReferenceFEs.get_shapefuns(scalar_reffe), length(cell_vertices_alpha_beta))
  lazy_map(linear_combination,cell_vertices_alpha_beta,cell_shape_funs)
end

function set_coarse_cell_vertices_coordinates!( pconn :: Ptr{P4est_wrapper.p4est_connectivity_t},
                                                coarse_discrete_model :: DiscreteModel{2,2},
                                                panel,
                                                ref_cell_coordinates)
  @assert panel ≤ num_cells(coarse_discrete_model)
  @assert panel ≥ 1
  trian=Triangulation(coarse_discrete_model)
  cell_vertices=Gridap.Geometry.get_cell_node_ids(trian)
  #println(cell_vertices)
  cell_vertices_panel=cell_vertices[panel]
  conn=pconn[]
  vertices=unsafe_wrap(Array,
                       conn.vertices,
                       length(Gridap.Geometry.get_node_coordinates(coarse_discrete_model))*3)
  for (l,g) in enumerate(cell_vertices_panel)
     vertices[(g-1)*3+1]=ref_cell_coordinates[l][1]
     vertices[(g-1)*3+2]=ref_cell_coordinates[l][2]
  end
end

function generate_cell_coordinates_and_panels(parts,
                                   coarse_discrete_model,
                                   coarse_cell_wise_vertex_coordinates,
                                   coarse_cell_panel,
                                   ptr_pXest_connectivity,
                                   ptr_pXest,
                                   ptr_pXest_ghost)

  Dc=2
  PXEST_CORNERS=4
  pXest_ghost = ptr_pXest_ghost[]
  pXest = ptr_pXest[]

  # Obtain ghost quadrants
  ptr_ghost_quadrants = Ptr{P4est_wrapper.p4est_quadrant_t}(pXest_ghost.ghosts.array)

  tree_offsets = unsafe_wrap(Array, pXest_ghost.tree_offsets, pXest_ghost.num_trees+1)
  dcell_coordinates_and_panels=map(parts) do part
     ncells=pXest.local_num_quadrants+pXest_ghost.ghosts.elem_count
     panels = Vector{Int}(undef,ncells)
     data = Vector{Point{Dc,Float64}}(undef,ncells*PXEST_CORNERS)
     ptr  = generate_ptr(Dc,ncells)
     current=1
     current_cell=1
     vxy=Vector{Cdouble}(undef,Dc)
     pvxy=pointer(vxy,1)
     for itree=1:pXest_ghost.num_trees
       tree = p4est_tree_array_index(pXest.trees, itree-1)[]
       if tree.quadrants.elem_count > 0
          set_coarse_cell_vertices_coordinates!( ptr_pXest_connectivity,
                                                 coarse_discrete_model,
                                                 itree,
                                                 coarse_cell_wise_vertex_coordinates[itree])
       end
       for cell=1:tree.quadrants.elem_count
          panels[current_cell]=coarse_cell_panel[itree]
          quadrant=p4est_quadrant_array_index(tree.quadrants, cell-1)[]
          for vertex=1:PXEST_CORNERS
            GridapP4est.p4est_get_quadrant_vertex_coordinates(ptr_pXest_connectivity,
                                                  p4est_topidx_t(itree-1),
                                                  quadrant.x,
                                                  quadrant.y,
                                                  quadrant.level,
                                                  Cint(vertex-1),
                                                  pvxy)
            data[current]=Point{Dc,Float64}(vxy...)
            current=current+1
          end
          current_cell=current_cell+1
       end
     end

     # Go over ghost cells
     for i=1:pXest_ghost.num_trees
      if tree_offsets[i+1]-tree_offsets[i] > 0
        set_coarse_cell_vertices_coordinates!( ptr_pXest_connectivity,
                                               coarse_discrete_model,
                                               i,
                                               coarse_cell_wise_vertex_coordinates[i])
      end
      for j=tree_offsets[i]:tree_offsets[i+1]-1
          panels[current_cell]=i
          quadrant = ptr_ghost_quadrants[j+1]
          for vertex=1:PXEST_CORNERS
            GridapP4est.p4est_get_quadrant_vertex_coordinates(ptr_pXest_connectivity,
                                                     p4est_topidx_t(i-1),
                                                     quadrant.x,
                                                     quadrant.y,
                                                     quadrant.level,
                                                     Cint(vertex-1),
                                                     pvxy)

          #  if (MPI.Comm_rank(comm.comm)==0)
          #     println(vxy)
          #  end
          data[current]=Point{Dc,Float64}(vxy...)
          current=current+1
         end
         current_cell=current_cell+1
       end
     end
     Gridap.Arrays.Table(data,ptr), panels
  end |> tuple_of_arrays
end

function _generate_octree_dmodel_alpha_beta_coordinates_and_panels(ranks,
                                                                   coarse_model::DiscreteModel{2,2},
                                                                   num_uniform_refinements,
                                                                   coarse_cell_wise_vertex_coordinates,
                                                                   coarse_cell_panel)
   comm = ranks.comm
   pXest_type = GridapP4est._dim_to_pXest_type(2)
   pXest_refinement_rule_type = GridapP4est.PXestUniformRefinementRuleType()

   ptr_pXest_connectivity=GridapP4est.setup_pXest_connectivity(coarse_model)
   # Create a new forest
   ptr_pXest = GridapP4est.setup_pXest(pXest_type,comm,ptr_pXest_connectivity,num_uniform_refinements)
   # Build the ghost layer
   ptr_pXest_ghost=GridapP4est.setup_pXest_ghost(pXest_type,ptr_pXest)

   ptr_pXest_lnodes=GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type, ptr_pXest, ptr_pXest_ghost)

   dmodel, non_conforming_glue  = GridapP4est.setup_non_conforming_distributed_discrete_model(pXest_type,
                                                    pXest_refinement_rule_type,
                                                    ranks,
                                                    coarse_model,
                                                    ptr_pXest_connectivity,
                                                    ptr_pXest,
                                                    ptr_pXest_ghost,
                                                    ptr_pXest_lnodes;
                                                    grid_and_topology_function=dummy_grid_and_topology_function)


   cell_coordinates, panels=generate_cell_coordinates_and_panels(ranks,
                                             coarse_model,
                                             coarse_cell_wise_vertex_coordinates,
                                             coarse_cell_panel,
                                             ptr_pXest_connectivity,
                                             ptr_pXest,
                                             ptr_pXest_ghost)

   GridapP4est.pXest_lnodes_destroy(pXest_type,ptr_pXest_lnodes)
   GridapP4est.pXest_ghost_destroy(pXest_type,ptr_pXest_ghost)

   omodel=OctreeDistributedDiscreteModel(2,
                                  2,
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


   omodel, cell_coordinates, panels
end

function _adapt_octree_dmodel(octree_dmodel::OctreeDistributedDiscreteModel,
                              coarse_cell_wise_vertex_coordinates,
                              coarse_cell_panel,
                              refinement_and_coarsening_flags::MPIArray{<:Vector})
  Dc=2
  pXest_type=octree_dmodel.pXest_type
  pXest_refinement_rule_type=octree_dmodel.pXest_refinement_rule_type

  ranks=octree_dmodel.parts

  ptr_new_pXest =
     GridapP4est._refine_coarsen_balance!(octree_dmodel,
                                          refinement_and_coarsening_flags)

  # Extract ghost and lnodes
  ptr_pXest_ghost  = GridapP4est.setup_pXest_ghost(pXest_type, ptr_new_pXest)
  ptr_pXest_lnodes = GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type, ptr_new_pXest, ptr_pXest_ghost)
  ptr_pXest_connectivity = octree_dmodel.ptr_pXest_connectivity
  coarse_model = octree_dmodel.coarse_model

  new_dmodel,non_conforming_glue  = GridapP4est.setup_non_conforming_distributed_discrete_model(pXest_type,
                                                    pXest_refinement_rule_type,
                                                    ranks,
                                                    coarse_model,
                                                    ptr_pXest_connectivity,
                                                    ptr_new_pXest,
                                                    ptr_pXest_ghost,
                                                    ptr_pXest_lnodes;
                                                    grid_and_topology_function=dummy_grid_and_topology_function)

  cell_coordinates, panels=generate_cell_coordinates_and_panels(ranks,
                                              coarse_model,
                                              coarse_cell_wise_vertex_coordinates,
                                              coarse_cell_panel,
                                              ptr_pXest_connectivity,
                                              ptr_new_pXest,
                                              ptr_pXest_ghost)

  GridapP4est.pXest_ghost_destroy(pXest_type,ptr_pXest_ghost)
  GridapP4est.pXest_lnodes_destroy(pXest_type,ptr_pXest_lnodes)

  stride = GridapP4est.pXest_stride_among_children(pXest_type,
                                                   pXest_refinement_rule_type,
                                                   octree_dmodel.ptr_pXest)

  adaptivity_glue = GridapP4est._compute_fine_to_coarse_model_glue(
                                                  pXest_type,
                                                  pXest_refinement_rule_type,
                                                  ranks,
                                                  octree_dmodel.dmodel,
                                                  new_dmodel,
                                                  refinement_and_coarsening_flags,
                                                  stride)


  adapted_models = map(local_views(octree_dmodel.dmodel), local_views(new_dmodel), adaptivity_glue) do parent_model,
                                                                                                       new_model,
                                                                                                       glue
      parent = isa(parent_model,AdaptedDiscreteModel) ? parent_model.model : parent_model
      Gridap.Adaptivity.AdaptedDiscreteModel(new_model, parent, glue)
  end
  adapted_dmodel=GridapDistributed.DistributedDiscreteModel(adapted_models,get_cell_gids(new_dmodel))

  adapted_omodel=OctreeDistributedDiscreteModel(2,
                                  2,
                                  ranks,
                                  adapted_dmodel,
                                  non_conforming_glue,
                                  coarse_model,
                                  octree_dmodel.ptr_pXest_connectivity,
                                  ptr_new_pXest,
                                  octree_dmodel.pXest_type,
                                  GridapP4est.PXestUniformRefinementRuleType(),
                                  false,
                                  octree_dmodel)
  adapted_omodel, cell_coordinates, panels, adaptivity_glue
end

function Gridap.Adaptivity.adapt(model::CubedSphere2DParametricOctreeDistributedDiscreteModel,
                                 refinement_and_coarsening_flags::MPIArray{<:Vector})

   coarse_cell_vertices_alpha_beta = setup_coarse_cell_vertices_alpha_beta_coordinates()
   coarse_cell_panels_2D = collect(1:NPANELS)

   adapted_octree_dmodel, cell_wise_vertex_alpha_beta, cell_panels, adaptivity_glue =
      _adapt_octree_dmodel(model.octree_dmodel,
                           coarse_cell_vertices_alpha_beta,
                           coarse_cell_panels_2D,
                           refinement_and_coarsening_flags)


   # Build the proc-local ParametricDiscreteModels
   parametric_models = _setup_parametric_models(adapted_octree_dmodel,
                                                cell_wise_vertex_alpha_beta,
                                                cell_panels,
                                                get_radius(model))

   adaptive_models = map(parametric_models,
                         local_views(model.parametric_dmodel),
                         local_views(adapted_octree_dmodel.dmodel)) do parametric_model,
                                                                       parametric_model_parent,
                                                                       octree_dmodel_adapted_model
      parent = isa(parametric_model_parent,AdaptedDiscreteModel) ? parametric_model_parent.model : parametric_model_parent
      Gridap.Adaptivity.AdaptedDiscreteModel(parametric_model,
                                             parent,
                                             get_adaptivity_glue(octree_dmodel_adapted_model))
   end
   generic_dmodel = GenericDistributedDiscreteModel(adaptive_models, get_cell_gids(adapted_octree_dmodel.dmodel))
   CubedSphere2DParametricOctreeDistributedDiscreteModel(adapted_octree_dmodel, generic_dmodel), adaptivity_glue
end



"""
get_octree_refined_models

returns array of dmodels that originate from an omodel
"""
function get_octree_refined_models(ranks,n_ref_lvls::Int,radius::Real,coarse_model=false)
  dmodels = Vector{CubedSphere2DParametricDistributedDiscreteModel}(undef,n_ref_lvls)

  for (i,n) in enumerate(n_ref_lvls:-1:1)
    parametric_octree_dmodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius;
                                                                        num_initial_uniform_refinements=n)
    dmodels[i] = parametric_octree_dmodel.parametric_dmodel
  end

  if coarse_model
    parametric_octree_dmodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius;
                                                                        num_initial_uniform_refinements=0)
    push!(dmodels,parametric_octree_dmodel.parametric_dmodel)
  end

  return dmodels
end
