const ref_panel_coordinates = [ -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0 ]

function set_panel_vertices_coordinates!( pconn :: Ptr{P4est_wrapper.p4est_connectivity_t},
                                             coarse_discrete_model :: DiscreteModel{2,2},
                                             panel )
  @assert num_cells(coarse_discrete_model) == 6
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
     vertices[(g-1)*3+1]=ref_panel_coordinates[(l-1)*2+1]
     vertices[(g-1)*3+2]=ref_panel_coordinates[(l-1)*2+2]
  end
end

function setup_cubed_sphere_coarse_discrete_model()
    # 6 panels (cells), 4 corners (vertices) each panel
    ptr  = [ 1, 5, 9, 13, 17, 21, 25 ]
    data = [ 1,2,3,4, 2,5,4,6, 7,8,5,6, 1,3,7,8, 3,4,8,6, 1,7,2,5  ]
    cell_vertex_lids = Gridap.Arrays.Table(data,ptr)
    node_coordinates = Vector{Point{2,Float64}}(undef,8)
    for i in 1:length(node_coordinates)
      node_coordinates[i]=Point{2,Float64}(0.0,0.0)
    end

    polytope=QUAD
    scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(polytope,Gridap.ReferenceFEs.lagrangian,Float64,1)
    cell_types=collect(Fill(1,length(cell_vertex_lids)))
    cell_reffes=[scalar_reffe]
    grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                            cell_vertex_lids,
                                            cell_reffes,
                                            cell_types,
                                            Gridap.Geometry.NonOriented())
    Gridap.Geometry.UnstructuredDiscreteModel(grid)
end

function generate_ptr(ncells)
  nvertices=4
  ptr  = Vector{Int}(undef,ncells+1)
  ptr[1]=1
  for i=1:ncells
    ptr[i+1]=ptr[i]+nvertices
  end
  ptr
end

function generate_cell_coordinates_and_panels(parts,
                                   coarse_discrete_model,
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
     ptr  = generate_ptr(ncells)
     current=1
     current_cell=1
     vxy=Vector{Cdouble}(undef,Dc)
     pvxy=pointer(vxy,1)
     for itree=1:pXest_ghost.num_trees
       tree = p4est_tree_array_index(pXest.trees, itree-1)[]
       if tree.quadrants.elem_count > 0
          set_panel_vertices_coordinates!( ptr_pXest_connectivity, coarse_discrete_model, itree)
       end
       for cell=1:tree.quadrants.elem_count
          panels[current_cell]=itree
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
        set_panel_vertices_coordinates!( ptr_pXest_connectivity, coarse_discrete_model, i)
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
  end
end

function generate_cube_grid_geo(cell_coordinates_and_panels)
  function map_panel_xy_2_xyz(xy,panel)
    a,b=xy
    if panel==1
      x=Point(1.0,a,b)
    elseif panel==2
      x=Point(-a,1.0,b)
    elseif panel==3
      x=Point(-1.0,b,a)
    elseif panel==4
      x=Point(-b,-1.0,a)
    elseif panel==5
      x=Point(-b,a,1.0)
    elseif panel==6
      x=Point(-a,b,-1.0)
    end
    x
  end

  map(cell_coordinates_and_panels) do (cell_coordinates,panels)
     ptr  = generate_ptr(length(cell_coordinates))
     data = collect(1:length(cell_coordinates)*4)
     cell_vertex_lids=Gridap.Arrays.Table(data,ptr)
     node_coordinates=Vector{Point{3,Float64}}(undef,length(cell_coordinates)*4)

     cache=array_cache(cell_coordinates)
     current=1
     for cell=1:length(cell_coordinates)
        current_cell_coordinates=getindex!(cache,cell_coordinates,cell)
        for i=1:length(current_cell_coordinates)
          node_coordinates[current]=map_panel_xy_2_xyz(current_cell_coordinates[i],panels[cell])
          current=current+1
        end
     end

     polytope=QUAD
     scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(polytope,Gridap.ReferenceFEs.lagrangian,Float64,1)
     cell_types=collect(Fill(1,length(cell_vertex_lids)))
     cell_reffes=[scalar_reffe]
     grid=Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                      cell_vertex_lids,
                                      cell_reffes,
                                      cell_types,
                                      Gridap.Geometry.NonOriented())
     grid
  end
end

function generate_cube_grid_top(cell_vertex_lids_nlvertices)
  map(cell_vertex_lids_nlvertices[1],cell_vertex_lids_nlvertices[2]) do cell_vertex_lids,nlvector
     node_coordinates=Vector{Point{2,Float64}}(undef,nlvector)
     polytope=QUAD
     scalar_reffe=Gridap.ReferenceFEs.ReferenceFE(polytope,Gridap.ReferenceFEs.lagrangian,Float64,1)
     cell_types=collect(Fill(1,length(cell_vertex_lids)))
     cell_reffes=[scalar_reffe]
    #  if (part==2)
    #   println(cell_vertex_lids)
    #  end
     cell_vertex_lids_gridap=Gridap.Arrays.Table(cell_vertex_lids.data,cell_vertex_lids.ptrs)
     grid=Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                      cell_vertex_lids_gridap,
                                      cell_reffes,
                                      cell_types,
                                      Gridap.Geometry.NonOriented())
     grid
  end
end

function setup_cubed_sphere_distributed_discrete_model(ranks,
                                                       coarse_discrete_model,
                                                       ptr_pXest_connectivity,
                                                       ptr_pXest,
                                                       ptr_pXest_ghost,
                                                       ptr_pXest_lnodes;
                                                       radius=1.0)
  Dc=2
  cellindices = GridapP4est.setup_cell_prange(Val{Dc},ranks,ptr_pXest,ptr_pXest_ghost)
  cell_vertex_gids=GridapP4est.generate_cell_vertex_gids(ptr_pXest_lnodes,cellindices)
  cell_vertex_lids_nlvertices=GridapP4est.generate_cell_vertex_lids_nlvertices(cell_vertex_gids)
  cell_coordinates_and_panels=generate_cell_coordinates_and_panels(ranks,
                                             coarse_discrete_model,
                                             ptr_pXest_connectivity,
                                             ptr_pXest,
                                             ptr_pXest_ghost)
  cube_grid_geo=generate_cube_grid_geo(cell_coordinates_and_panels)
  cube_grid_top=generate_cube_grid_top(cell_vertex_lids_nlvertices)
  ddiscretemodel=
    map(cube_grid_geo,cube_grid_top) do cube_grid_geo, cube_grid_top
      cube_model_top=Gridap.Geometry.UnstructuredDiscreteModel(cube_grid_top)
      D2toD3AnalyticalMapCubedSphereDiscreteModel(cube_grid_geo, cube_model_top, radius=radius)
    end
  GridapDistributed.DistributedDiscreteModel(ddiscretemodel,cellindices)
end

function _setup_non_adaptive_cubed_sphere_discrete_model(ranks::MPIArray,
                                                         num_uniform_refinements::Int;
                                                         radius=1.0)
  Dc=2
  comm = ranks.comm
  coarse_discrete_model=setup_cubed_sphere_coarse_discrete_model()

  ptr_pXest_connectivity,
    ptr_pXest,
      ptr_pXest_ghost,
        ptr_pXest_lnodes = GridapP4est.setup_ptr_pXest_objects(Val{Dc},
                                                   comm,
                                                   coarse_discrete_model,
                                                   num_uniform_refinements)

  dmodel=setup_cubed_sphere_distributed_discrete_model(ranks,
                                                       coarse_discrete_model,
                                                       ptr_pXest_connectivity,
                                                       ptr_pXest,
                                                       ptr_pXest_ghost,
                                                       ptr_pXest_lnodes;
                                                       radius=radius)
  GridapP4est.p4est_lnodes_destroy(ptr_pXest_lnodes)
  GridapP4est.p4est_ghost_destroy(ptr_pXest_ghost)
  GridapP4est.p4est_destroy(ptr_pXest)
  GridapP4est.p4est_connectivity_destroy(ptr_pXest_connectivity)
  return dmodel
end                                                          


struct D2toD3AnalyticalMapCubedSphereTriangulation{M} <: Triangulation{2,3}
  model::M
end

function Gridap.CellData.get_normal_vector(trian::D2toD3AnalyticalMapCubedSphereTriangulation)
  cell_normal = Gridap.Geometry.get_facet_normal(trian)
  Gridap.CellData.GenericCellField(cell_normal,trian,ReferenceDomain())
end

function _unit_outward_normal(v::Gridap.Fields.MultiValue{Tuple{2,3}})
  n1 = v[1,2]*v[2,3] - v[1,3]*v[2,2]
  n2 = v[1,3]*v[2,1] - v[1,1]*v[2,3]
  n3 = v[1,1]*v[2,2] - v[1,2]*v[2,1]
  n = VectorValue(n1,n2,n3)
  n/norm(n)
end

function Gridap.Geometry.get_facet_normal(trian::D2toD3AnalyticalMapCubedSphereTriangulation)
  # Get the Jacobian of the cubed sphere mesh
  map   = get_cell_map(trian)
  Jt    = lazy_map(∇,map)
  p=lazy_map(Operation(_unit_outward_normal),Jt)
  p
end

# Triangulation API

# Delegating to the underlying face Triangulation

Gridap.Geometry.get_cell_coordinates(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_coordinates(trian.model.cubed_sphere_grid_geo)

Gridap.Geometry.get_reffes(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_reffes(trian.model.cube_grid_geo)

Gridap.Geometry.get_cell_type(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_type(trian.model.cube_grid_geo)

Gridap.Geometry.get_node_coordinates(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = trian.model.cubed_sphere_node_coordinates

Gridap.Geometry.get_cell_node_ids(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_node_ids(trian.model.cube_grid_geo)

Gridap.Geometry.get_cell_map(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = trian.model.cell_map

Gridap.Geometry.get_background_model(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = trian.model

function Gridap.Geometry.get_glue(a::D2toD3AnalyticalMapCubedSphereTriangulation,D::Val{2})
  nc=num_cells(a.model.cube_model_top)
  tface_to_mface=Gridap.Fields.IdentityVector(nc)
  tface_to_mface_map=Fill(Gridap.Fields.GenericField(identity),nc)
  mface_to_tface=tface_to_mface
  Gridap.Geometry.FaceToFaceGlue(tface_to_mface,tface_to_mface_map,mface_to_tface)
end

Gridap.Geometry.get_grid(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = trian.model.cubed_sphere_grid_geo

struct D2toD3AnalyticalMapCubedSphereDiscreteModel{T,B,C,D} <: Gridap.Geometry.DiscreteModel{2,3}
  cell_map::T
  cube_model_top::B
  cube_grid_geo::C
  cubed_sphere_grid_geo::D
  function D2toD3AnalyticalMapCubedSphereDiscreteModel(
      cube_grid_geo  :: Gridap.Geometry.UnstructuredGrid{2,3},
      cube_model_top :: Gridap.Geometry.UnstructuredDiscreteModel{2,2};
      radius=1)

    mcts=Gridap.Fields.GenericField(MapCubeToSphere(radius))
    grid_geo_node_coordinates=Gridap.Geometry.get_node_coordinates(cube_grid_geo)
    cubed_sphere_node_coordinates=evaluate(mcts,grid_geo_node_coordinates)
    cubed_sphere_grid_geo=Gridap.Geometry.UnstructuredGrid(cubed_sphere_node_coordinates,
                                           Gridap.Geometry.get_cell_node_ids(cube_grid_geo),
                                           Gridap.Geometry.get_reffes(cube_grid_geo),
                                           Gridap.Geometry.get_cell_type(cube_grid_geo),
                                           Gridap.Geometry.NonOriented())
    
    m1=Fill(mcts,num_cells(cube_grid_geo))
    m2=get_cell_map(cube_grid_geo)
    m=lazy_map(∘,m1,m2)

    # Build output object
    T=typeof(m)
    B=typeof(cube_model_top)
    C=typeof(cube_grid_geo)
    D=typeof(cubed_sphere_grid_geo)
    GC.gc()
    new{T,B,C,D}(m,cube_model_top,cube_grid_geo,cubed_sphere_grid_geo)
  end
end

# IMPORTANT NOTE: this method is needed as its default definition in Gridap
#   num_point_dims(model::DiscreteModel) = num_point_dims(get_grid_topology(model))
# returns Dp=2 as the topological dimension of the model is 2
Gridap.Geometry.num_point_dims(::D2toD3AnalyticalMapCubedSphereDiscreteModel) = 3
Gridap.Geometry.get_cell_map(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = model.cell_map
Gridap.Geometry.get_grid(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = model.cubed_sphere_grid_geo
Gridap.Geometry.get_grid_topology(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid_topology(model.cube_model_top)
Gridap.Geometry.get_face_labeling(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_face_labeling(model.cube_model_top)
function Gridap.Geometry.Triangulation(a::D2toD3AnalyticalMapCubedSphereDiscreteModel)
  D2toD3AnalyticalMapCubedSphereTriangulation(a)
end

function Gridap.Geometry.Triangulation(
  ::Type{Gridap.ReferenceFEs.ReferenceFE{2}},
  model::D2toD3AnalyticalMapCubedSphereDiscreteModel,
  labels::Gridap.Geometry.FaceLabeling;tags=nothing)
  Gridap.Helpers.@notimplementedif tags!=nothing
  D2toD3AnalyticalMapCubedSphereTriangulation(model)
end 


struct ForestOfOctreesCubedSphereDiscreteModel{M<:OctreeDistributedDiscreteModel{2,3}, N<:Real} <: GridapDistributed.DistributedDiscreteModel{2,3}
    octree_model::M
    radius::N
end 

GridapDistributed.get_parts(model::ForestOfOctreesCubedSphereDiscreteModel) = model.octree_model.parts
GridapDistributed.local_views(model::ForestOfOctreesCubedSphereDiscreteModel) = GridapDistributed.local_views(model.octree_model.dmodel)
GridapDistributed.get_cell_gids(model::ForestOfOctreesCubedSphereDiscreteModel) = GridapDistributed.get_cell_gids(model.octree_model.dmodel)
GridapDistributed.get_face_gids(model::ForestOfOctreesCubedSphereDiscreteModel,dim::Integer) = GridapDistributed.get_face_gids(model.octree_model.dmodel,dim)

function ForestOfOctreesCubedSphereDiscreteModel(ranks::MPIArray{<:Integer},
                                                 num_uniform_refinements;
                                                 radius=1.0)

  
   Dc=2
   Dp=3

   comm = ranks.comm                                              

   coarse_model=setup_cubed_sphere_coarse_discrete_model()
    
   ptr_pXest_connectivity,
      ptr_pXest,
        ptr_pXest_ghost,
          ptr_pXest_lnodes = GridapP4est.setup_ptr_pXest_objects(Val{Dc},
                                                     comm,
                                                     coarse_model,
                                                     num_uniform_refinements)

    
    dmodel=setup_cubed_sphere_distributed_discrete_model(ranks,
                                                         coarse_model,
                                                         ptr_pXest_connectivity,
                                                         ptr_pXest,
                                                         ptr_pXest_ghost,
                                                         ptr_pXest_lnodes;
                                                         radius=radius)

    GridapP4est.pXest_lnodes_destroy(Val{Dc},ptr_pXest_lnodes)
    GridapP4est.pXest_ghost_destroy(Val{Dc},ptr_pXest_ghost)

    non_conforming_glue = GridapP4est._create_conforming_model_non_conforming_glue(dmodel)

    octree_dmodel=OctreeDistributedDiscreteModel(Dc,
                                          Dp,
                                          ranks,
                                          dmodel,
                                          non_conforming_glue,
                                          coarse_model,
                                          ptr_pXest_connectivity,
                                          ptr_pXest,
                                          true,
                                          nothing)
    ForestOfOctreesCubedSphereDiscreteModel(octree_dmodel,radius)
end

function Gridap.Adaptivity.adapt(model::ForestOfOctreesCubedSphereDiscreteModel, 
                                 refinement_and_coarsening_flags::MPIArray{<:Vector})

  Dc=2
  Dp=3

  ranks=model.octree_model.parts

  ptr_new_pXest = 
     GridapP4est._refine_coarsen_balance!(model.octree_model, 
                                          refinement_and_coarsening_flags)

  # Extract ghost and lnodes
  ptr_pXest_ghost  = GridapP4est.setup_pXest_ghost(Val{Dc}, ptr_new_pXest)
  ptr_pXest_lnodes = GridapP4est.setup_pXest_lnodes_nonconforming(Val{Dc}, ptr_new_pXest, ptr_pXest_ghost)
  ptr_pXest_connectivity = model.octree_model.ptr_pXest_connectivity
  coarse_model = model.octree_model.coarse_model

  cell_prange = GridapP4est.setup_cell_prange(Val{Dc}, ranks, ptr_new_pXest, ptr_pXest_ghost)

  gridap_cell_faces,
    non_conforming_glue=
       GridapP4est.generate_cell_faces_and_non_conforming_glue(Val{Dc},ptr_pXest_lnodes, cell_prange)

  GridapP4est.pXest_lnodes_destroy(Val{Dc},ptr_pXest_lnodes)

  nlvertices = map(non_conforming_glue) do ncglue
    ncglue.num_regular_faces[1]+ncglue.num_hanging_faces[1]
  end

  node_coordinates=GridapP4est.generate_node_coordinates(Val{Dc},
                                             gridap_cell_faces[1],
                                             nlvertices,
                                             ptr_pXest_connectivity,
                                             ptr_new_pXest,
                                             ptr_pXest_ghost)
  
  grid,topology=GridapP4est.generate_grid_and_topology(Val{Dc},
                                                       gridap_cell_faces[1],
                                                       nlvertices,
                                                       node_coordinates)

  map(topology,gridap_cell_faces[Dc]) do topology,cell_faces
    cell_faces_gridap = Gridap.Arrays.Table(cell_faces.data,cell_faces.ptrs)
    topology.n_m_to_nface_to_mfaces[Dc+1,Dc] = cell_faces_gridap
    topology.n_m_to_nface_to_mfaces[Dc,Dc+1] = Gridap.Geometry.generate_cells_around(cell_faces_gridap)
  end

  face_labeling=GridapP4est.generate_face_labeling(ranks,
                                       cell_prange,
                                       model.octree_model.coarse_model,
                                       topology,
                                       ptr_new_pXest,
                                       ptr_pXest_ghost)

  GridapP4est._set_hanging_labels!(face_labeling,non_conforming_glue)

  cell_coordinates_and_panels=generate_cell_coordinates_and_panels(ranks,
                                              coarse_model,
                                              ptr_pXest_connectivity,
                                              ptr_new_pXest,
                                              ptr_pXest_ghost)

  GridapP4est.pXest_ghost_destroy(Val{Dc},ptr_pXest_ghost)

  cube_grid_geo=generate_cube_grid_geo(cell_coordinates_and_panels)

  # # do_on_parts(comm, cell_coordinates) do part, cell_coords
  # #   if part==1
  # #     println(cell_coords[9:12])
  # #   end
  # # end

  ddiscretemodel=
  map(cube_grid_geo,grid,topology,face_labeling) do cube_grid_geo, grid, topology, face_labeling
    cube_model_top=Gridap.Geometry.UnstructuredDiscreteModel(grid,topology,face_labeling)
    D2toD3AnalyticalMapCubedSphereDiscreteModel(cube_grid_geo, cube_model_top, radius=model.radius)
  end
  fmodel=GridapDistributed.DistributedDiscreteModel(ddiscretemodel,cell_prange)

  adaptivity_glue = GridapP4est._compute_fine_to_coarse_model_glue(ranks,
                                                  model.octree_model.dmodel,
                                                  fmodel,
                                                  refinement_and_coarsening_flags)

  adaptive_models = map(local_views(model.octree_model),
                        local_views(ddiscretemodel),
                        adaptivity_glue) do model, fmodel, glue 
      Gridap.Adaptivity.AdaptedDiscreteModel(fmodel,model,glue)
  end
  fmodel = GridapDistributed.GenericDistributedDiscreteModel(adaptive_models,get_cell_gids(fmodel))

  ref_model = OctreeDistributedDiscreteModel(Dc,Dp,
                                            ranks,
                                            fmodel,
                                            non_conforming_glue,
                                            coarse_model,
                                            ptr_pXest_connectivity,
                                            ptr_new_pXest,
                                            false,
                                            model.octree_model)

  ForestOfOctreesCubedSphereDiscreteModel(ref_model,model.radius), adaptivity_glue
end 

function CubedSphereDiscreteModel(
  ranks::MPIArray,
  num_uniform_refinements::Int;
  radius=1.0,
  adaptive=false)

  if (!adaptive)
    _setup_non_adaptive_cubed_sphere_discrete_model(ranks,num_uniform_refinements,radius=radius)
  else 
    ForestOfOctreesCubedSphereDiscreteModel(ranks,num_uniform_refinements,radius=radius)
  end  
end


