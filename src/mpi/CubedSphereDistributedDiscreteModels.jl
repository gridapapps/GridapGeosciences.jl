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
     # println("XXX $(l) $(g) $(lref)")
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
  dcell_coordinates_and_panels=map_parts(parts) do part
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

  map_parts(cell_coordinates_and_panels) do (cell_coordinates,panels)
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
  map_parts(cell_vertex_lids_nlvertices) do (cell_vertex_lids,nlvector)
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


function CubedSphereDiscreteModel(
  parts::MPIData,
  num_uniform_refinements::Int;
  p4est_verbosity_level=P4est_wrapper.SC_LP_DEFAULT)

  comm = parts.comm

  sc_init(parts.comm, Cint(true), Cint(true), C_NULL, p4est_verbosity_level)
  p4est_init(C_NULL, p4est_verbosity_level)

  Dc=2

  coarse_discrete_model=setup_cubed_sphere_coarse_discrete_model()

  ptr_pXest_connectivity=GridapP4est.setup_pXest_connectivity(coarse_discrete_model)

  # Create a new forest
  ptr_pXest = GridapP4est.setup_pXest(Val{Dc},comm,ptr_pXest_connectivity,num_uniform_refinements)

  # Build the ghost layer
  ptr_pXest_ghost=GridapP4est.setup_pXest_ghost(Val{Dc},ptr_pXest)

  cellindices = GridapP4est.setup_cell_prange(Val{Dc},parts,ptr_pXest,ptr_pXest_ghost)

  ptr_pXest_lnodes=GridapP4est.setup_pXest_lnodes(Val{Dc}, ptr_pXest, ptr_pXest_ghost)

  cell_vertex_gids=GridapP4est.generate_cell_vertex_gids(ptr_pXest_lnodes,cellindices)

  cell_vertex_lids_nlvertices=GridapP4est.generate_cell_vertex_lids_nlvertices(cell_vertex_gids)

  cell_coordinates_and_panels=generate_cell_coordinates_and_panels(parts,
                                             coarse_discrete_model,
                                             ptr_pXest_connectivity,
                                             ptr_pXest,
                                             ptr_pXest_ghost)

  cube_grid_geo=generate_cube_grid_geo(cell_coordinates_and_panels)
  cube_grid_top=generate_cube_grid_top(cell_vertex_lids_nlvertices)

  # # do_on_parts(comm, cell_coordinates) do part, cell_coords
  # #   if part==1
  # #     println(cell_coords[9:12])
  # #   end
  # # end

  ddiscretemodel=
  map_parts(cube_grid_geo,cube_grid_top) do cube_grid_geo, cube_grid_top
    D2toD3AnalyticalMapCubedSphereDiscreteModel(cube_grid_geo, cube_grid_top)
  end

  # Write forest to VTK file
  #p4est_vtk_write_file(unitsquare_forest, C_NULL, "my_step")

  # Destroy lnodes
  GridapP4est.p4est_lnodes_destroy(ptr_pXest_lnodes)
  GridapP4est.p4est_ghost_destroy(ptr_pXest_ghost)
  # Destroy the forest
  GridapP4est.p4est_destroy(ptr_pXest)
  # Destroy the connectivity
  GridapP4est.p4est_connectivity_destroy(ptr_pXest_connectivity)

  sc_finalize()

  GridapDistributed.DistributedDiscreteModel(ddiscretemodel,cellindices)
end

struct D2toD3AnalyticalMapCubedSphereTriangulation{M} <: Triangulation{2,3}
  model::M
end

# Triangulation API

# Delegating to the underlying face Triangulation

Gridap.Geometry.get_cell_coordinates(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_coordinates(trian.model.cube_grid_geo)

Gridap.Geometry.get_reffes(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_reffes(trian.model.cube_grid_geo)

Gridap.Geometry.get_cell_type(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_type(trian.model.cube_grid_geo)

Gridap.Geometry.get_node_coordinates(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_node_coordinates(trian.model.cube_grid_geo)

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

Gridap.Geometry.get_grid(trian::D2toD3AnalyticalMapCubedSphereTriangulation) = trian.model.cube_grid_geo



struct D2toD3AnalyticalMapCubedSphereDiscreteModel{T,B,C} <: Gridap.Geometry.DiscreteModel{2,3}
  cell_map::T
  cube_model_top::B
  cube_grid_geo::C
  function D2toD3AnalyticalMapCubedSphereDiscreteModel(
      cube_grid_geo::Gridap.Geometry.UnstructuredGrid{2,3},
      cube_grid_top::Gridap.Geometry.UnstructuredGrid{2,2};
      radius=1)
    m1=Fill(Gridap.Fields.GenericField(MapCubeToSphere(radius)),num_cells(cube_grid_geo))
    m2=get_cell_map(cube_grid_geo)
    m=lazy_map(∘,m1,m2)

    cube_model_top=Gridap.Geometry.UnstructuredDiscreteModel(cube_grid_top)

    # Build output object
    T=typeof(m)
    B=typeof(cube_model_top)
    C=typeof(cube_grid_geo)
    GC.gc()
    new{T,B,C}(m,cube_model_top,cube_grid_geo)
  end
end

Gridap.Geometry.get_cell_map(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = model.cell_map
Gridap.Geometry.get_grid(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid(model.cube_model_top)
Gridap.Geometry.get_grid_topology(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid_topology(model.cube_model_top)
Gridap.Geometry.get_face_labeling(model::D2toD3AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_face_labeling(model.cube_model_top)
function Gridap.Geometry.Triangulation(a::D2toD3AnalyticalMapCubedSphereDiscreteModel)
  D2toD3AnalyticalMapCubedSphereTriangulation(a)
end
function Gridap.CellData.get_normal_vector(model::D2toD3AnalyticalMapCubedSphereDiscreteModel)
    cell_normal = Gridap.Geometry.get_facet_normal(model)
    Gridap.CellData.GenericCellField(cell_normal,Triangulation(model),ReferenceDomain())
end
function _unit_outward_normal(v::Gridap.Fields.MultiValue{Tuple{2,3}})
  n1 = v[1,2]*v[2,3] - v[1,3]*v[2,2]
  n2 = v[1,3]*v[2,1] - v[1,1]*v[2,3]
  n3 = v[1,1]*v[2,2] - v[1,2]*v[2,1]
  n = VectorValue(n1,n2,n3)
  n/norm(n)
end
function Gridap.Geometry.get_facet_normal(model::D2toD3AnalyticalMapCubedSphereDiscreteModel)
  # Get the Jacobian of the cubed sphere mesh
  map   = get_cell_map(model)
  Jt    = lazy_map(∇,map)
  p=lazy_map(Operation(_unit_outward_normal),Jt)
  p
end
