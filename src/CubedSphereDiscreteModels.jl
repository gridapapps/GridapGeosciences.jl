"""
Function to map points on the cube surface to the sphere. For a cubed sphere
of radius 1 we need sides of length 2, centred at the origin using the approach
of http://mathproofs.blogspot.com/2005/07/mapping-cube-to-sphere.html.
Arguments:
  xyz: 3d cartesian VectorValue
"""
function map_cube_to_sphere(xyz)
    x,y,z = xyz
    xₛ = x*sqrt(1-y^2/2-z^2/2+y^2*z^2/(3))
    yₛ = y*sqrt(1-z^2/2-x^2/2+x^2*z^2/(3))
    zₛ = z*sqrt(1-x^2/2-y^2/2+x^2*y^2/(3))
    Point(xₛ,yₛ,zₛ)
end

function CubedSphereDiscreteModel(n,order)
  function _cell_vector_to_dof_vector!(dof_vector,cell_node_ids, cell_vector)
    cache_cell_node_ids = array_cache(cell_node_ids)
    cache_cell_vector   = array_cache(cell_vector)
    for k=1:length(cell_node_ids)
       current_node_ids = getindex!(cache_cell_node_ids,cell_node_ids,k)
       current_values   = getindex!(cache_cell_vector,cell_vector,k)
       for (i,id) in enumerate(current_node_ids)
        dof_vector[current_node_ids[i]]=current_values[i]
       end
    end
  end

  domain = (-1,1,-1,1,-1,1)
  cells  = (n,n,n)
  model  = CartesianDiscreteModel(domain,cells)

  # Restrict model to cube surface
  labels = get_face_labeling(model)
  bgface_to_mask = Gridap.Geometry.get_face_mask(labels,"boundary",2)
  Γface_to_bgface = findall(bgface_to_mask)
  cube_surface_model = Gridap.Geometry.BoundaryDiscreteModel(Polytope{2},model,Γface_to_bgface)

  # Generate high-order FE map and ordering
  vector_reffe=ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  V = FESpace(cube_surface_model,vector_reffe; conformity=:H1)
  vh = interpolate(map_cube_to_sphere,V)
  scalar_reffe=ReferenceFE(QUAD,lagrangian,Float64,order)
  xref=Gridap.ReferenceFEs.get_node_coordinates(scalar_reffe)
  xrefₖ=Fill(xref,num_cells(cube_surface_model))
  vhx=lazy_map(evaluate,Gridap.CellData.get_data(vh),xrefₖ)
  node_coordinates = Vector{Point{3,Float64}}(undef,num_free_dofs(V))
  V = FESpace(cube_surface_model,scalar_reffe; conformity=:H1)
  cell_node_ids    = get_cell_dof_ids(V)
  _cell_vector_to_dof_vector!(node_coordinates,cell_node_ids,vhx)
  cell_types  = collect(Fill(1,num_cells(cube_surface_model)))
  cell_reffes = [scalar_reffe]

  cube_surface_grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                                       Gridap.Arrays.Table(cell_node_ids),
                                                       cell_reffes,
                                                       cell_types,
                                                       Gridap.Geometry.Oriented())
  topology=Gridap.Geometry.get_grid_topology(cube_surface_model)
  face_labeling=Gridap.Geometry.get_face_labeling(cube_surface_model)
  Gridap.Geometry.UnstructuredDiscreteModel(cube_surface_grid,topology,face_labeling)
end

struct AnalyticalMapCubedSphereDiscreteModel{T,B,C} <: Gridap.Geometry.DiscreteModel{2,3}
  cell_map::T
  cubed_sphere_model::B
  trian::C
  function AnalyticalMapCubedSphereDiscreteModel(n)
    domain = (-1,1,-1,1,-1,1)
    cells  = (n,n,n)
    model  = CartesianDiscreteModel(domain,cells)

    # Restrict model to cube surface
    labels = get_face_labeling(model)
    bgface_to_mask = Gridap.Geometry.get_face_mask(labels,"boundary",2)
    Γface_to_bgface = findall(bgface_to_mask)
    cube_surface_model = Gridap.Geometry.BoundaryDiscreteModel(Polytope{2},model,Γface_to_bgface)

    m1=Fill(Gridap.Fields.GenericField(map_cube_to_sphere),num_cells(cube_surface_model))
    m2=get_cell_map(cube_surface_model)
    m=lazy_map(∘,m1,m2)

    cubed_mesh_model=CubedSphereDiscreteModel(n,1)

    # Wrap up BoundaryTriangulation
    btrian=Triangulation(cubed_mesh_model)
    trian=AnalyticalMapCubedSphereTriangulation(m,btrian)

    # Build output object
    T=typeof(m)
    B=typeof(cubed_mesh_model)
    C=typeof(trian)
    GC.gc()
    new{T,B,C}(m,cubed_mesh_model,trian)
  end
end

Gridap.Geometry.get_cell_map(model::AnalyticalMapCubedSphereDiscreteModel) = model.cell_map
Gridap.Geometry.get_grid(model::AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid(model.cubed_sphere_model)
Gridap.Geometry.get_grid_topology(model::AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid_topology(model.cubed_sphere_model)
Gridap.Geometry.get_face_labeling(model::AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_face_labeling(model.cubed_sphere_model)
Gridap.Geometry.get_triangulation(a::AnalyticalMapCubedSphereDiscreteModel) = a.trian
Gridap.Geometry.Triangulation(a::AnalyticalMapCubedSphereDiscreteModel) = a.trian

function CubedSphereDiscreteModel(n)
  AnalyticalMapCubedSphereDiscreteModel(n)
end

const CSDMT = Union{AnalyticalMapCubedSphereDiscreteModel,
                 <:Gridap.Geometry.UnstructuredDiscreteModel{2,3}}

function Gridap.CellData.get_normal_vector(model::CSDMT)
    cell_normal = Gridap.Geometry.get_facet_normal(model)
    Gridap.CellData.GenericCellField(cell_normal,Triangulation(model),ReferenceDomain())
end

function Gridap.Geometry.get_facet_normal(model::CSDMT)
  function _unit_outward_normal(v::Gridap.Fields.MultiValue{Tuple{2,3}},sign_flip::Bool)
    n1 = v[1,2]*v[2,3] - v[1,3]*v[2,2]
    n2 = v[1,3]*v[2,1] - v[1,1]*v[2,3]
    n3 = v[1,1]*v[2,2] - v[1,2]*v[2,1]
    n = VectorValue(n1,n2,n3)
    (-1)^sign_flip*n/norm(n)
  end

  # Get the Jacobian of the cubed sphere mesh
  trian = Triangulation(model)
  map   = get_cell_map(trian)
  Jt    = lazy_map(∇,map)

  # Get the index of the panel for each element
  fl = get_face_labeling(model)
  panel_id  = fl.d_to_dface_to_entity[3]
  sign_flip = [panel_id[i] == 25 || panel_id[i] == 21 || panel_id[i] == 24 for i=1:length(panel_id)]
  fsign_flip = lazy_map(Gridap.Fields.ConstantField,sign_flip)

  lazy_map(Operation(_unit_outward_normal),Jt,fsign_flip)
end
