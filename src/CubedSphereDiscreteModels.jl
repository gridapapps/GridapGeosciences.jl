"""
Function object to map points on the cube surface to the sphere of radius r.
We need sides of length 2 centred at the origin using the approach
in http://mathproofs.blogspot.com/2005/07/mapping-cube-to-sphere.html.

Construct arguments:
  r  : sphere radius
Function object arguments:
  xyz: 3D cartesian coordinates of a point on the cube surface
"""
struct MapCubeToSphere{T} <: Function
  radius::T
end

function (map::MapCubeToSphere{T})(xyz) where T
  x,y,z = xyz
  xₛ = x*sqrt(1.0-y^2/2-z^2/2+y^2*z^2/(3.0))
  yₛ = y*sqrt(1.0-z^2/2-x^2/2+x^2*z^2/(3.0))
  zₛ = z*sqrt(1.0-x^2/2-y^2/2+x^2*y^2/(3.0))
  map.radius*Point(xₛ,yₛ,zₛ)
end

function generate_Γface_to_bgface(model)
  panel_to_entity=[23,26,24,25,21,22]
  entity_to_panel=Dict(23=>1,26=>2,24=>3,25=>4,21=>5,22=>6)
  ptr_ncells_panel=zeros(Int64,length(panel_to_entity)+1)
  labels = get_face_labeling(model)
  bgface_to_mask = Gridap.Geometry.get_face_mask(labels,"boundary",2)
  Γface_to_bgface = findall(bgface_to_mask)
  face_to_entity  = labels.d_to_dface_to_entity[3][Γface_to_bgface]
  for entity in face_to_entity
    panel=entity_to_panel[entity]
    ptr_ncells_panel[panel+1]=ptr_ncells_panel[panel+1]+1
  end
  ptr_ncells_panel[1]=1
  for i=1:length(ptr_ncells_panel)-1
    ptr_ncells_panel[i+1]=ptr_ncells_panel[i+1]+ptr_ncells_panel[i]
  end
  Γface_to_bgface_panelwise=similar(Γface_to_bgface)
  for (i,entity) in enumerate(face_to_entity)
    panel=entity_to_panel[entity]
    Γface_to_bgface_panelwise[ptr_ncells_panel[panel]]=Γface_to_bgface[i]
    ptr_ncells_panel[panel]=ptr_ncells_panel[panel]+1
  end
  Γface_to_bgface_panelwise
end

struct PolynomialMapCubedSphereDiscreteModel <: Gridap.Geometry.DiscreteModel{2,3}
  model::Gridap.Geometry.UnstructuredDiscreteModel{2,3}
end

Gridap.Geometry.get_cell_map(model::PolynomialMapCubedSphereDiscreteModel) = Gridap.Geometry.get_cell_map(model.model)
Gridap.Geometry.get_grid(model::PolynomialMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid(model.model)
Gridap.Geometry.get_grid_topology(model::PolynomialMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid_topology(model.model)
Gridap.Geometry.get_face_labeling(model::PolynomialMapCubedSphereDiscreteModel) = Gridap.Geometry.get_face_labeling(model.model)
Gridap.Geometry.Triangulation(a::PolynomialMapCubedSphereDiscreteModel) = PolynomialMapCubedSphereTriangulation(a.model)

function CubedSphereDiscreteModel(n,order; radius=1)
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
  Γface_to_bgface=generate_Γface_to_bgface(model)
  cube_surface_trian = BoundaryTriangulation(model,Γface_to_bgface)

  # Generate high-order FE map and ordering
  vector_reffe=ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  V = FESpace(cube_surface_trian,vector_reffe; conformity=:H1)
  vh = interpolate(MapCubeToSphere(radius),V)
  scalar_reffe=ReferenceFE(QUAD,lagrangian,Float64,order)
  xref=Gridap.ReferenceFEs.get_node_coordinates(scalar_reffe)
  xrefₖ=Fill(xref,num_cells(cube_surface_trian))
  vhx=lazy_map(evaluate,Gridap.CellData.get_data(vh),xrefₖ)
  V = FESpace(cube_surface_trian,scalar_reffe; conformity=:H1)
  node_coordinates = Vector{Point{3,Float64}}(undef,num_free_dofs(V))
  cell_node_ids    = get_cell_dof_ids(V)
  _cell_vector_to_dof_vector!(node_coordinates,cell_node_ids,vhx)
  cell_types  = collect(Fill(1,num_cells(cube_surface_trian)))
  cell_reffes = [scalar_reffe]

  cube_surface_grid = Gridap.Geometry.UnstructuredGrid(node_coordinates,
                                                       Gridap.Arrays.Table(cell_node_ids),
                                                       cell_reffes,
                                                       cell_types,
                                                       Gridap.Geometry.Oriented())

  cube_surface_model = Gridap.Geometry.compute_active_model(cube_surface_trian)
  topology            = Gridap.Geometry.get_grid_topology(cube_surface_model)
  labeling           = Gridap.Geometry.get_face_labeling(cube_surface_model)
  PolynomialMapCubedSphereDiscreteModel(
    Gridap.Geometry.UnstructuredDiscreteModel(cube_surface_grid,topology,labeling))
end


struct AnalyticalMapCubedSphereDiscreteModel{T,B} <: Gridap.Geometry.DiscreteModel{2,3}
  cell_map::T
  cubed_sphere_linear_model::B
  function AnalyticalMapCubedSphereDiscreteModel(n;radius=1)
    domain = (-1,1,-1,1,-1,1)
    cells  = (n,n,n)
    model  = CartesianDiscreteModel(domain,cells)

    # Restrict model to cube surface
    Γface_to_bgface=generate_Γface_to_bgface(model)
    cube_surface_trian = BoundaryTriangulation(model,Γface_to_bgface)

    m1=Fill(Gridap.Fields.GenericField(MapCubeToSphere(radius)),num_cells(cube_surface_trian))
    m2=get_cell_map(cube_surface_trian)
    m=lazy_map(∘,m1,m2)

    cubed_sphere_linear_model=CubedSphereDiscreteModel(n,1)

    # Build output object
    T=typeof(m)
    B=typeof(cubed_sphere_linear_model)
    new{T,B}(m,cubed_sphere_linear_model)
  end
end

Gridap.Geometry.get_cell_map(model::AnalyticalMapCubedSphereDiscreteModel) = model.cell_map
Gridap.Geometry.get_grid(model::AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid(model.cubed_sphere_linear_model)
Gridap.Geometry.get_grid_topology(model::AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_grid_topology(model.cubed_sphere_linear_model)
Gridap.Geometry.get_face_labeling(model::AnalyticalMapCubedSphereDiscreteModel) = Gridap.Geometry.get_face_labeling(model.cubed_sphere_linear_model)
Gridap.Geometry.Triangulation(a::AnalyticalMapCubedSphereDiscreteModel) = AnalyticalMapCubedSphereTriangulation(a)

function CubedSphereDiscreteModel(n;radius=1)
  AnalyticalMapCubedSphereDiscreteModel(n;radius)
end
