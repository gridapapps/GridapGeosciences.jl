
"""
Function to map points on the cube surface to the sphere. For a cubed sphere
of radius 1 we need sides of length 2, centred at the origin using the approach
of http://mathproofs.blogspot.com/2005/07/mapping-cube-to-sphere.html.
Arguments:
  xyz: 3d cartesian VectorValue
"""
function map_cube_to_sphere(xyz::VectorValue{3,Float64})
    x,y,z = xyz
    xₛ = x*sqrt(1-y^2/2-z^2/2+y^2*z^2/(3))
    yₛ = y*sqrt(1-z^2/2-x^2/2+x^2*z^2/(3))
    zₛ = z*sqrt(1-x^2/2-y^2/2+x^2*y^2/(3))
    Point(xₛ,yₛ,zₛ)
end

struct CubedSphereDiscreteModel{T,B,C} <: Gridap.Geometry.DiscreteModel{2,3}
  cell_map::T
  boun_model::B
  trian::C
  function CubedSphereDiscreteModel(n,order)
    domain = (-1,1,-1,1,-1,1)
    cells  = (n,n,n)
    model  = CartesianDiscreteModel(domain,cells)

    # Restrict model to cube surface
    labels = get_face_labeling(model)
    bgface_to_mask = Gridap.Geometry.get_face_mask(labels,"boundary",2)
    Γface_to_bgface = findall(bgface_to_mask)
    cube_surface_model = Gridap.Geometry.BoundaryDiscreteModel(Polytope{2},model,Γface_to_bgface)

    # Generate high-order FE map
    reffe=ReferenceFE(lagrangian,Float64,order)
    ψₖ=get_cell_map(cube_surface_model)
    reffe=ReferenceFE(QUAD,lagrangian,Float64,order)
    xref=Gridap.ReferenceFEs.get_node_coordinates(reffe)
    xrefₖ=Fill(xref,num_cells(cube_surface_model))
    ψₖxrefₖ=lazy_map(evaluate,ψₖ,xrefₖ)
    ψₖx   = lazy_map(Broadcasting(map_cube_to_sphere), ψₖxrefₖ)
    ϕref  = Gridap.ReferenceFEs.get_shapefuns(reffe)
    ϕrefₖ = Fill(ϕref,num_cells(cube_surface_model))
    cell_map = lazy_map(Gridap.Fields.linear_combination,ψₖx,ϕrefₖ)

    # Wrap up BoundaryTriangulation
    btrian=Triangulation(cube_surface_model)
    trian=CubedSphereTriangulation(cell_map,btrian)

    # Build output object
    T=typeof(cell_map)
    B=typeof(cube_surface_model)
    C=typeof(trian)
    GC.gc()
    new{T,B,C}(cell_map,cube_surface_model,trian)
  end
end

Gridap.Geometry.get_cell_map(model::CubedSphereDiscreteModel) = model.cell_map

# CAVEAT! We are returning a grid of the cube surface, i.e.,  no curvature, a flat surface.
# I am positive this will not affect numerical computations. These rely on the get_cell_map function, which
# actually reflects the appropiate curvature. Anyway, in case there is numerical failure, this is
# number one place to look at.
Gridap.Geometry.get_grid(model::CubedSphereDiscreteModel) = Gridap.Geometry.get_grid(model.boun_model)

Gridap.Geometry.get_grid_topology(model::CubedSphereDiscreteModel) = Gridap.Geometry.get_grid_topology(model.boun_model)
Gridap.Geometry.get_face_labeling(model::CubedSphereDiscreteModel) = Gridap.Geometry.get_face_labeling(model.boun_model)
Gridap.Geometry.get_triangulation(a::CubedSphereDiscreteModel) = a.trian
Gridap.Geometry.Triangulation(a::CubedSphereDiscreteModel) = a.trian
