module RefinementGeometryTests
#
# Verifies that the geometry of a uniformly refined AtlasDiscreteModel agrees with its
# topology, for the atlases of a mesh and of its simplices, both through refine(model)
# and through the num_refinements argument of the AtlasDiscreteModel constructor.
#
# The corner order of the children of a simplex depends on the global vertex ids of
# its parent, so on simplices the refinement rule varies from cell to cell.
#

using FillArrays: Fill
using Gridap
using Gridap.Adaptivity, Gridap.Arrays, Gridap.Geometry, Gridap.ReferenceFEs
using GridapGeosciences
using GridapGeosciences: get_atlas_grid, get_cell_ambient_maps
using Test

const MESHES = (
  CubedSphereMesh(1.0),
  CylinderMesh(1.0, 1.0),
  MobiusStripMesh(1.0, 0.3),
  CubedSphereWithThicknessMesh(1.0, 0.1),
)

vertex_images(model) =
  lazy_map(evaluate, get_cell_map(model), Fill(get_vertex_coordinates(only(get_polytopes(model))), num_cells(model)))

# Largest distance between the images of a vertex from two cells around it
function max_vertex_mismatch(model)
  cell_to_vertices = get_faces(get_grid_topology(model), num_cell_dims(model), 0)
  vertex_to_image = Dict{Int,Point{3,Float64}}()
  maximum(zip(cell_to_vertices, vertex_images(model))) do (vertices, images)
    maximum(norm(y - get!(vertex_to_image, v, y)) for (v, y) in zip(vertices, images))
  end
end

# Largest distance between a child cell map and its parent cell map at the child corners
function max_parent_mismatch(adapted)
  glue = get_adaptivity_glue(adapted)
  x = get_vertex_coordinates(only(get_polytopes(adapted)))
  parent_maps = get_cell_map(get_parent(adapted))
  ref_maps = get_n2o_reference_coordinate_map(glue)
  maximum(zip(vertex_images(adapted), glue.n2o_faces_map[end], ref_maps)) do (images, parent, ref_map)
    maximum(norm, images - evaluate(parent_maps[parent], evaluate(ref_map, x)))
  end
end

# The atlas of the cells of `model` as charts, refined at construction
function reconstruct(model, num_refinements)
  g = get_atlas_grid(model)
  coarse = UnstructuredDiscreteModel(
    UnstructuredGrid(g.param_grid), UnstructuredGridTopology(get_grid_topology(model)), get_face_labeling(model))
  AtlasDiscreteModel(coarse, g.cell_chart_coords, get_cell_ambient_maps(model), num_refinements)
end

@testset "$(typeof(mesh).name.name), $name" for
    mesh in MESHES, (name, split) in (("cells", identity), ("simplices", simplexify))
  # One refinement at least: the two unrefined Möbius panels have 4 vertices, too few
  # for the vertices of an edge to identify it once the panels are split.
  model = split(AtlasDiscreteModel(mesh, 1))
  once = refine(model)
  twice = refine(once)

  rules = unique(objectid, get_adaptivity_glue(once).refinement_rules)
  @test (length(rules) > 1) == (split === simplexify)

  for adapted in (once, twice)
    @test max_vertex_mismatch(adapted) < 1.0e-12
    @test max_parent_mismatch(adapted) < 1.0e-12
  end
  for num_refinements in 0:2
    @test max_vertex_mismatch(reconstruct(model, num_refinements)) < 1.0e-12
  end
end

end # module
