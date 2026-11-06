module SimplexifyAtlasTests

using Test
using Gridap
using Gridap.Geometry: Triangulation, num_cells, num_vertices, get_cell_polytopes
using Gridap.CellData: Measure
using Gridap.ReferenceFEs: is_simplex, simplexify as simplexify_polytope
using GridapGeosciences
using GridapGeosciences.Geometry: ManifoldStyle

const MESHES = (
  CubedSphereMesh(1.0),
  CylinderMesh(1.0, 1.0),
  MobiusStripMesh(1.0, 0.3),
  CubedSphereWithThicknessMesh(1.0, 0.1),
)

total_measure(model) = sum(∫(1.0)Measure(Triangulation(model), 4))

@testset "simplexify an atlas, $(typeof(mesh).name.name) at $r refinements" for
    mesh in MESHES, r in (0, 2)
  atlas = AtlasDiscreteModel(mesh, r)
  simplices = simplexify(atlas)

  @test isa(simplices, AtlasDiscreteModel)

  # One simplex per entry of the local split of the cell polytope: 2 triangles
  # per quadrilateral, 6 tetrahedra per hexahedron.
  per_cell = length(first(simplexify_polytope(first(get_cell_polytopes(atlas)))))
  @test num_cells(simplices) == per_cell * num_cells(atlas)
  @test all(is_simplex, get_cell_polytopes(simplices))

  @test num_vertices(simplices) == num_vertices(atlas)

  @test ManifoldStyle(simplices) == ManifoldStyle(atlas)

  @test total_measure(simplices) ≈ total_measure(atlas) rtol=1e-2
end

end # module SimplexifyAtlasTests
