module AtlasRefinementTests

# Tests `Gridap.Adaptivity.refine(::AtlasDiscreteModel)`, which splits every
# chart cell into 2^Dc children, composes the parent's ref-to-chart map with the
# sub-cell reference positions to give the children their chart coordinates, and
# hands the children their parent's ambient map and metric.
#
# What is checked here is the chart data an atlas carries, not the connectivity:
# Gridap's own adaptivity tests cover the glue and the grid transfers, and
# `RefinementOrderingTests` covers the parent-major ordering of the underlying
# `UnstructuredDiscreteModel` that `refine` builds on.

using Test
using Gridap
using Gridap.Adaptivity: refine, get_model
using Gridap.Geometry: Triangulation, num_cells
using Gridap.CellData: Measure, get_cell_measure
using GridapGeosciences
using LinearAlgebra: norm

const MESHES = (                         # Dc, area
  (CubedSphereMesh(1.0),                   2, 4π),
  (CylinderMesh(1.0, 1.0),                 2, 2π),
  (MobiusStripMesh(1.0, 0.3),              2, nothing),
  (CubedSphereWithThicknessMesh(1.0, 0.1), 3, 4π/3*(1.1^3 - 1)),
)

total_measure(model) = sum(∫(1.0)Measure(Triangulation(model), 4))
cell_measures(model) = sort(collect(get_cell_measure(Triangulation(model))))
chart_coords(model) = GridapGeosciences.get_atlas_grid(model).cell_chart_coords

@testset "children share their parent's chart, $(typeof(mesh).name.name)" for
    (mesh, Dc, _) in MESHES
  coarse = AtlasDiscreteModel(mesh, 0)
  adapted = refine(coarse)
  fine = get_model(adapted)
  parent = Vector{Int}(adapted.glue.n2o_faces_map[Dc+1])

  coarse_grid = GridapGeosciences.get_atlas_grid(coarse)
  fine_grid = GridapGeosciences.get_atlas_grid(fine)

  @test all(eachindex(parent)) do cell
    fine_grid.cell_ambient_maps[cell] === coarse_grid.cell_ambient_maps[parent[cell]]
  end
  @test all(eachindex(parent)) do cell
    fine_grid.cell_metric[cell] === coarse_grid.cell_metric[parent[cell]]
  end

  # Every parent keeps exactly its 2^Dc children, which is what lets the chart
  # coordinates be built by tiling one parent's sub-cell positions.
  @test all(p -> count(==(p), parent) == 2^Dc, 1:num_cells(coarse))
end

@testset "refine ∘ refine is the mesh refined twice at construction, $(typeof(mesh).name.name)" for
    (mesh, Dc, _) in MESHES
  twice = refine(refine(AtlasDiscreteModel(mesh, 0)))
  direct = AtlasDiscreteModel(mesh, 2)

  @test num_cells(get_model(twice)) == num_cells(direct)
  @test total_measure(twice) == total_measure(direct)
  @test maximum(abs, cell_measures(twice) - cell_measures(direct)) < 1e-15
end

@testset "refinement makes the volume measure converge, $(typeof(mesh).name.name)" for
    (mesh, Dc, exact) in MESHES
  exact === nothing && continue
  models = Any[AtlasDiscreteModel(mesh, 0)]
  for _ in 1:2
    push!(models, refine(last(models)))
  end
  errors = map(m -> abs(total_measure(m) - exact), models)

  if mesh isa CylinderMesh
    @test all(e -> e < 1e-14, errors)
  else
    @test errors[3] < errors[2] < errors[1]
  end
end

end # module AtlasRefinementTests
