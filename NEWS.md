# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed 

- Bumped version of Gridap to 0.20.9. Since [PR #61](https://github.com/gridapapps/GridapGeosciences.jl/pull/61).

## [0.7.1] - 2026-08-14

### Added
- tests for the surface Stokes problem (2D) using the Taylor--Hood inf-sup pair, and the associated vector Laplacian problem
in 2D.  
- added `test/Geometry/seq/SkeletonTriangulationTests.jl` to test the fix for [Issue#59](https://github.com/gridapapps/GridapGeosciences.jl/issues/59) 

### Changed
- ` ∇s` and `skew_∇s` now return contravariant components. This aligns with the theory paper
- For `SkeletonPair{<:CellField}`, each side lives on the corresponding `BoundaryTriangulation`. i.e. `SkeletonTriangulation.plus`
and `SkeletonTriangulation.minus` respectively. This is different to what Gridap current does where both sides live on the 
`SkeletonTriangulation`, which is problematic for `AtlasGrid` without a global coordinate system. See [Issue#59](https://github.com/gridapapps/GridapGeosciences.jl/issues/59) 

### Removed

## [0.7.0] - 2026-07-21

Major rewrite introducing the **Atlas** discrete model abstraction, a general multi-chart representation of curved manifolds that replaces the previous ad-hoc family of `CubedSphere*Parametric*`/`CubedSphere*Ambient*` models.

### Added
- `AtlasGrid{Dc,Da}` and `AtlasDiscreteModel{Dc,Da}`: a `Grid`/`DiscreteModel` pair for `Dc`-dimensional manifolds embedded in `Da`-dimensional ambient space, storing per-cell chart coordinates, ambient maps, and analytic metric fields. Distinguishes `IntrinsicManifold` (`Da=Dc`) and `ExtrinsicManifold` (`Da>Dc`) styles via the `ManifoldStyle` trait, with convenience type aliases `IntrinsicAtlasDiscreteModel`/`ExtrinsicAtlasDiscreteModel`.
- Canonical coarse meshes: `CylinderMesh`, `MobiusStripMesh`, `CubedSphereMesh`, the new 3D spherical-shell geometry `CubedSphereWithThicknessMesh`, and its adaptively-extruded counterpart `ExtrudedCubedSphereWithThicknessMesh` — each bundling coarse topology, chart coordinates, ambient maps, and analytic metric/inverse-metric fields (`CylinderMap`/`Metric`, `MobiusMap`/`Metric`, `CubedSphereMap`/`Metric`, `CubedSphereWithThicknessMap`/`Metric`, with their `Inv` counterparts and analytic first/second-order derivatives for automatic differentiation).
- Distributed Atlas models: `AtlasDistributedDiscreteModel` (linear cell partitioning with one-layer ghost cells, no `p4est` dependency), `AtlasOctreeDistributedDiscreteModel` (`p4est`-backed, 2D and 3D), and `ExtrudedAtlasOctreeDistributedDiscreteModel` (`p6est`-backed, extruded 3D shell), together with `IntrinsicAtlasDistributedDiscreteModel`/`ExtrinsicAtlasDistributedDiscreteModel` and their `Adapted*` counterparts.
- `Gridap.Adaptivity.refine`/`adapt` support for every Atlas model flavour (serial, distributed, octree, extruded), preserving the refinement parent chain and updating global ids across ranks; `generate_refined_models`, `generate_distributed_refined_models`, `generate_octree_distributed_refined_models`, and `generate_extruded_octree_distributed_refined_models` build full refinement hierarchies via successive `refine` calls. `vertically_uniformly_refine`/`horizontally_uniformly_refine` support anisotropic refinement of the extruded model.
- New `CellData` submodule consolidating the cell-field/differential-geometry API: `AmbientMapCellField`, `InvAmbientMapCellField`, `MetricCellField`, `InvMetricCellField`, `MeasureCellField`, `LatLonMapCellField`, and the surface differential operators `Δs`, `vecΔs`, `∇s`, `divs`, `curls`, `skew_∇s`, `skew_divs`, `dagger` — all working on `BodyFittedTriangulation`, `BoundaryTriangulation`, `SkeletonTriangulation`, `TriangulationView`, and `AdaptedTriangulation` over Atlas models, both serially and in distributed form. `Δs`/`∇s` support an automatic-differentiation path as well as an analytic one (`use_automatic_differentiation=false`).
- `pushforward_reference_normal`, `pushforward_parametric_normal`, `pullback_area_form`, and `get_sphere_surface_normal` for boundary/skeleton/adapted triangulations over Atlas models.
- Extensive new serial/MPI test suite under `test/AtlasDiscreteModels/` (cylinder, Möbius strip, cubed sphere and cubed-sphere-with-thickness geometry, Darcy on cylinder/cubed sphere, H(div) and Poisson on the cylinder, quadrature convergence, refinement ordering, metric-field validation, compressed-array optimisation), plus extended MPI coverage of the new octree/extruded models across the Laplacian, Geophysical, Projection, and AmbientModel test suites.
- Automated documentation deployment to GitHub Pages via `Documenter.jl`/`Literate.jl`.

### Changed
- Renamed `CubedSphereForwardMap`→`CubedSphereMap`, `CubedSphereInverseMap`→`CubedSphereInvMap`; `get_refined_models`→`generate_refined_models`, `get_distributed_refined_models`→`generate_distributed_refined_models`; `get_radius`/`get_thickness`→`get_sphere_radius`/`get_sphere_thickness` (and moved their implementation to `ConvergenceTools`); `get_surface_normal`→`get_sphere_surface_normal`; `normal_vec`→`sphere_surface_normal_vec`, `tangent_vec`→`sphere_tangent_vec_component`; `get_atlas_dmodel`→`get_atlas_model`.
- All tutorials/examples and solver test drivers (`AdvectionUpwinding`, `LaplaceBeltrami`, `ShallowWater`, `ThermalShallowWater`, `WaveEquation`, `LinearBoussinesq`, the Hodge-Laplacian and L2-projection tests, ...) ported from the old `ParametricCellField`/`AmbientCellField`/`piola`-based API to the new Atlas + `CellData` API; initial-condition and forcing functions now take ambient coordinates directly instead of closures over a forward map.
- `writevtk_with_cell_geomap`/`createvtk_with_cell_geomap` (serial and distributed) now take the geometry map as a `CellField`/`DistributedCellField` and call `change_domain` internally, replacing the previous raw-array (`AbstractArray`) based API.
- `test/Examples/` renamed to `test/Tutorials/`.
- Bumped `GridapP4est` compat bound from 0.3.14 to 0.3.15; package version bumped 0.6.2 → 0.7.0.

### Removed
- The entire pre-Atlas cubed-sphere model hierarchy: `CubedSphere2DParametricDiscreteModel`, `CubedSphere3DParametricDiscreteModel`, `CubedSphereAmbientDiscreteModel`, and their distributed/octree counterparts, together with `ParametricCellField`, `AmbientCellField`, and the panel-ids infrastructure (`PanelIds`, `TriangulationPanelIds`, `distributed_panel_ids`, `get_panel_ids`, ...) — all superseded by the Atlas + `CellData` API.
- `Helpers` functions tied to the removed API: `ambient_surflap`/`ambient_surfdiv`/`ambient_sgrad`, `panel_to_cartesian`, `piola` and the rest of `VectorPullback`, `forward_jacobian`, `covariant_basis`, `forward_pinv_jacobian`.
- The old custom `Adaptivity` module (`src/Adaptivity/`); uniform refinement is now provided by Gridap's `EdgeBasedRefinement` via `Gridap.Adaptivity.refine`.

### Fixed
- `get_cell_coordinates` for `GridView{<:AtlasGrid}`: local portions of a `DistributedTriangulation` (as `TriangulationView`) were returning junk node coordinates due to missing reindexing against the parent's cell coordinates.
- `MetricCellField`, `MeasureCellField`, `InvMetricCellField`, and `AmbientMapCellField` on skeleton triangulations now correctly rebase the plus/minus results onto the skeleton triangulation instead of the individual boundary triangulations.
- `Δs`, `∇s`, `divs`, and `LatLonMapCellField` on `AdaptedTriangulation` now return fields living on the adapted triangulation rather than on the underlying (pre-adaptation) one.
- `pullback_area_form` for `AdaptedTriangulation` now returns a proper `SkeletonPair`.
- `_adapt_atlas_octree_dmodel` now preserves the refinement parent chain correctly across successive adaptations.
- Fixed a swapped argument order (`num_vertical_refinements`/`num_horizontal_refinements`) in the `ExtrudedAtlasOctreeDistributedDiscreteModel` constructors.
- Fixed a segfault caused by garbage-collecting the shared `p4est_connectivity_t` object underlying refined octree/extruded models.
- Fixed the `OrientationStyle` type-parameter count in `AtlasGrid`, and the ambient-dimension type parameter of `AtlasOctreeDistributedDiscreteModel` (was hard-coded to 2, should be 3).
- `skew_∇s`, `skew_divs`, and `dagger` are now restricted to 2D manifolds; previously they could be (silently incorrectly) called on 3D shell models.

## [0.6.2] - 2026-06-25 

### Added
- `benchmark/` to compare the grad-grad term in the intrinsic vs. extrinsic approach. Follow the instructions in `benchmark/README.md` to run locally.
- Added the followng tutorials to `Examples/`
    * AmbientAdvectionSUPG.jl: to extrinsically solve the scalar transport equation with SUPG upwinding  
    * AmbientAdvectionUpwinding.jl: to extrinsically solve the scalar transport equation with DG upwinding  
    * LinearBoussinesq.jl: to solve the 3D linear Boussinesq equations 
    * ThermalShallowWater.jl: to solve the 2D thermal shallow water equations

### Changed
- Bumped dependencies to Gridap v0.20.7, GridapDistributed v0.4.16, GridapP4est v0.3.14, GridapSolvers v0.7.1, P4est_wrapper v0.2.5, and PartitionedArrays v0.3.4.
- Renamed `ForwardMap` → `CubedSphereForwardMap` and `InverseMap` → `CubedSphereInverseMap` (and the corresponding source files), for clarity and to avoid name clashes.
- Refactored serial `GradConformingFESpaces` to implement the new Gridap v0.20 hook `compute_cell_bases_changes` (dispatching on value type via the private helper `_compute_cell_bases_changes`) in place of the now-removed `get_cell_shapefuns` / `get_cell_dof_basis` overloads. The `collect∘transpose` workaround that was required by Gridap v0.19 has been dropped in favour of plain `transpose`.
- Refactored distributed `GradConformingFESpaces` to implement `compute_cell_bases_changes` for `CubedSphereParametricDistributedDiscreteModel` and to use the new `DistributedSingleFieldFESpace` constructor API from GridapDistributed v0.4.14+, replacing the old `FESpace` overloads that depended on the now-removed `_setup_dmodel_and_dtrian` helper.
- Changed the Nedelec constructor in `test/Laplacian/HodgeLaplacian_vector.jl` to take the kwarg `change_dof=false`


## [0.6.1]
### Added
- Added `CubedSphereAmbientDiscreteModel`. This is a serial implementation of the two dimensional cubed sphere in the 
ambient space. i.e. the dimension of the physical points is 3. 
    * Added refinement, triangulation, and interface of parametric model
- Added `CubedSphereAmbientDistributedDiscreteModel` and `CubedSphereAmbientOctreeDistributedDiscreteModel` as the distributed 
version of the ambient model. 
    * Added interface of parametric model, and `adapt_model`
- Added `AmbientCellField` as an anaglous version of `ParametricCellField`
- Added `ambient_sgrad`, `ambient_surflap`, `ambient_surfdiv`, to compute surface operators in ambient space
- Added `CellField` to recompute the triangulation to ensure proper handling of ghost cells in octree periodic meshes.
- Added `get_surface_normal` to compute the outward point normal to the surface for ambient models in serial and parallel
- Added `dagger` to compute $\tilde{k}\times \tilde{u}$ for ambient model 
- Added `perp` to compute $R u$ for parametric models
- Added `InverseMap`, appropriate generators and tests

- Added test for the ambient model that compare outputs to those of the parametric model for 
    * surface area, 
    * Laplace Beltrami, Hodge Laplacian (scalar)
    * Transient wave equation, transient shallow water
    * skew operators, surface differential operators 
    * panel ids

- Added tutorial for ambient model and Hodge Laplacian (scalar)

### Changed
- `nref`, `nc`, `nc_horizontal`, `nc_vertical`,`dx`, to be ammenable with ambient model

### Fixed 
- `CubedSphere2DParametricDistributedDiscreteModel` to properly distribute serial models
- `evaluate!(c,f::FieldGradient{1,<:ForwardMap3D},cellx::AbstractArray{<:VectorValue{3}})` to handle multiple caches
- Magic numbers in test/Geometry/seq/CellMapTests.jl

## [0.6.0] - 2026-05-06

This is the first public release of GridapGeosciences. Non-exhaustive list of new features:

**Cubed sphere discrete models**
- `CubedSphere2DParametricDiscreteModel` is a serial implementation of the two dimensional cubed sphere, parametric model
- `CubedSphere2DParametricDistributedDiscreteModel` is a distributed implementation of the two dimensional cubed sphere, parametric model
- `CubedSphere2DParametricOctreeDistributedDiscreteModel` is a distributed, adaptive implementation of the two dimensional cubed sphere, parametric model, provided by `p4est` through `GridapP4est.jl`
- `CubedSphere3DParametricOctreeDistributedDiscreteModel` is a distributed, adaptive implementation of the three dimensional cubed sphere, parametric model, provided by `p4est` through `GridapP4est.jl`

**Continuous vector-valued Lagrangian finite elements**
- We provide a serial and distributed implementation of vector-valued Lagrangrian finite elements on manifold.

**Time integrator for differential algebraic equations**
- `DAEFEOperator` extends the ODE framework in `Gridap` to differential algebraic equations that arise in atmospheric systems like the shallow water equations. Currently implemented for explicit Runge Kutta methods

**Mapped vtk files**
- `writevtk_with_cell_geomap(geo_map::AbstractArray,...)` and `createvtk_with_cell_geomap(geo_map::AbstractArray,...)` extend the `writevtk(...)` functionality of `Gridap` by applying a cell-wise geometrical map the triangulation for visualisation purposes only. The evaluation of the fields remains on the input triangulation.
 
## [0.5.0] - 2026-05-06
Release that includes all changes introduced from v0.4.0 untill right before the major refactoring of GridapGeosciences that uses an intrinsic differential geometry approach to represent the cube sphere manifold in 2D and 3D.

Importante note: No manually written Changelog is avaliable for this release.

## [0.4.0] - 2023-10-17

A changelog is not maintained for this version.
Bump dependencies to the latest version


## [0.1.0] - 2021-12-09

A changelog is not maintained for this version.
Tagging the repo right before implementing support for distributed computing

