# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `benchmark/` to compare the grad-grad term in the intrinsic vs. extrinsic approach. Follow the instructions in `benchmark/README.md` to run locally.

### Changed
- Bumped dependencies to Gridap v0.20.7, GridapDistributed v0.4.16, GridapP4est v0.3.14, GridapSolvers v0.7.1, P4est_wrapper v0.2.5, and PartitionedArrays v0.3.4.
- Renamed `ForwardMap` → `CubedSphereForwardMap` and `InverseMap` → `CubedSphereInverseMap` (and the corresponding source files), for clarity and to avoid name clashes.
- Refactored serial `GradConformingFESpaces` to implement the new Gridap v0.20 hook `compute_cell_bases_changes` (dispatching on value type via the private helper `_compute_cell_bases_changes`) in place of the now-removed `get_cell_shapefuns` / `get_cell_dof_basis` overloads. The `collect∘transpose` workaround that was required by Gridap v0.19 has been dropped in favour of plain `transpose`.
- Refactored distributed `GradConformingFESpaces` to implement `compute_cell_bases_changes` for `CubedSphereParametricDistributedDiscreteModel` and to use the new `DistributedSingleFieldFESpace` constructor API from GridapDistributed v0.4.14+, replacing the old `FESpace` overloads that depended on the now-removed `_setup_dmodel_and_dtrian` helper.

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

**Time integrator for differential algrabic equations**
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

