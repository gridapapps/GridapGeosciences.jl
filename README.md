# GridapGeosciences 

 
GridapGeosciences.jl extends the [Gridap ecosystem](https://github.com/gridap) to the numerical approximation of partial differential equations on two and three dimensional general manifolds (including the cubed sphere for atmospheric dynamic simulations).  The manifold meshes are designed with high performance computing in mind.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](PENDING)
[![Build Status](https://github.com/gridapapps/GridapGeosciences.jl/workflows/CI/badge.svg?branch=master)](https://github.com/gridapapps/GridapGeosciences.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/gridapapps/GridapGeosciences.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridapapps/GridapGeosciences.jl)


## Installation
GridapGeosciences is a registered package in the official [Julia package registry](https://github.com/JuliaRegistries/General).  Thus, the installation of GridapGeosciences is straight forward using the [Julia's package manager](https://julialang.github.io/Pkg.jl/v1/). Open the Julia REPL, type `]` to enter package mode, and install as follows

```julia
pkg> add GridapGeosciences
```

## (Non-exhaustive) list of features

- **Cubed sphere discrete models**: We provide serial and distributed implementations of the two dimensional cubed sphere, and a distributed implementation of the three dimensional cubed sphere. Highly scalable, adaptive meshes are provided by `p4est` through `GridapP4est.jl`.
- **Continuous vector-valued Lagrangian finite elements**: We provide a serial and distributed implementation of vector-valued Lagrangrian finite elements on manifold.
- **Time integrator for differential algrabic equations**: We extend the explicit Runge Kutta framework in `Gridap` to differential algebraic equations that arise in atmospheric systems like the shallow water equations. 
- **Mapped vtk files**: We extend the visualisation framework in `Gridap` by applying a cell-wise geometrical map the parametric triangulation to visualise solutions in the ambient space.  


## Examples

A list of examples is available in `test/Examples`. 
These include well known atmospheric applications such as scalar advection, linear wave equation, and the shallow water equations. 
The benchmark Laplace Beltrami problem is also available. 


## Citation

Please cite the `GridapGeosciences` main project as indicated [here](https://github.com/gridap/Gridap.jl#how-to-cite-gridap), 
and use the reference below in any publication that uses `GridapGeosciences`:

```
@article{Tambyah2026, 
  year = {2026},   
  author = {Tamara A. Tambyah and Alberto F. Martín and David Lee and Santiago Badia}, 
  title = {GridapGeosciences.jl: A Julia finite element package for partial differential equations on the sphere [in preparation]}, 
} 
```
## Contributing

GridapGeosciences is a collaborative project open to contributions. If you want to contribute, please take into account:

- Before opening a PR with a significant contribution, contact the project administrators by [opening an issue](https://github.com/gridapapps/GridapGeosciences.jl/issues/new) describing what you are willing to implement. Wait for feedback from other community members.
- We adhere to the contribution and code-of-conduct instructions of the Gridap.jl project, available [here](https://github.com/gridap/Gridap.jl/blob/master/CONTRIBUTING.md) and [here](https://github.com/gridap/Gridap.jl/blob/master/CODE_OF_CONDUCT.md), resp.  Please, carefully read and follow the instructions in these files.
- Open a PR with your contribution.
 
## Image gallery 

<p align="center">
  <img src="_readme/NSWE_48x48_1_trapezoidal_dt_480_tau_dtdiv2.gif">
  Vorticity field for the Nonlinear Rotating Shallow Water Equations on the cubed sphere. Galewsky benchmark.
</p>
