# GridapGeosciences

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://santiagobadia.github.io/GridapGeosciences.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://santiagobadia.github.io/GridapGeosciences.jl/dev)
[![Build Status](https://github.com/gridapapps/GridapGeosciences.jl/workflows/CI/badge.svg?branch=master)](https://github.com/gridapapps/GridapGeosciences.jl/actions)
[![Codecov](https://codecov.io/gh/gridapapps/GridapGeosciences.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridapp/GridapGeosciences.jl)

The goal of this repository is to showcase the suitability of the [Gridap](https://github.com/gridap) ecosystem of packages to solve geophysical flow problems. In this repo, you will find:

* [[click here]](https://github.com/gridapapps/GridapGeosciences.jl/blob/master/test/DarcyCubedSphereTests.jl) A convergence study of Raviart-Thomas-DG mixed finite elements for the solution of a Darcy problem on the cubed sphere.

* [[click here]](https://github.com/gridapapps/GridapGeosciences.jl/blob/master/test/LaplaceBeltramiCubedSphereTests.jl) A convergence study of grad-conforming finite elements for the solution of a Laplace-Beltrami problem on the cubed sphere.

* [[click here]](https://github.com/gridapapps/GridapGeosciences.jl/blob/master/test/sequential/WaveEquationCubedSphereTests.jl) Numerical solution of the linear wave equation on the cubed sphere using a Strong-Stabilitity-Preserving Runge-Kutta explicit 2nd order method (SSPRK2) for time integration and Raviart-Thomas-DG mixed finite elements for spatial discretization.

* [[click here]](https://github.com/gridapapps/GridapGeosciences.jl/blob/master/src/ShallowWaterThetaMethodFullNewton.jl) Numerical solution of the Nonlinear Rotating Shallow Water Equations on the cubed sphere using a fully implicit theta-method for time integration, full Newton nonlinear solves, and compatible mixed finite elements for spatial discretization.

* [[click here]](https://github.com/gridapapps/GridapGeosciences.jl/tree/master/test/mpi) MPI-parallelization of all the above using [GridapDistributed.jl](https://github.com/gridap/GridapDistributed.jl) and its satellite packages [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl) and [GridapP4est.jl](https://github.com/gridap/GridapP4est.jl).

* [[click here]] More to come ...
 

<p align="center">
  <img src="_readme/NSWE_48x48_1_trapezoidal_dt_480_tau_dtdiv2.gif">
  Vorticity field for the Nonlinear Rotating Shallow Water Equations on the cubed sphere. Galewsky benchmark.
</p>
