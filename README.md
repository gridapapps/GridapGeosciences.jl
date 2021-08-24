# GridapGeosciences

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://santiagobadia.github.io/GridapGeosciences.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://santiagobadia.github.io/GridapGeosciences.jl/dev)
[![Build Status](https://github.com/gridapapps/GridapGeosciences.jl/workflows/CI/badge.svg?branch=master)](https://github.com/gridapapps/GridapGeosciences.jl/actions)
[![Codecov](https://codecov.io/gh/gridapapps/GridapGeosciences.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridapp/GridapGeosciences.jl)

The goal of this repository is to showcase the suitability of the [Gridap](https://github.com/gridap) ecosystem of packages to solve geophysical flow problems. In this repo, you will find:

[[click here]](https://github.com/gridapapps/GridapGeosciences.jl/blob/master/test/DarcyCubedSphereTests.jl) A convergence study of Raviart-Thomas-DG mixed finite elements for the solution of a Darcy problem on the cubed sphere.

[[click here]](https://github.com/gridapapps/GridapGeosciences.jl/blob/master/test/LaplaceBeltramiCubedSphereTests.jl) A convergence study of grad-conforming finite elements for the solution of a Laplace-Beltrami problem on the cubed sphere.

[[click here]](https://github.com/gridapapps/GridapGeosciences.jl/blob/master/test/WaveEquationCubedSphereTests.jl) Numerical solution of the linear wave equation on the cubed sphere using a Strong-Stabilitity-Preserving Runge-Kutta explicit 2nd order method (SSPRK2) for time integration and Raviart-Thomas-DG mixed finite elements for spatial discretization.

  ![fig1](_readme/WAVE_10x10_0_2PI_2000_SSRK2.gif)  |  ![fig2](_readme/WAVE_10x10_1_2PI_4000_SSRK2.gif)
  --- | ---
  RT0-DG0 10x10 2000 steps  | RT1-DG1 10x10 4000 steps  
  ![fig3](_readme/WAVE_24x24_0_2PI_2000_SSRK2.gif) |  ![fig7](_readme/WAVE_24x24_1_2PI_4000_SSRK2.gif)
  RT0-DG0 10x10 2000 steps | RT1-DG1 24x24 4000 steps

  Wave equation velocity magnitude plots, time interval T=[0,2PI]