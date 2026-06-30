module ConvergenceTools


using Test
using Gridap
using Gridap.Geometry, Gridap.Adaptivity
using GridapDistributed
using GridapP4est
using PartitionedArrays

using GridapGeosciences.Geometry
using GridapGeosciences.Distributed

import GridapGeosciences.Geometry: _chart_maps
import GridapGeosciences.Fields: CubedSphereMap, CubedSphereWithThicknessMap

include("Tools.jl")

export p_convergence_auto_test, h_convergence_auto_test
export nref, nc, nc_horizontal, nc_vertical, dx, dx_horizontal
export convergence_rate


end
