module Visualisation

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces

import Gridap.CellData: change_domain

using GridapGeosciences.Geometry

include("Vtk.jl")

export writevtk_with_cell_geomap, write_vtk_file_with_cell_geomap
export createvtk_with_cell_geomap,  create_vtk_file_with_cell_geomap
export mapped_vtkpoints

end
