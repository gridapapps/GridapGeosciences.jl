module FESpaces

import Gridap.ReferenceFEs: GenericLagrangianRefFE, GradConformity, get_prebasis,
                            get_node_coordinates, get_polytope, num_dofs,
                            get_face_own_dofs, get_face_own_nodes, num_faces, get_offset,
                            ReferenceFEName, IdentityPiolaMap

import Gridap.Geometry: get_grid_topology, num_faces, num_cell_dims,
                        get_faces, num_cells, get_cell_map,
                        BodyFittedTriangulation
import Gridap.Fields: Map, VectorValue, TensorValue, Point, ∇
import Gridap.Arrays: array_cache, CachedMatrix, return_type, getindex!,
                      testitem, lazy_map, IdentityVector, setsize!, return_cache,
                      evaluate, evaluate!
import GridapGeosciences.Geometry: BFTATDMIM, IntrinsicAtlasDiscreteModel
import GridapGeosciences.Geometry: AtlasDiscreteModel,
                                   get_cell_ambient_maps,
                                   AmbientMapCellField, IntrinsicManifold
import Gridap.Geometry: Triangulation
import Gridap.FESpaces: compute_cell_bases_changes, _use_clagrangian, H1Conformity
import Gridap.Adaptivity: AdaptedDiscreteModel, AdaptedTriangulation
import Gridap.Helpers: @notimplemented
import Gridap.ReferenceFEs: linear_combination
import LinearAlgebra: I, ⋅
import Gridap.CellData: get_data
include("GradConformingFESpaces.jl")
export _generate_change_of_basis_matrices
export _get_value_type

end
