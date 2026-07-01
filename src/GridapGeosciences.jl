module GridapGeosciences

# __precompile__(false)

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays


include("Fields/Fields.jl")

include("Helpers/Helpers.jl")

include("Geometry/Geometry.jl")

include("CellData/CellData.jl")

include("FESpaces/FESpaces.jl")

include("ODEs/ODEs.jl")

include("Visualisation/Visualisation.jl")

include("Distributed/Distributed.jl")

include("MultilevelTools/MultilevelTools.jl")

include("ConvergenceTools/ConvergenceTools.jl")

include("Exports.jl")

end
