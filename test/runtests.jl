using GridapGeosciences
using Test

TESTCASE = get(ENV, "TESTCASE", "seq")

# Sequential tests
if TESTCASE ∈ ("all", "seq", "seq-atlas-discrete-models")
  include("AtlasDiscreteModels/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-l2-projection")
  include("Projection/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-laplacian")
  include("Laplacian/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-geophysical")
  include("Geophysical/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-transient")
  include("Transient/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-geometry")
  include("Geometry/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-multigrid")
  include("Multigrid/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-tutorials")
  include("Examples/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-fields")
  include("Fields/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-ambient-model")
  include("AmbientModel/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-autodiff")
  include("Autodiff/seq/runtests.jl")
end

# MPI tests
if TESTCASE ∈ ("all", "mpi", "mpi-atlas-discrete-models")
   include("AtlasDiscreteModels/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-l2-projection")
   include("Projection/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-laplacian")
  include("Laplacian/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-geophysical")
  include("Geophysical/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-transient")
  include("Transient/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-geometry")
  include("Geometry/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-models")
  include("Distributed/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-multigrid")
  include("Multigrid/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-tutorials")
  include("Examples/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-ambient-model")
  include("AmbientModel/mpi/runtests.jl")
end
