# This script is to be executed from this folder (compile/)
#julia --project=../.. --color=yes -e 'using Pkg; Pkg.instantiate()'
#julia --project=../.. --color=yes -e 'using Pkg; pkg"precompile"'
julia --project=../.. -O2 --check-bounds=no --color=yes compile.jl
