
julia --project=docs/ -e 'using Pkg; Pkg.update()'

julia --project=docs/ docs/literateDocs.jl

julia --project=docs/ docs/make.jl
