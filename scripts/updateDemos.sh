
#! /bin/bash

julia --project=docs/ -e 'using Pkg; Pkg.update()'

julia --project=docs/ docs/literateDemo.jl