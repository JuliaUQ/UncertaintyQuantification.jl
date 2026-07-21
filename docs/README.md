# Documentation

To build the documentation, run `scripts/buildDocs.sh` (Linux / Mac) or `scripts/buildDocs.ps1` (Windows) from project root folder.

Alternatively start julia in the `docs` environment with `julia --project=docs` and run the following code.

```julia
pkg> up # install latest documentation dependencies
pkg> dev . # add UncertaintyQuantification.jl as dev dependency
include("docs/make.jl") # build the docs
```

The documentation can be viewed using the *LiveServer* package.

```julia
using LiveServer

LiveServer.serve(; dir = "build/1")
```
