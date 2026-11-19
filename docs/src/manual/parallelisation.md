# Parallelisation

_UncertaintyQuantification_ provides several possibilities to parallelise the execution of a model. Which method to use strongly depends on the model and the available hardware. The simplest way to run a model in parallel is passing the `distributed=true` keyword to the [`Model`](@ref) constructor. If any _Distributed_ workers are present, the evaluation of the rows in the `DataFrame` will be distributed among them. For this to work, _UncertaintyQuantification_ must be loaded with the `@everywhere` macro in order to be loaded on all workers. Each call to the function used in the [`Model`](@ref) is passed a `DataFrameRow` instead of the full `DataFrame`.

In the following example, we first load `Distributed` and add four local workers. A simple model is then evaluated in parallel. Finally, the workers are removed.

```julia
using Distributed
addprocs(4) # add 4 local workers

# setup of the model and inputs
@everywhere begin

    using UncertaintyQuantification

    x = RandomVariable(Normal(), :x)
    y = RandomVariable(Normal(), :y)

    m = Model(df -> sqrt(df.x^2 .+ df.y^2), :z; distributed = true)
end

samples = sample([x, y], 1000)
evaluate!(m, samples)

rmprocs(workers()) # release the local workers
```

It is important to note, that the parallelisation requires some overhead to distribute the function calls to the workers. Therefore it performs significantly slower than the [`Model`](@ref) with `distributed=false` with vectorized operations for a simple function as in this example. The method of executing a model in parallel presented above also applies to the `ExternalModel`.

!!! note
For heavier external models the use of parallel compute clusters through the [`SlurmInterface`](@ref) is recommended. See [High Performance Computing](@ref).
