"""
    Model(f::Function, inputs::Union{Symbol, Vector{Symbol}}, name::Symbol; distributed::Bool = false)

By default the function `f` must accept a `DataFrame` and return the result of the model for each row in the `DataFrame` as a vector. The `inputs` parameter specifies which columns from the `DataFrame` the function uses. The `name` is used to add the output to the `DataFrame`.
When `distributed` is set to `true` a `pmap` will evaluate each row the the `DataFrame` in parallel, making use of any available workers. In this case `f` must accept a `DataFrameRow`.
"""
struct Model <: UQModel
    func::Function
    inputs::Vector{Symbol}
    name::Symbol
    distributed::Bool
end

function Model(f::Function, inputs::Union{Symbol, Vector{Symbol}}, name::Symbol; distributed::Bool = false)
    return Model(f, wrap(inputs), name, distributed)
end

"""
    evaluate!(m::Model, df::DataFrame)

Calls `m.func` with `df` and adds the result to the `DataFrame` as a column `m.name` If `m.distributed` is `true` a `pmap` evaluates m.func for each row of `df` in parallel.
"""
function evaluate!(m::Model, df::DataFrame)
    if m.distributed
        df[!, m.name] = pmap(m.func, eachrow(df))

    else
        df[!, m.name] = m.func(df)
    end
    return nothing
end

function evaluate!(models::Vector{<:UQModel}, df::DataFrame)
    # use same working directory for all external models
    if any(isa.(models, ExternalModel))
        datetime = Dates.format(now(), "YYYY-mm-dd-HH-MM-SS")
        for m in models
            if isa(m, ExternalModel)
                evaluate!(m, df; datetime = datetime)
            else
                evaluate!(m, df)
            end
        end
    else
        for m in models
            evaluate!(m, df)
        end
    end
    # end
    return nothing
end
