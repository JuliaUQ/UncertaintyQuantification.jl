"""
    Model(f::Function, name::Symbol)

The function `f` must accept a `DataFrame` and return the result of the model for each row in the `DataFrame` as a vector. The `name` is used to add the output to the `DataFrame`.

"""
struct Model <: UQModel
    func::Function
    name::Symbol
end

"""
    ParallelModel(f::Function, name::Symbol)

The  `ParallelModel`  does what the `Model` does with a small difference. The function `f` is passed a `DataFrameRow` not the full `DataFrame`.
If workers (through `Distributed`) are present, the rows are evaluated in parallel.

"""
struct ParallelModel <: UQModel
    func::Function
    name::Symbol
end

function (m::Model)(df::DataFrame)
    return m.func(df)
end

function (m::ParallelModel)(df::DataFrame)
    return pmap(row -> only(m.func(DataFrame(row))), eachrow(df))
end

"""
    evaluate!(m::Model, df::DataFrame)

Calls `m.func` with `df` and adds the result to the `DataFrame` as a column `m.name`
"""
function evaluate!(m::Model, df::DataFrame)
    df[!, m.name] = m.func(df)
    return nothing
end

"""
    evaluate!(m::ParallelModel, df::DataFrame)

Calls `m.func` for each row of `df` and adds the result to the `DataFrame` as a column `m.name`.
If workers are added through `Distributed`, the rows will be evaluated in parallel.
"""
function evaluate!(m::ParallelModel, df::DataFrame)
    df[!, m.name] = pmap(row -> only(m.func(DataFrame(row))), eachrow(df))
    return nothing
end

function evaluate!(models::Vector{<:UQModel}, df::DataFrame)
    # use same working directory for all external models
    datetime = Dates.format(now(), "YYYY-mm-dd-HH-MM-SS")
    for m in models
        if isa(m, ExternalModel)
            evaluate!(m, df; datetime=datetime)
        else
            evaluate!(m, df)
        end
    end
    return nothing
end
