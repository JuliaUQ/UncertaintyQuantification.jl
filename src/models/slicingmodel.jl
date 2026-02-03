# TODO: Add interval propagation method
struct SlicingModel <: UQModel
    models::AbstractVector{<:UQModel}
    inputs::AbstractVector{<:UQInput}
    max::Bool
end

# evaluates the transformed SNS problem
function evaluate!(m::SlicingModel, df::DataFrame)
    intervals = filter(x -> isa(x, IntervalVariable), m.inputs)

    physical = copy(df)

    to_physical_space!(m.inputs, physical)

    if !isempty(intervals)
        DataFrames.hcat!(physical, sample(intervals, size(df, 1)))
    end

    perf = m.models[end].name

    propagate_intervals!(m.models, physical, m.max ? :ub : :lb)

    df[!, perf] = physical[:, perf]

    return nothing
end
