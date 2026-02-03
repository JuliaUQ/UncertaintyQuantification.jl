function to_standard_normal_space!(rv::RandomVariable{<:ProbabilityBox}, x::DataFrame)
    x[!, rv.name] =
        quantile.(Normal(), reverse_quantile.(Ref(rv.dist), collect(x[:, rv.name])))
    return nothing
end

function to_physical_space!(rv::RandomVariable{<:ProbabilityBox}, x::DataFrame)
    x[!, rv.name] = quantile.(rv.dist, cdf.(Normal(), collect(x[:, rv.name])))
    return nothing
end

bounds(rv::RandomVariable{<:ProbabilityBox}) = bounds(rv.dist)

function map_to_precise(x::AbstractVector{<:Real}, rv::RandomVariable{<:ProbabilityBox})
    return RandomVariable(map_to_precise(x, rv.dist), rv.name)
end
