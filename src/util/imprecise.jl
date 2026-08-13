"""
    isimprecise(i::UQInput)

Returns `true` if input `i` is imprecise.
"""
isimprecise(_::UQInput) = false

isimprecise(rv::RandomVariable{<:ProbabilityBox}) = true

isimprecise(i::IntervalVariable) = true

function isimprecise(jd::JointDistribution{<:Copulas.Copula, <:RandomVariable})
    return any(isimprecise.(jd.m))
end

"""
    isimprecise(m::UQModel)

Returns `true` if model `m` is imprecise.
"""
isimprecise(_::UQModel) = false

isimprecise(ipm::IntervalPredictorModel) = true

"""
    isimprecise(inputs::AbstractVector{<:UQInput}, models::AbstractVector{<:UQModel})

Returns `true` if any of the `inputs` or `models` is imprecise.
"""
function isimprecise(inputs::AbstractVector{<:UQInput}, models::AbstractVector{<:UQModel})
    return isimprecise(inputs) || isimprecise(models)
end

"""
    isimprecise(inputs::AbstractVector{<:UQInput})
    
Returns `true` if any of the `inputs` is imprecise.
"""
function isimprecise(inputs::AbstractVector{<:UQInput})
    return any(isimprecise.(inputs))
end
"""
    isimprecise(models::AbstractVector{<:UQModel})

Returns `true` if any of the `models` is imprecise.
"""
function isimprecise(models::AbstractVector{<:UQModel})
    return any(isimprecise.(models))
end
