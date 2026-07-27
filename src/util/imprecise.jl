isimprecise(_::UQInput) = false

isimprecise(rv::RandomVariable{<:ProbabilityBox}) = true

isimprecise(i::IntervalVariable) = true

function isimprecise(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable})
    return any(isimprecise.(jd.m))
end

isimprecise(_::UQModel) = false

isimprecise(ipm::IntervalPredictorModel) = true

function isimprecise(inputs::AbstractVector{<:UQInput}, models::AbstractVector{<:UQModel})
    return isimprecise(inputs) || isimprecise(models)
end

function isimprecise(inputs::AbstractVector{<:UQInput})
    return any(isimprecise.(inputs))
end

function isimprecise(models::AbstractVector{<:UQModel})
    return any(isimprecise.(models))
end
