function isimprecise(inputs::AbstractVector{<:UQInput}, models::AbstractVector{<:UQModel})
    return any(isimprecise.(inputs)) || any(isimprecise.(models))
end

function isimprecise(input::UQInput)
    return isa(input, IntervalVariable) ||
           isa(input, RandomVariable{<:ProbabilityBox}) ||
           (
               isa(input, JointDistribution{<:Copulas.Copula,<:RandomVariable}) &&
               any(isa.(input.m, RandomVariable{<:ProbabilityBox}))
           )
end

