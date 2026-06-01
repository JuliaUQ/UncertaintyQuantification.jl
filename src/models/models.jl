isimprecise(m::UQModel) = false

name(m::UQModel) = m.name

names(models::AbstractVector{<:UQModel}) = UncertaintyQuantification.name.(models)
