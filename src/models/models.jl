name(m::UQModel) = m.name

names(models::AbstractVector{<:UQModel}) = UncertaintyQuantification.name.(models)
