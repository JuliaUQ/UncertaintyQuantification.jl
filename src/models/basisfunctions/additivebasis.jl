struct AdditiveBasis <: AbstractBasis
	bases::AbstractVector{<:AbstractBasis}
end

function (ab::AdditiveBasis)(x::AbstractVecOrMat{<:Real})
	return mapreduce(b -> b(x), vcat, ab.bases)
end
