abstract type AbstractRadialBasis <: AbstractBasis end

struct GaussianRadialBasis <: AbstractRadialBasis
	c::AbstractMatrix{<:Real}
	σ::Real
end

struct PolyharmonicRadialBasis <: AbstractRadialBasis
	c::AbstractMatrix{<:Real}
	k::Int
end

function (pb::PolyharmonicRadialBasis)(x::AbstractMatrix{<:Real})
	return mapreduce(xᵢ -> pb(xᵢ), hcat, eachcol(x))
end

function (pb::PolyharmonicRadialBasis)(x::AbstractVector{<:Real})
	return vec(_ϕ.(sqrt.(sum((pb.c .- x) .^ 2; dims = 1)), pb.k))
end

function _ϕ(r::Real, k::Int)
	if k % 2 != 0
		return r^k
	elseif r < 1
		return r^(k - 1) * log(r^r)
	else
		return r^k * log(r)
	end
end
