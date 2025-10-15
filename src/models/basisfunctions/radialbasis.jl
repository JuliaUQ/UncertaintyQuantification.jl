abstract type AbstractRadialBasis <:AbstractBasis end

struct GaussianRadialBasis <: AbstractRadialBasis
    c::AbstractMatrix{<:Real}
    σ::Real
end

struct PolyharmonicRadialBasis <: AbstractRadialBasis
    c::AbstractMatrix{<:Real}
    k::Int
end
