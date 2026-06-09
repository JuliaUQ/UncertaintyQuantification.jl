abstract type AbstractRadialBasis <: AbstractBasis end

"""
    GaussianRadialBasis(c::AbstractMatrix{<:Real}, ϵ::Union{Real,Vector{<:Real}})

Construct a guassian radial basis using the center points given in the columns of `c` and shape parameter `ϵ`.
Different shape parameters can be assigned to each basis function by passing a vector as `ϵ`.
"""
struct GaussianRadialBasis <: AbstractRadialBasis
    c::AbstractMatrix{<:Real}
    ϵ::Union{Real,Vector{<:Real}}
end

"""
    PolyharmonicRadialBasis(c::AbstractMatrix{<:Real}, k::Int)

Construct a polyharmonic radial basis of degree `k` using the center points given in the columns of `c`.
"""
struct PolyharmonicRadialBasis <: AbstractRadialBasis
    c::AbstractMatrix{<:Real}
    k::Int
end

function (pb::PolyharmonicRadialBasis)(x::AbstractMatrix{<:Real})
    return mapreduce(xᵢ -> pb(xᵢ), hcat, eachcol(x))
end

function (pb::PolyharmonicRadialBasis)(x::AbstractVector{<:Real})
    return vec(_ϕ.(sqrt.(sum((pb.c .- x) .^ 2; dims=1)), pb.k))
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

function (gb::GaussianRadialBasis)(x::AbstractMatrix{<:Real})
    return mapreduce(xᵢ -> pb(xᵢ), hcat, eachcol(x))
end

function (gb::GaussianRadialBasis)(x::AbstractVector{<:Real})
    return vec(exp(-gb.ϵ .* sqrt.(sum((gb.c .- x) .^ 2; dims=1))))
end

Base.length(b::GaussianRadialBasis) = size(b.c, 2)
Base.length(b::PolyharmonicRadialBasis) = size(b.c, 2)
