abstract type AbstractOrthogonalBasis end

struct PolynomialChaosBasis
    bases::Vector{<:AbstractOrthogonalBasis}
    p::Int
    d::Int
    α::Vector{Vector{Int64}}

    function PolynomialChaosBasis(bases::Vector{<:AbstractOrthogonalBasis}, p::Int)
        d = length(bases)
        return new(bases, p, d, multivariate_indices(p, d))
    end
end

function evaluate(Ψ::PolynomialChaosBasis, x::AbstractVector{Float64})
    res = ones(length(Ψ.α))
    for (i,α) in enumerate(Ψ.α)
        for (j,order) in enumerate(α)
            res[i] *= evaluate(Ψ.bases[j], x[j], order)
        end
    end
    return res
end

struct LegendreBasis <: AbstractOrthogonalBasis
    normalize::Bool
end

LegendreBasis() = LegendreBasis(true)

struct HermiteBasis <: AbstractOrthogonalBasis
    normalize::Bool
end

HermiteBasis() = HermiteBasis(true)

function evaluate(Ψ::AbstractOrthogonalBasis, x::AbstractVector{<:Real}, d::Int)
    return map(xᵢ -> evaluate(Ψ, xᵢ, d), x)
end

function evaluate(Ψ::LegendreBasis, x::Real, d::Int)
    val = P(x, d)
    return Ψ.normalize ? val * sqrt(2d + 1) : val
end

function evaluate(Ψ::HermiteBasis, x::Real, d::Int)
    val = He(x, d)
    return Ψ.normalize ? val / sqrt(factorial(d > 20 ? big(d) : d)) : val
end

function P(x, n::Integer)
    P⁻, P = zero(x), one(x)

    for i in 1:n
        P, P⁻ = ((2i - 1) * x * P - (i - 1) * P⁻) / i, P
    end
    return P
end

function He(x::Real, n::Integer)
    He⁻, He = zero(x), one(x)
    for i in 1:n
        He⁻, He = He, x * He - (i - 1) * He⁻
    end
    return He
end

function multivariate_indices(p::Int, d::Int)
    No = binomial(p + d, d)

    idx = vcat(zeros(Int64, 1, d), Matrix(I, d, d), zeros(Int64, No - d - 1, d))

    pᵢ = ones(Int64, d, No)

    for k in 2:No
        for i in 1:d
            pᵢ[i, k] = sum(pᵢ[i:d, k - 1])
        end
    end

    P = d + 1
    for k in 2:p
        L = P
        for j in 1:d, m in (L - pᵢ[j, k] + 1):L
            P += 1
            idx[P, :] = idx[m, :]
            idx[P, j] = idx[P, j] + 1
        end
    end

    return map(collect, eachrow(idx))
end

function map_to_base(_::LegendreBasis, x::AbstractVector)
    return quantile.(Uniform(-1, 1), cdf.(Normal(), x))
end

function map_to_base(_::HermiteBasis, x::AbstractVector)
    return x
end

function map_to_bases(Ψ::PolynomialChaosBasis, x::AbstractMatrix)
    return mapreduce((b, xᵢ) -> map_to_base(b, xᵢ), hcat, Ψ.bases, eachcol(x))
end

function map_from_base(_::LegendreBasis, x::AbstractVector)
    return quantile.(Normal(), cdf.(Uniform(-1, 1), x))
end

function map_from_base(_::HermiteBasis, x::AbstractVector)
    return x
end

function map_from_bases(Ψ::PolynomialChaosBasis, x::AbstractMatrix)
    return hcat(map((b, xᵢ) -> map_from_base(b, xᵢ), Ψ.bases, eachcol(x))...)
end

function quadrature_nodes(n::Int, _::LegendreBasis)
    x, _ = gausslegendre(n)
    return x
end

function quadrature_weights(n::Int, _::LegendreBasis)
    _, w = gausslegendre(n)
    return w ./ 2
end

function quadrature_nodes(n::Int, _::HermiteBasis)
    x, _ = gausshermite(n)
    return sqrt(2) * x
end

function quadrature_weights(n::Int, _::HermiteBasis)
    _, w = gausshermite(n)
    return w / sqrt(π)
end

function sample(n::Int, _::LegendreBasis)
    return rand(Uniform(-1, 1), n)
end

function sample(n::Int, _::HermiteBasis)
    return rand(Normal(), n)
end
