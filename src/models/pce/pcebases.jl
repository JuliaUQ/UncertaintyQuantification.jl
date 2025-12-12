abstract type AbstractOrthogonalBasis end

struct PolynomialChaosBasis
    bases::Vector{<:AbstractOrthogonalBasis}
    p::Int
    d::Int
    α::Vector{Vector{Int64}}

    function PolynomialChaosBasis(bases::Vector{<:AbstractOrthogonalBasis}, p::Int, index_set::Symbol=:TD; param=0.5)
        d = length(bases)
        return new(bases, p, d, multivariate_indices(p, d, index_set; param=param))
    end

    # in_index_set must take in (idx,p) and return boolean whether or not in multi-index set
    function PolynomialChaosBasis(bases::Vector{<:AbstractOrthogonalBasis}, p::Int, in_index_set::Function)
        d = length(bases)
        return new(bases, p, d, multivariate_indices(p, d, in_index_set))
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

# Total degree multi-index set (sum of polynomial degrees <= p)
function TD(idx::Vector{Int}, p::Int)
    return sum(idx) <= p
end

# Tensor product multi-index set (maximal polynomial degree <= p)
function TP(idx::Vector{Int}, p::Int)
    return maximum(idx) <= p
end

# Hyperbolic cross multi-index set (Sparse alternative to TD or TP)
function HC(idx::Vector{Int}, p::Int)
    return prod(idx .+ 1) <= (p + 1)
end

# Q-ball multi-index set (Sparse alternative to TD or TP when q < 1, same as TD when q=1, same as TP when q=Inf)
function QB(idx::Vector{Int}, p::Int, q::Real=0.5)
    @assert q > 0
    return norm(idx, q) <= p
end

function multivariate_indices(p::Int, d::Int, in_index_set::Function=TD; max_size=1_000_000_000_000)
    idx = zeros(Int, d)
    index_set = [copy(idx)]
    if p == 0
        return index_set
    end
    idx[1] += 1
    for _ in 1:max_size
        # Add to index set
        push!(index_set, copy(idx))
        # Update idx
        for i in 1:d
            idx[i] += 1
            if in_index_set(idx, p)
                break
            end
            idx[i] = 0
        end
        if all(iszero, idx)
            break
        end
    end
    return index_set
end

function multivariate_indices(p::Int, d::Int, index_set::Symbol; param=0.5)
    if index_set in (:TD, :total_degree)
        return multivariate_indices(p, d, TD)
    elseif index_set in (:TP, :total_product)
        return multivariate_indices(p, d, TP)
    elseif index_set in (:HC, :hyperbolic_cross)
        return multivariate_indices(p, d, HC)
    elseif index_set in (:QB, :q_ball)
        return multivariate_indices(p, d, (idx,p) -> QB(idx,p,param))
    else
        errstr = "Unknown index_set=$index_set, choose from following\n"
        errstr *= "(:TD, :total_degree, :TP, :total_product"
        errstr *= ", :HC, :hyperbolic_cross, :QB, :q_ball)"
        error(errstr)
    end
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
