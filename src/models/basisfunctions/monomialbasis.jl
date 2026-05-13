"""
    MonomialBasis(d::Integer, p::Integer; include_zero::Bool=true)

Construct a monomial basis in `d` dimensions of degree less than or equal to `p`.
The keyword `include_zero` controls wether the zero degree monomial is included.
"""
struct MonomialBasis <: AbstractBasis
    d::Int # dimensions
    p::Int # degree
    m::Vector{Monomial}
    function MonomialBasis(d::Integer, p::Integer; include_zero::Bool=true)
        x = ["x$i" for i in 1:d]
        m = monomials(x, p, GradedLexicographicOrder(); include_zero)
        return new(d, p, m)
    end
end

function (b::MonomialBasis)(x::AbstractVecOrMat{<:Real})
    return b.m(x)
end
