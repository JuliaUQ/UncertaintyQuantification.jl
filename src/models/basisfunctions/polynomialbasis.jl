struct PolynomialBasis
    d::Int # dimensions
    p::Int # degree
    m::Vector{Monomial}
    function PolynomialBasis(d::Integer, p::Integer; include_zero::Boolean=true)
        x = ["x$i" for i in 1:d]
        m = monomials(x, p, GradedLexicographicOrder(); include_zero)
        return new(d, p, m)
    end
end
