"""
    PolyharmonicSpline(data::DataFrame, k::Int64, output::Symbol)

Creates a polyharmonic spline that is trained by given data.

#Examples
```jldoctest
julia> data = DataFrame(x = 1:10, y = [1, -5, -10, -12, -8, -1, 5, 12, 23, 50]);

julia> PolyharmonicSpline(data, 2, :y)
PolyharmonicSpline([1.1473268119780484, -0.4496094703147227, 0.014037919753846809, -1.0285894238254238, -0.21920393437701655, 0.900367489486589, 0.008955916582757163, 1.071449150274845, -5.331010776968249, 3.886276317409325], [-112.00528786482374, 6.844431546357884], PolyharmonicRadialBasis([1.0 2.0 … 9.0 10.0], 2), 2, [:x], :y)
```
"""
struct PolyharmonicSpline <: UQModel
    w::Vector{Float64}
    v::Vector{Float64}
    b::PolyharmonicRadialBasis
    k::Int64
    n::Vector{Symbol}
    output::Symbol

    function PolyharmonicSpline(data::DataFrame, k::Int64, output::Symbol)
        f = data[:, output]

        centers = select(data, Not(output))
        names = propertynames(centers)
        centers = permutedims(Matrix{Float64}(centers))

        n = size(centers, 2)
        phbasis = PolyharmonicRadialBasis(centers, k)

        dim = size(centers, 1)

        A = phbasis(centers)

        B = permutedims(vcat(ones(1, n), centers))

        M = [A B; B' zeros(dim + 1, dim + 1)]

        F = [f; zeros(dim + 1)]

        wv = vec(M \ F)

        w = wv[1:n]
        v = wv[(n + 1):end]

        return new(w, v, phbasis, k, names, output)
    end
end

"""
    evaluate!(ps::PolyharmonicSpline, df::DataFrame)

Evaluate given data using a previously contructed PolyharmonicSpline metamodel.

#Examples
```jldoctest
julia> data = DataFrame(x = 1:10, y = [1, -5, -10, -12, -8, -1, 5, 12, 23, 50]);

julia> ps = PolyharmonicSpline(data, 2, :y);

julia> df = DataFrame( x = [2.5, 7.5, 12, 30]);

julia> evaluate!(ps, df);

julia> df.y |> DisplayAs.withcontext(:compact => true)
4-element Vector{Float64}:
  -7.75427
   8.29083
  84.4685
 260.437
```
"""
function evaluate!(ps::PolyharmonicSpline, df::DataFrame)
    x = Matrix{Float64}(df[:, ps.n]) # convert to matrix and order variables by ps.n

    out = map(row -> dot(ps.w, ps.b(vec(row))) + dot(ps.v, [1, vec(row)...]), eachrow(x))
    return df[!, ps.output] = out
end
