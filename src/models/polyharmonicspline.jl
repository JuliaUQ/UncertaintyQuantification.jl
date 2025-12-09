"""
    PolyharmonicSpline(data::DataFrame, k::Int64, output::Symbol)

Creates a polyharmonic spline that is trained by given data.

#Examples
```jldoctest
julia> data = DataFrame(x = 1:10, y = [1, -5, -10, -12, -8, -1, 5, 12, 23, 50]);

julia> PolyharmonicSpline(data, 2, :y) |> DisplayAs.withcontext(:compact => true)
PolyharmonicSpline([1.14733, -0.449609, 0.0140379, -1.02859, -0.219204, 0.900367, 0.00895592, 1.07145, -5.33101, 3.88628], [-112.005, 6.84443], [1.0; 2.0; … ; 9.0; 10.0;;], 2, [:x], :y)
```
"""
struct PolyharmonicSpline <: UQModel
    w::Vector{Float64}
    v::Vector{Float64}
    c::Matrix{Float64}
    k::Int64
    n::Vector{Symbol}
    output::Symbol

    function PolyharmonicSpline(data::DataFrame, k::Int64, output::Symbol)
        f = data[:, output]

        centers = select(data, Not(output))
        n = propertynames(centers)
        centers = Matrix{Float64}(centers)

        dim = size(centers, 1)

        A = zeros(dim, dim)
        for i in 1:dim, j in 1:dim
            if i == j
                continue
            end
            r = sqrt(sum((centers[i, :] - centers[j, :]) .^ 2))
            A[i, j] = ϕ(r, k)
        end

        B = [ones(dim, 1) centers]

        M = [A B; transpose(B) zeros(size(B, 2), size(B, 2))]
        F = [f; zeros(size(B, 2))]

        wv = vec(M \ F)

        w = wv[1:dim]
        v = wv[(dim + 1):end]

        return new(w, v, centers, k, n, output)
    end
end

function ϕ(r::Float64, k::Int64)
    if k % 2 != 0
        return r^k
    elseif r < 1
        return r^(k - 1) * log(r^r)
    else
        return r^k * log(r)
    end
end

function calc(ps::PolyharmonicSpline, x::Vector{Float64})
    r = sqrt.(sum((ps.c .- transpose(x)) .^ 2; dims=2))
    f = sum(ϕ.(r, ps.k) .* ps.w)
    return f += (transpose(ps.v) * [1; x])[1]
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

    out = map(row -> calc(ps, convert(Array, row)), eachrow(x))
    return df[!, ps.output] = out
end
