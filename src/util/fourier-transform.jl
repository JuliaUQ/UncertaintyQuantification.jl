function dft(x::AbstractVector)
    N = length(x)
    X = zeros(ComplexF64, N)

    for n in 1:N
        X += x[n] * exp.(-1im * 2π .* (0:(N - 1)) ./ N * (n - 1))
    end

    return X
end

function idft(X::AbstractVector)
    N = length(X)
    x = zeros(ComplexF64, N)

    for k in 1:N
        x += 1 / N * X[k] * exp.(1im * 2π * (k - 1) / N .* (0:(N - 1)))
    end

    return x
end


function periodogram(x::AbstractVector, t::AbstractVector, twosided::Bool = true)
    N = length(x)
    p = zeros(N)
    dt = t[2] - t[1]

    p = dt^2 / t[end] * abs.(dft(x)) .^ 2 / (2π)

    if twosided == false
        return 2 .* p[1:ceil(Int, N / 2)]
    end

    return p

end
