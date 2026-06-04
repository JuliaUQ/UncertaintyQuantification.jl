@testitem "Fourier transform" begin
    T0 = 10
    t = collect(range(0, T0, 1000))
    Δt = t[2] - t[1]
    x = sin.(2π * t)

    X = UncertaintyQuantification.dft(x)

    x_hat = UncertaintyQuantification.idft(X)
    x_approx = real(x_hat)

    Ŝ = Δt^2 / T0 * abs.(X) .^ 2 / (2π)
    S̃ = periodogram(x, t)
    S̄ = periodogram(x, t, false)

    @test x ≈ x_approx

    @test Ŝ ≈ S̃
    @test S̄ ≈ 2 .* Ŝ[1:ceil(Int, length(t) / 2)]
    @test S̄ ≈ 2 .* S̃[1:ceil(Int, length(t) / 2)]
end
