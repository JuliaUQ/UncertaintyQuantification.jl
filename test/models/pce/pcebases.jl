@testset "PolyChaosBasis" begin
    @testset "LegendreBasis" begin
        x = range(-1, 1; length=10)

        Ψ = LegendreBasis(false)

        @test evaluate(Ψ, x, 0) == ones(10)
        @test evaluate(Ψ, x, 1) == x
        @test evaluate(Ψ, x, 2) ≈ (1 / 2) * (3x .^ 2 .- 1)
        @test evaluate(Ψ, x, 3) ≈ (1 / 2) * (5x .^ 3 .- 3x)
        @test evaluate(Ψ, x, 4) ≈ (1 / 8) * (35x .^ 4 .- 30x .^ 2 .+ 3)

        Ψ = LegendreBasis()

        @test evaluate(Ψ, x, 0) == ones(10)
        @test evaluate(Ψ, x, 1) == x .* sqrt(3)
        @test evaluate(Ψ, x, 2) ≈ (1 / 2) * (3x .^ 2 .- 1) .* sqrt(5)
        @test evaluate(Ψ, x, 3) ≈ (1 / 2) * (5x .^ 3 .- 3x) .* sqrt(7)
        @test evaluate(Ψ, x, 4) ≈ (1 / 8) * (35x .^ 4 .- 30x .^ 2 .+ 3) .* sqrt(9)

        @test quadrature_nodes(5, Ψ) ≈ [
            -0.906179845938664,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.906179845938664,
        ]
        @test quadrature_weights(5, Ψ) ≈
            [
            0.23692688505618908,
            0.47862867049936647,
            0.5688888888888889,
            0.47862867049936647,
            0.23692688505618908,
        ] ./ 2
    end

    @testset "HermiteBasis" begin
        x = range(-3, 3; length=10)

        Ψ = HermiteBasis(false)

        @test evaluate(Ψ, x, 0) == ones(10)
        @test evaluate(Ψ, x, 1) == x
        @test evaluate(Ψ, x, 2) ≈ x .^ 2 .- 1
        @test evaluate(Ψ, x, 3) ≈ x .^ 3 .- 3x
        @test evaluate(Ψ, x, 4) ≈ x .^ 4 .- 6x .^ 2 .+ 3

        Ψ = HermiteBasis()

        @test evaluate(Ψ, x, 0) == ones(10)
        @test evaluate(Ψ, x, 1) == x
        @test evaluate(Ψ, x, 2) ≈ (x .^ 2 .- 1) / sqrt(2)
        @test evaluate(Ψ, x, 3) ≈ (x .^ 3 .- 3x) / sqrt(6)
        @test evaluate(Ψ, x, 4) ≈ (x .^ 4 .- 6x .^ 2 .+ 3) / sqrt(24)

        @test quadrature_nodes(5, Ψ) ≈
            sqrt(2) .* [
            -2.0201828704560856,
            -0.9585724646138196,
            -8.881784197001252e-16,
            0.9585724646138196,
            2.0201828704560856,
        ]
        @test quadrature_weights(5, Ψ) ≈
            [
            0.019953242059045872,
            0.39361932315224074,
            0.9453087204829428,
            0.39361932315224074,
            0.019953242059045872,
        ] ./ sqrt(π)
    end

    @testset "multivariate_indices" begin
        @test multivariate_indices(2, 1) == [[0], [1], [2]]
        @test multivariate_indices(2, 2) == [
            [0, 0], [1, 0], [2, 0], 
            [0, 1], [1, 1], [0, 2]
        ]
        @test multivariate_indices(2, 3) == [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 2]
        ]
        @test multivariate_indices(2, 2, :TP) == [
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ]
        @test multivariate_indices(3, 2, :HC) == [
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0, 1], [1, 1], [0, 2], [0, 3]
        ]
        @test multivariate_indices(3, 2, :QB) == [
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0, 1], [0, 2], [0, 3]
        ]
    end
end
