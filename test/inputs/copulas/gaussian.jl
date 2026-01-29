using Random

correlation = [1 0.7071; 0.7071 1]

@testset "GaussianCopula" begin
    copula = GaussianCopula(correlation)
    @test copula.correlation == correlation
    @test dimensions(copula) == 2

    @testset "sample" begin
        u = sample(copula, 10^5)

        @test corkendall(u[:, 1], u[:, 2]) ≈ 0.5 atol = 0.01
    end

    @testset "sample_conditional_copula" begin
        marginals = [RandomVariable(Normal(0,1), :x1), RandomVariable(Normal(0,1), :x2), RandomVariable(Normal(0,1), :x3)]
        joint_dist = JointDistribution(GaussianCopula([1 0.7071 0.5; 0.7071 1 0.3; 0.5 0.3 1]), marginals)
        unconditional = sample(joint_dist, 500000)
    
        x_val = 0.5
        cond1 = sample_conditional_copula(joint_dist, [(:x1, x_val)], 10000)
        filtered1 = unconditional[abs.(unconditional[:, :x1] .- x_val) .< 0.1, :]
        @test mean(filtered1[:, :x2]) ≈ mean(cond1[:, :x2]) rtol=0.1
        @test std(filtered1[:, :x2]) ≈ std(cond1[:, :x2]) rtol=0.1
        @test mean(filtered1[:, :x3]) ≈ mean(cond1[:, :x3]) rtol=0.1
        @test std(filtered1[:, :x3]) ≈ std(cond1[:, :x3]) rtol=0.1
    
        x_val2, x_val3 = -0.5, 0.2
        cond2 = sample_conditional_copula(joint_dist, [(:x2, x_val2), (:x3, x_val3)], 10000)
        filtered2 = unconditional[(abs.(unconditional[:, :x2] .- x_val2) .< 0.15) .& (abs.(unconditional[:, :x3] .- x_val3) .< 0.15), :]
        @test mean(filtered2[:, :x1]) ≈ mean(cond2[:, :x1]) rtol=0.15
        @test std(filtered2[:, :x1]) ≈ std(cond2[:, :x1]) rtol=0.15
    end

    @testset "to_standard_normal_space" begin
        u = sample(copula, 10^5)
        s = to_standard_normal_space(copula, u)

        @test mean(s[:, 1]) ≈ 0 atol = 0.05
        @test median(s[:, 1]) ≈ 0 atol = 0.05
        @test std(s[:, 1]) ≈ 1 atol = 0.05

        @test mean(s[:, 2]) ≈ 0 atol = 0.05
        @test median(s[:, 2]) ≈ 0 atol = 0.05
        @test std(s[:, 2]) ≈ 1 atol = 0.05

        @test corkendall(s[:, 1], s[:, 2]) ≈ 0.0 atol = 0.01
    end

    @testset "to_copula_space" begin
        s = randn(10^5, 2)
        u = to_copula_space(copula, s)

        @test corkendall(u[:, 1], u[:, 2]) ≈ 0.5 atol = 0.01
    end
end
