@testset "kucherenkoindices" begin
    # Testing Kucherenko indices - Test Case 1: Linear model with correlated variables from Kucherenko et al. (2012) (DOI: 10.1016/j.cpc.2011.12.020)
    ρ, σ = 0.5, 2.0
    Σ = [1.0 0.0 0.0; 0.0 1.0 ρ*σ; 0.0 ρ*σ σ^2]
    R = Σ ./ (sqrt.(diag(Σ)) * sqrt.(diag(Σ))')
    
    marginals = RandomVariable[
        RandomVariable(Normal(0, 1), :x1),
        RandomVariable(Normal(0, 1), :x2),
        RandomVariable(Normal(0, σ), :x3)
    ]

    inputs = [ JointDistribution(GaussianCopula(R), marginals) ]
    model = Model(df -> df.x1 .+ df.x2 .+ df.x3, :y)
    sim = MonteCarlo(50000)

    # Analytical calculations
    denom = 2 + σ^2 + 2*ρ*σ
    firstorder_analytical = [1 / denom, (1 + ρ*σ)^2 / denom, (σ + ρ)^2 / denom]
    totaleffect_analytical = [1 / denom, (1 - ρ^2) / denom, (σ^2 * (1 - ρ^2)) / denom]
    
    model_samples = sample(inputs, sim)
    evaluate!(model, model_samples)

    @testset "Standard Kucherenko Indices" begin
        indices = kucherenkoindices([model], inputs, :y, sim)

        @test size(indices, 1) == 3  # 3 variables
        @test names(indices) == ["Variables", "FirstOrder", "TotalEffect"]
        
        # Check that indices are close to analytical values
        @test indices.FirstOrder ≈ firstorder_analytical rtol = 0.1
        @test indices.TotalEffect ≈ totaleffect_analytical rtol = 0.1
    end

    @testset "Kucherenko Indices with multiple outputs" begin
        # Create separate models for each output
        model_y1 = Model(df -> df.x1 .+ df.x2 .+ df.x3, :y1)
        model_y2 = Model(df -> df.x1 .* df.x2, :y2)
        
        sim_small = MonteCarlo(2000)
        indices_multi = kucherenkoindices([model_y1, model_y2], inputs, [:y1, :y2], sim_small)
        
        @test isa(indices_multi, Dict)
        @test haskey(indices_multi, :y1)
        @test haskey(indices_multi, :y2)
        @test size(indices_multi[:y1], 1) == 3
        @test size(indices_multi[:y2], 1) == 3
    end

    @testset "Kucherenko Indices with bins - Existing Samples" begin
        indices = kucherenkoindices(model_samples, :y ; min_bin_sample_multi_dims=10)
        
        @test indices.FirstOrder ≈ firstorder_analytical rtol = 0.1
        @test indices.TotalEffect ≈ totaleffect_analytical rtol = 0.1
    end

    @testset "Kucherenko Indices with bins - First Order" begin
        random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
        X = Matrix(model_samples[:, random_names])
        Y = Vector(model_samples[:, :y])
        n_samples, n_vars = size(X)
        total_var = var(Y)

        S_i = zeros(n_vars)
        for i in 1:n_vars
            S_i[i] = UncertaintyQuantification._compute_first_order_kucherenko_bins(X, Y, i, min(100, floor(Int, n_samples / 25)), total_var)
        end
        @test S_i ≈ firstorder_analytical rtol = 0.1
    end

    @testset "Kucherenko Indices with bins - Total Order" begin
        random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
        X = Matrix(model_samples[:, random_names])
        Y = Vector(model_samples[:, :y])
        n_samples, n_vars = size(X)
        total_var = var(Y)

        ST_i = zeros(n_vars)
        for i in 1:n_vars
            ST_i[i] = UncertaintyQuantification._compute_total_effect_kucherenko_bins(X, Y, i, floor(Int, n_samples / 25), total_var)
        end
        @test ST_i ≈ totaleffect_analytical rtol = 0.1
    end

    @testset "_assign_multidimensional_bins" begin
        X = [1 10 100;
             2 20 110;
             3 30 120;
             4 40 130;
             5 50 140;
             6 60 150;
             7 70 160;
             8 80 170;
             9 90 180;
             10 100 190;
             11 110 200;
             12 120 210]
        num_bins = 8 
        bin_assignments = UncertaintyQuantification._assign_multidimensional_bins(X, num_bins)
        @test length(bin_assignments) == size(X, 1)
        @test all(bin_assignments .>= 1)
        @test all(bin_assignments .<= num_bins)
        @test bin_assignments[1] == bin_assignments[2]
        @test bin_assignments[end-1] == bin_assignments[end]
        @test bin_assignments[1] != bin_assignments[end]
    end

    @testset "Mixed inputs (RandomVariables + JointDistribution)" begin
        # Test case: Extract x1 from the JointDistribution as an independent RandomVariable
        # and verify that the computed indices are consistent with the pure JD case
        
        x1_indep = RandomVariable(Normal(0, 1), :x1)
        
        marginals_reduced = RandomVariable[
            RandomVariable(Normal(0, 1), :x2),
            RandomVariable(Normal(0, σ), :x3)
        ]
        R_reduced = [1.0 ρ*σ; ρ*σ σ^2]
        R_reduced = R_reduced ./ (sqrt.(diag(R_reduced)) * sqrt.(diag(R_reduced))')
        jd_reduced = JointDistribution(GaussianCopula(R_reduced), marginals_reduced)
        
        mixed_inputs = [x1_indep, jd_reduced]
        model_mixed = Model(df -> df.x1 .+ df.x2 .+ df.x3, :y)
        
        indices_mixed = kucherenkoindices([model_mixed], mixed_inputs, :y, sim)
        
        @test size(indices_mixed, 1) == 3
        @test Set(indices_mixed.Variables) == Set([:x1, :x2, :x3])
        
        x1_idx_mixed = indices_mixed[indices_mixed.Variables .== :x1, :FirstOrder][1]
        @test x1_idx_mixed ≈ firstorder_analytical[1] rtol = 0.2
        
        @test all(indices_mixed.FirstOrder .> 0)
        @test all(indices_mixed.TotalEffect .> 0)
        @test all(isfinite.(indices_mixed.FirstOrder))
        @test all(isfinite.(indices_mixed.TotalEffect))
    end

end