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
        indices = kucherenkoindices([model], inputs, [:y], sim)

        @test indices.FirstOrder ≈ firstorder_analytical rtol = 0.1
        @test indices.TotalEffect ≈ totaleffect_analytical rtol = 0.1
    end


    @testset "Standard Kucherenko Indices - First Order" begin
        random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
        Y_orig = Vector(model_samples[:, :y])
        total_var = var(Y_orig)
        
        S_i = zeros(length(random_names))

        for (j, i) in enumerate(random_names)
            i_cond_samples = UncertaintyQuantification._generate_conditional_samples(model_samples, inputs[1], [i])
            evaluate!(model, i_cond_samples)
            S_i[j] = UncertaintyQuantification._compute_first_order_kucherenko(model_samples, i_cond_samples, :y, total_var)
        end
        @test S_i ≈ firstorder_analytical rtol = 0.1
    end

    @testset "Stnadard Kucherenko Indices - Total Effect" begin
        random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
        Y_orig = Vector(model_samples[:, :y])
        total_var = var(Y_orig)
        
        ST_i = zeros(length(random_names))

        for (j, i) in enumerate(random_names)
            not_i_cond_samples = UncertaintyQuantification._generate_conditional_samples(model_samples, inputs[1], setdiff(random_names, [i]))
            evaluate!(model, not_i_cond_samples)
            ST_i[j] = UncertaintyQuantification._compute_total_effect_kucherenko(model_samples, not_i_cond_samples, :y, total_var)
        end
        @test ST_i ≈ totaleffect_analytical rtol = 0.1
    end

    @testset "_generate_conditional_samples" begin
        marginals = [RandomVariable(Normal(0,1), :x1), RandomVariable(Normal(0,1), :x2)]
        R = [1.0 0.5; 0.5 1.0]
        joint_dist = JointDistribution(GaussianCopula(R), marginals)
        samples_df = DataFrame(sample([joint_dist], MonteCarlo(100)))
        cond_samples = UncertaintyQuantification._generate_conditional_samples(samples_df, joint_dist, [:x1])
        @test names(cond_samples) == ["x1", "x2"]
        @test nrow(cond_samples) == 100
        @test cond_samples[!, :x1] ≈ samples_df[!, :x1]
    end

    @testset "Kucherenko Indices with bins" begin
        indices, bin_samples = kucherenkoindices_bin([model], inputs, [:y], sim; min_bin_sample_multi_dims=10)

        @test indices.FirstOrder ≈ firstorder_analytical rtol = 0.1
        @test indices.TotalEffect ≈ totaleffect_analytical rtol = 0.1
    end

    @testset "Kucherenko Indices with bins - Existing Samples" begin
        random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
        X = Matrix(model_samples[:, random_names])
        Y = Vector(model_samples[:, :y])

        indices = kucherenkoindices_bin(X, Y; min_bin_sample_multi_dims=10)
        
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

end