function build_and_evaluate_gp(
    data::DataFrame,
    input::Union{Symbol, Vector{Symbol}},
    output::Symbol,
    gp::Union{AbstractGPs.GP, UncertaintyQuantification.NoisyGP};
    input_transform::UncertaintyQuantification.AbstractDataTransform=IdentityTransform(),
    output_transform::UncertaintyQuantification.AbstractDataTransform=IdentityTransform()
)
    gpr = GaussianProcess(
            gp,
            data,
            output;
            input_transform=input_transform,
            output_transform=output_transform,
            optimization=NoHyperparameterOptimization()
        )
    
    
    test_data = select(data, input)
    evaluate!(gpr, test_data)
    mean_and_var!(gpr, test_data)
    return test_data
end

function build_and_evaluate_gp(
    input::Union{UQInput, Vector{<:UQInput}},
    model::Union{UQModel, Vector{<:UQModel}},
    output::Symbol,
    gp::Union{AbstractGPs.GP, UncertaintyQuantification.NoisyGP};
    experimentaldesign::Union{AbstractMonteCarlo, AbstractDesignOfExperiments}=LatinHypercubeSampling(10),
    input_transform::UncertaintyQuantification.AbstractDataTransform=UncertaintyQuantification.IdentityTransform(),
    output_transform::UncertaintyQuantification.AbstractDataTransform=UncertaintyQuantification.IdentityTransform()
)
    Random.seed!(1337)
    gpr = GaussianProcess(
            gp,
            input,
            model,
            output,
            experimentaldesign;
            input_transform=input_transform,
            output_transform=output_transform,
            optimization=NoHyperparameterOptimization()
        )
    
    Random.seed!(42)
    test_data = sample(input, experimentaldesign)
    evaluate!(gpr, test_data)
    mean_and_var!(gpr, test_data)
    return test_data
end

@testset "GaussianProcessRegression" begin
    # Input samples
    n_input_samples = 10
    design = LatinHypercubeSampling(n_input_samples)

    # Use same base gp for every test
    σ² = 1e-5
    base_gp = GP(0.0, SqExponentialKernel())
    base_gp_noisy = with_gaussian_noise(GP(0.0, SqExponentialKernel()), σ²)

    @testset "OneDimensionalInput" begin
        # DataFrame input
        x= collect(range(0, stop=5, length=n_input_samples))
        y = sin.(x)
        data = DataFrame(:x => x, :y => y)

        # UQInput
        xrv = RandomVariable(Uniform(0, 5), :x)
        model = Model(
            df -> sin.(df.x), :y
        )

        # Test construction from DataFrame
        test_data = build_and_evaluate_gp(
            data,
            :x,
            :y,
            base_gp;
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )
        
        # evaluate! returns mean as standard
        @test all(test_data[!, :y] .== test_data[!, :y_mean])
        # outputs at trainingset should be very close
        @test all(isapprox.(test_data[!, :y], y; atol=100*eps(Float64)))
        # variance should be very close to zero as we did not use observation noise
        @test all(isapprox.(test_data[!, :y_var], 0.0; atol=100*eps(Float64)))

        test_data_noisy = build_and_evaluate_gp(
            data,
            :x,
            :y,
            base_gp_noisy;
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )

        # check if prediction variance is within 5% deviation from prescribed noise
        @test all(abs.(test_data_noisy[!, :y_var] .- σ²) .< 0.05σ²)

        # Test construction from UQInput + UQModel
        test_data_uqinput = build_and_evaluate_gp(
            xrv,
            model,
            :y,
            base_gp;
            experimentaldesign=design,
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )
        
        # evaluate! returns mean as standard
        @test all(test_data_uqinput[!, :y] .== test_data_uqinput[!, :y_mean])

        test_data_uqinput_noisy = build_and_evaluate_gp(
            xrv,
            model,
            :y,
            base_gp_noisy;
            experimentaldesign=design,
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )

        # evaluate! returns mean as standard
        @test all(test_data_uqinput_noisy[!, :y] .== test_data_uqinput_noisy[!, :y_mean])
    end

    @testset "MultiDimensionalInput" begin
        # DataFrame input
        x = [collect(range(0, stop=5, length=n_input_samples)) collect(range(0, stop=5, length=n_input_samples))]
        y = sin.(x[:, 1]) + cos.(x[:, 2])
        data = DataFrame(:x1 => x[:, 1], :x2 => x[:, 2], :y => y)

        # UQInput
        xrv = RandomVariable.([Uniform(0, 5), Uniform(0, 5)], [:x1, :x2])
        model = Model(
            df -> sin.(df.x1) + cos.(df.x2), :y
        )
        
        # Test construction from DataFrame
        test_data = build_and_evaluate_gp(
            data,
            [:x1, :x2],
            :y,
            base_gp;
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )
        
        # evaluate! returns mean as standard
        @test all(test_data[!, :y] .== test_data[!, :y_mean])
        # outputs at trainingset should be very close
        @test all(isapprox.(test_data[!, :y], y; atol=100*eps(Float64)))
        # variance should be very close to zero as we did not use observation noise
        @test all(isapprox.(test_data[!, :y_var], 0.0; atol=100*eps(Float64)))

        test_data_noisy = build_and_evaluate_gp(
            data,
            [:x1, :x2],
            :y,
            base_gp_noisy;
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )

        # check if prediction variance is within 5% deviation from prescribed noise
        @test all(abs.(test_data_noisy[!, :y_var] .- σ²) .< 0.05σ²)

        # Test construction from UQInput + UQModel
        test_data_uqinput = build_and_evaluate_gp(
            xrv,
            model,
            :y,
            base_gp;
            experimentaldesign=design,
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )
        
        # evaluate! returns mean as standard
        @test all(test_data_uqinput[!, :y] .== test_data_uqinput[!, :y_mean])

        test_data_uqinput_noisy = build_and_evaluate_gp(
            xrv,
            model,
            :y,
            base_gp_noisy;
            experimentaldesign=design,
            input_transform=UncertaintyQuantification.IdentityTransform(),
            output_transform=UncertaintyQuantification.IdentityTransform()
        )

        # evaluate! returns mean as standard
        @test all(test_data_uqinput_noisy[!, :y] .== test_data_uqinput_noisy[!, :y_mean])
    end
end