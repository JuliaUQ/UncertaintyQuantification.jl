function create_test_data(dim::Int)
    data = rand(10, dim)
    df = DataFrame()
    for i in 1:dim
        name = Symbol("x$i")
        df[!, name] = data[:, i]
    end
    return df
end

@testset "Gaussian Process" begin
    # Input samples
    n_input_samples = 10
    design = LatinHypercubeSampling(n_input_samples)

    # Use same base gp for every test
    σ² = 1e-5
    prior_gp = GP(0.0, SqExponentialKernel())
    prior_gp_noisy = with_gaussian_noise(GP(0.0, SqExponentialKernel()), σ²)

    # Possible transforms
    input_transform_choices = [
        IdentityTransformChoice, StandardNormalTransformChoice, 
        UnitRangeTransformChoice, ZScoreTransformChoice
    ]
    output_transform_choices = [
        IdentityTransformChoice, UnitRangeTransformChoice, ZScoreTransformChoice
    ]

    @testset "1D Input" begin
        @testset "Dataframe" begin
            x = collect(range(0, stop=5, length=n_input_samples))
            y = sin.(x)
            data = DataFrame(:x1 => x, :y => y)
            for input_transform in input_transform_choices, output_transform in output_transform_choices
                @testset "$input_transform → $output_transform" begin
                    if input_transform==StandardNormalTransformChoice
                        @test_throws ArgumentError GaussianProcess(
                            prior_gp, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                        @test_throws ArgumentError GaussianProcess(
                            prior_gp_noisy, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                    else
                        gp = GaussianProcess(
                            prior_gp, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                        df = create_test_data(1)
                        evaluate!(gp, df; mode=:mean_and_var)
                        @test :y_mean in propertynames(df)
                        @test :y_var in propertynames(df)

                        gp = GaussianProcess(
                            prior_gp_noisy, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                        df = create_test_data(1)
                        evaluate!(gp, df; mode=:mean_and_var)
                        @test :y_mean in propertynames(df)
                        @test :y_var in propertynames(df)
                    end
                end
            end
        end
        @testset "UQInput" begin
            xrv = [Parameter(1.5, :p), RandomVariable(Uniform(0, 5), :x1)]
            model = Model(
                df -> df.p .* sin.(df.x1), :y
            )
            for input_transform in input_transform_choices, output_transform in output_transform_choices
                @testset "$input_transform → $output_transform" begin
                    gp = GaussianProcess(
                        prior_gp, xrv, model, :y, design;
                        input_transform=input_transform(),
                        output_transform=output_transform()
                    )
                    df = create_test_data(1)
                    evaluate!(gp, df; mode=:mean_and_var)
                    @test :y_mean in propertynames(df)
                    @test :y_var in propertynames(df)

                    gp = GaussianProcess(
                        prior_gp_noisy, xrv, model, :y, design;
                        input_transform=input_transform(),
                        output_transform=output_transform()
                    )
                    df = create_test_data(1)
                    evaluate!(gp, df; mode=:mean_and_var)
                    @test :y_mean in propertynames(df)
                    @test :y_var in propertynames(df)
                end
            end
        end
    end
    @testset "2D Input" begin
        @testset "Dataframe" begin
            x = [collect(range(0, stop=5, length=n_input_samples)) collect(range(0, stop=5, length=n_input_samples))]
            y = sin.(x[:, 1]) + cos.(x[:, 2])
            data = DataFrame(:x1 => x[:, 1], :x2 => x[:, 2], :y => y)
            for input_transform in input_transform_choices, output_transform in output_transform_choices
                @testset "$input_transform → $output_transform" begin
                    if input_transform==StandardNormalTransformChoice
                        @test_throws ArgumentError GaussianProcess(
                            prior_gp, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                        @test_throws ArgumentError GaussianProcess(
                            prior_gp_noisy, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                    else
                        gp = GaussianProcess(
                            prior_gp, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                        df = create_test_data(2)
                        evaluate!(gp, df; mode=:mean_and_var)
                        @test :y_mean in propertynames(df)
                        @test :y_var in propertynames(df)

                        gp = GaussianProcess(
                            prior_gp_noisy, data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform()
                        )
                        df = create_test_data(2)
                        evaluate!(gp, df; mode=:mean_and_var)
                        @test :y_mean in propertynames(df)
                        @test :y_var in propertynames(df)
                    end
                end
            end
        end
        @testset "UQInput" begin
            xrv = [Parameter(1.5, :p), RandomVariable(Uniform(0, 5), :x1), RandomVariable(Uniform(0, 5), :x2)]
            model = Model(
                df -> df.p .* sin.(df.x1) + df.p .* cos.(df.x2), :y
            )
            for input_transform in input_transform_choices, output_transform in output_transform_choices
                @testset "$input_transform → $output_transform" begin
                    gp = GaussianProcess(
                        prior_gp, xrv, model, :y, design;
                        input_transform=input_transform(),
                        output_transform=output_transform()
                    )
                    df = create_test_data(2)
                    evaluate!(gp, df; mode=:mean_and_var)
                    @test :y_mean in propertynames(df)
                    @test :y_var in propertynames(df)

                    gp = GaussianProcess(
                        prior_gp_noisy, xrv, model, :y, design;
                        input_transform=input_transform(),
                        output_transform=output_transform()
                    )
                    df = create_test_data(2)
                    evaluate!(gp, df; mode=:mean_and_var)
                    @test :y_mean in propertynames(df)
                    @test :y_var in propertynames(df)
                end
            end
        end  
    end
end