function create_test_data(n_samples::Int, lower::Real, upper::Real, dim::Int)
    data = lower .+ (upper-lower) .* rand(n_samples, dim)
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
    # Lower and upper bound for inputs
    lower = 0
    upper = 5

    # Use same base gp for every test
    σ² = 1e-5
    mean_fct = ConstMean(0.0)
    kernel = SqExponentialKernel()

    # Possible transforms
    input_transform_choices = [
        IdentityTransformChoice, StandardNormalTransformChoice, 
        UnitRangeTransformChoice, ZScoreTransformChoice
    ]
    output_transform_choices = [
        IdentityTransformChoice, UnitRangeTransformChoice, ZScoreTransformChoice
    ]

    @testset "GP Optimization" begin
        @testset "Dataframe" begin
            x = collect(range(lower, stop=upper, length=n_input_samples))
            y = sin.(x)
            data = DataFrame(:x1 => x, :y => y)
            gp = GaussianProcess(
                data, :y;
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=0.0,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=false
            )
            #check if σ² was left unchanged in the optimization
            @test gp.σ² == 0.0

            gp = GaussianProcess(
                data, :y;
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=0.0,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=true
            )
            @test gp.σ² > 0.0

            gp = GaussianProcess(
                data, :y;
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=σ²,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=false
            )
            @test gp.σ² == σ²

            gp = GaussianProcess(
                data, :y;
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=σ²,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=true
            )
            @test gp.σ² != σ²

            df_mean = create_test_data(n_input_samples, lower, upper, 1)
            df_var = create_test_data(n_input_samples, lower, upper, 1)
            df_mean_var = create_test_data(n_input_samples, lower, upper, 1)
            df_samples = create_test_data(n_input_samples, lower, upper, 1)

            evaluate!(gp, df_mean; mode=:mean)
            evaluate!(gp, df_var; mode=:var)
            evaluate!(gp, df_mean_var; mode=:mean_and_var)
            evaluate!(gp, df_samples; mode=:sample, n_samples=1)
            @test :y_mean in propertynames(df_mean)
            @test !(:y_var in propertynames(df_mean))
            @test :y_var in propertynames(df_var)
            @test !(:y_mean in propertynames(df_var))
            @test :y_mean in propertynames(df_mean_var)
            @test :y_var in propertynames(df_mean_var)
            @test :y_sample_1 in propertynames(df_samples)
            @test_throws ArgumentError evaluate!(gp, df_mean; mode=:error)

            @test_throws DomainError gp = GaussianProcess(
                            data, :y;
                            input_transform=IdentityTransformChoice(),
                            output_transform=IdentityTransformChoice(),
                            σ²=-1.0,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
        end
        @testset "UQInput" begin
            xrv = [Parameter(1.5, :p), RandomVariable(Uniform(lower, upper), :x1)]
            xrv_single = RandomVariable(Uniform(lower, upper), :x1)
            model = Model(
                df -> df.p .* sin.(df.x1), :y
            )
            model_single = Model(
                df -> 1.5 .* sin.(df.x1), :y
            )

            gp = GaussianProcess(
                xrv_single, model_single, :y;
                experimental_design=design,
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=0.0,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=false
            )

            gp = GaussianProcess(
                xrv, model, :y;
                experimental_design=design,
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=0.0,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=false
            )
            @test gp.σ² == 0.0

            gp = GaussianProcess(
                xrv, model, :y;
                experimental_design=design,
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=0.0,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=true
            )
            @test gp.σ² > 0.0

            gp = GaussianProcess(
                xrv, model, :y;
                experimental_design=design,
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=σ²,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=false
            )
            @test gp.σ² == σ²

            gp = GaussianProcess(
                xrv, model, :y;
                experimental_design=design,
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=σ²,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=true
            )
            @test gp.σ² != σ²

            df_mean = sample(xrv, n_input_samples)
            df_var = sample(xrv, n_input_samples)
            df_mean_var = sample(xrv, n_input_samples)
            evaluate!(gp, df_mean; mode=:mean)
            evaluate!(gp, df_var; mode=:var)
            evaluate!(gp, df_mean_var; mode=:mean_and_var)
            @test :y_mean in propertynames(df_mean)
            @test !(:y_var in propertynames(df_mean))
            @test :y_var in propertynames(df_var)
            @test !(:y_mean in propertynames(df_var))
            @test :y_mean in propertynames(df_mean_var)
            @test :y_var in propertynames(df_mean_var)

            @test_throws DomainError gp = GaussianProcess(
                xrv, model, :y;
                experimental_design=design,
                input_transform=IdentityTransformChoice(),
                output_transform=IdentityTransformChoice(),
                σ²=-1.0,
                mean_fct=mean_fct,
                kernel=kernel,
                learn_noise=false
            )
        end
    end
    @testset "1D Input" begin
        @testset "Dataframe" begin
            x = collect(range(lower, stop=upper, length=n_input_samples))
            y = sin.(x)
            data = DataFrame(:x1 => x, :y => y)
            for input_transform in input_transform_choices, output_transform in output_transform_choices
                @testset "$input_transform → $output_transform" begin
                    if input_transform==StandardNormalTransformChoice
                        @test_throws ArgumentError GaussianProcess(
                            data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel
                        )
                    else
                        gp = GaussianProcess(
                            data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
                         #check if σ² was left unchanged in the optimization
                        @test gp.σ² == σ²
                        df = create_test_data(n_input_samples, lower, upper, 1)
                        evaluate!(gp, df; mode=:mean_and_var)
                        @test :y_mean in propertynames(df)
                        @test :y_var in propertynames(df)
                    end
                end
            end
        end
        @testset "UQInput" begin
            xrv = [Parameter(1.5, :p), RandomVariable(Uniform(lower, upper), :x1)]
            model = Model(
                df -> df.p .* sin.(df.x1), :y
            )
            for input_transform in input_transform_choices, output_transform in output_transform_choices
                @testset "$input_transform → $output_transform" begin
                    if input_transform==StandardNormalTransformChoice
                        @test_throws ArgumentError GaussianProcess(
                            xrv, model, :y;
                            experimental_design=design,
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
                    else
                        gp = GaussianProcess(
                            xrv, model, :y;
                            experimental_design=design,
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
                        df = sample(xrv, n_input_samples)
                        evaluate!(gp, df; mode=:mean_and_var)
                        @test :y_mean in propertynames(df)
                        @test :y_var in propertynames(df)
                    end
                end
            end
        end
    end
    @testset "2D Input" begin
        @testset "Dataframe" begin
            x = [collect(range(lower, stop=upper, length=n_input_samples)) collect(range(lower, stop=upper, length=n_input_samples))]
            y = sin.(x[:, 1]) + cos.(x[:, 2])
            data = DataFrame(:x1 => x[:, 1], :x2 => x[:, 2], :y => y)
            for input_transform in input_transform_choices, output_transform in output_transform_choices
                @testset "$input_transform → $output_transform" begin
                    if input_transform==StandardNormalTransformChoice
                        @test_throws ArgumentError GaussianProcess(
                            data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
                    else
                        gp = GaussianProcess(
                            data, :y;
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
                        df = create_test_data(n_input_samples, lower, upper, 2)
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
                    if input_transform==StandardNormalTransformChoice
                        @test_throws ArgumentError GaussianProcess(
                            xrv, model, :y;
                            experimental_design=design,
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
                    else
                        gp = GaussianProcess(
                            xrv, model, :y;
                            experimental_design=design,
                            input_transform=input_transform(),
                            output_transform=output_transform(),
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_noise=false
                        )
                        df = sample(xrv, n_input_samples)
                        evaluate!(gp, df; mode=:mean_and_var)
                        @test :y_mean in propertynames(df)
                        @test :y_var in propertynames(df)
                    end
                end
            end
        end  
    end
end