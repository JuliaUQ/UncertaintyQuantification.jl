function make_standardizer(
    data::DataFrame, 
    input::Union{Symbol, Vector{Symbol}}, 
    output::Symbol,
    transform::UncertaintyQuantification.AbstractDataTransform,
)
    UncertaintyQuantification.DataStandardizer(
        data, input, output, 
        UncertaintyQuantification.InputTransform(transform),
        UncertaintyQuantification.OutputTransform(transform),
    )
end

function check_transform(
    data::DataFrame,
    input::Union{Symbol, Vector{Symbol}},
    output::Symbol,
    transformed_vars::Tuple,
    ::UncertaintyQuantification.IdentityTransform
)
    tranformed_in, transformed_out, 
    inv_transformed_out, inv_transformed_out_var = transformed_vars

    if isa(input, Symbol) # 1D case
        # Test input scaling
        @test all(tranformed_in .== data[!, input])
        # Transformation should not do anything here
        @test all(transformed_out .== inv_transformed_out)
        # Test output inverse transform
        @test all(data[!, output] .== inv_transformed_out)
        # Test output inverse transform for variance (should not do anything to variance)
        @test all(data[!, output] .== inv_transformed_out_var)
    else # 2D case
        # Test input scaling
        tranformed_in = mapreduce(rv -> rv', vcat, tranformed_in)
        @test all(tranformed_in .== Matrix(data[!, input]))
        # Test output scaling
        @test all(transformed_out .== inv_transformed_out)
        # Test output inverse transform
        @test all(data[!, output] .== inv_transformed_out)
        # Test output inverse transform for variance
        @test all(data[!, output] .== inv_transformed_out_var)
    end
end

function check_transform(
    data::DataFrame,
    input::Union{Symbol, Vector{Symbol}},
    output::Symbol,
    transformed_vars::Tuple,
    ::UncertaintyQuantification.ZScoreTransform
)
    tranformed_in, transformed_out, 
    inv_transformed_out, inv_transformed_out_var = transformed_vars

    if isa(input, Symbol) # 1D case
        # Test input scaling
        μ = mean(data[!, input])
        σ = std(data[!, input]) 
        manually_transformed_in = (data[!, input] .- μ) ./ σ
        @test all(tranformed_in .≈ manually_transformed_in)

        # Test output scaling
        μ = mean(data[!, output])
        σ = std(data[!, output]) 
        manually_transformed_out = (data[!, output] .- μ) ./ σ                                    
        @test all(manually_transformed_out .≈ transformed_out)

        # Test output inverse transform
        @test all(data[!, output] .≈ inv_transformed_out)

        # Test output inverse transform for variance
        @test all(σ^2 * transformed_out .≈ inv_transformed_out_var)
    else # 2D case
        # Test input scaling
        tranformed_in = mapreduce(rv -> rv', vcat, tranformed_in)
        μ = mean(Matrix(data[!, input]), dims=1)
        σ = std(Matrix(data[!, input]), dims=1) 
        manually_transformed_in = (Matrix(data[!, input]) .- μ) ./ σ
        @test all(tranformed_in .≈ manually_transformed_in)

        # Test output scaling
        μ = mean(data[!, output])
        σ = std(data[!, output]) 
        manually_transformed_out = (data[!, output] .- μ) ./ σ                                    
        @test all(manually_transformed_out .≈ transformed_out)

        # Test output inverse transform
        @test all(data[!, output] .≈ inv_transformed_out)

        # Test output inverse transform for variance
        @test all(σ^2 * transformed_out .≈ inv_transformed_out_var)
    end
end

function check_transform(
    data::DataFrame,
    input::Union{Symbol, Vector{Symbol}},
    output::Symbol,
    transformed_vars::Tuple,
    ::UncertaintyQuantification.UnitRangeTransform
)
    tranformed_in, transformed_out, 
    inv_transformed_out, inv_transformed_out_var = transformed_vars

    if isa(input, Symbol) # 1D case
        # Test input scaling
        tmin, tmax = extrema(data[!, input])
        shift = tmin
        scale = 1 / (tmax - tmin)
        manually_transformed_in = (data[!, input] .- shift) * scale
        @test all(tranformed_in .≈ manually_transformed_in)

        # Test output scaling
        tmin, tmax = extrema(data[!, output])
        shift = tmin
        scale = 1 / (tmax - tmin)
        manually_transformed_out = (data[!, output] .- shift) * scale                                   
        @test all(manually_transformed_out .≈ transformed_out)

        # Test output inverse transform
        @test all(data[!, output] .≈ inv_transformed_out)

        # Test output inverse transform for variance
        @test all(scale^2 * transformed_out .≈ inv_transformed_out_var)
    else # 2D case
        # Test input scaling
        tranformed_in = mapreduce(rv -> rv', vcat, tranformed_in)
        extrs = extrema(Matrix(data[!, input]), dims=1)
        shift = map(t -> t[1], extrs[1, :])
        scale = map(t -> 1 / (t[2] - t[1]), extrs[1, :])
        manually_transformed_in = (Matrix(data[!, input]) .- shift') .* scale'
        @test all(tranformed_in .≈ manually_transformed_in)

        # Test output scaling
        tmin, tmax = extrema(data[!, output])
        shift = tmin
        scale = 1 / (tmax - tmin)
        manually_transformed_out = (data[!, output] .- shift) * scale                                   
        @test all(manually_transformed_out .≈ transformed_out)

        # Test output inverse transform
        @test all(data[!, output] .≈ inv_transformed_out)

        # Test output inverse transform for variance
        @test all(scale^2 * transformed_out .≈ inv_transformed_out_var)
    end
end

@testset "GaussianProcessDataStandardizer" begin
    transforms = [
        UncertaintyQuantification.IdentityTransform(),
        UncertaintyQuantification.ZScoreTransform(),
        UncertaintyQuantification.UnitRangeTransform(),
        UncertaintyQuantification.StandardNormalTransform()
    ]

    N = 10
    output = :y

    @testset "OneDimensionalInput" begin
        input = RandomVariable(Normal(-1, 0.5), :x1)
        df = sample(input, N)
        df[!, output] = rand(N)
        names = propertynames(df[:, Not(output)])

        for transform in transforms
            @testset "$(nameof(typeof(transform)))" begin
                # StandardNormalTransform should not work for Outputs!
                if isa(transform, UncertaintyQuantification.StandardNormalTransform)
                    @test_throws ArgumentError datastandardizer = UncertaintyQuantification.DataStandardizer(
                            df, input, output, 
                            UncertaintyQuantification.InputTransform(transform),
                            UncertaintyQuantification.OutputTransform(transform),
                        )

                    datastandardizer = UncertaintyQuantification.DataStandardizer(
                        df, input, output, 
                        UncertaintyQuantification.InputTransform(transform),
                        UncertaintyQuantification.OutputTransform(UncertaintyQuantification.IdentityTransform()),
                    )

                    tranformed_in = datastandardizer.fᵢ(df)
                    # input gets transformed to a Vector
                    @test isa(tranformed_in, Vector)
                    # TODO: Should test if input does get transformed to standard normal space, even though this relies on already tested internal implementation.
                    continue
                end

                # Test all other transforms
                datastandardizer = make_standardizer(df, names, output, transform)
                tranformed_in = datastandardizer.fᵢ(df)
                transformed_out = datastandardizer.fₒ(df)
                inv_transformed_out = datastandardizer.fₒ⁻¹(transformed_out)
                inv_transformed_out_var = datastandardizer.var_fₒ⁻¹(transformed_out)

                check_transform(
                    df, only(names), output, 
                    (tranformed_in, transformed_out, inv_transformed_out, inv_transformed_out_var),
                    transform
                )
            end
        end
    end

    @testset "MultiDimensionalInput" begin
        input = RandomVariable.([Uniform(-2, 0), Normal(-1, 0.5), Uniform(0, 1)], [:x1, :x2, :x3])
        df = sample(input, N)
        df[!, output] = rand(N)
        names = propertynames(df[:, Not(output)])

        for transform in transforms
            @testset "$(nameof(typeof(transform)))" begin
                # StandardNormalTransform should not work for Outputs!
                if isa(transform, UncertaintyQuantification.StandardNormalTransform)
                    @test_throws ArgumentError datastandardizer = UncertaintyQuantification.DataStandardizer(
                            df, input, output, 
                            UncertaintyQuantification.InputTransform(transform),
                            UncertaintyQuantification.OutputTransform(transform),
                        )

                    datastandardizer = UncertaintyQuantification.DataStandardizer(
                        df, input, output, 
                        UncertaintyQuantification.InputTransform(transform),
                        UncertaintyQuantification.OutputTransform(UncertaintyQuantification.IdentityTransform()),
                    )

                    tranformed_in = datastandardizer.fᵢ(df)
                    # input gets transformed to RowVecs
                    @test isa(tranformed_in, RowVecs)
                    # TODO: Should test if input does get transformed to standard normal space, even though this relies on already tested internal implementation.
                    continue
                end

                # Test all other transforms
                datastandardizer = make_standardizer(df, names, output, transform)
                tranformed_in = datastandardizer.fᵢ(df)
                transformed_out = datastandardizer.fₒ(df)
                inv_transformed_out = datastandardizer.fₒ⁻¹(transformed_out)
                inv_transformed_out_var = datastandardizer.var_fₒ⁻¹(transformed_out)

                check_transform(
                    df, names, output, 
                    (tranformed_in, transformed_out, inv_transformed_out, inv_transformed_out_var),
                    transform
                )
            end
        end
    end
end
