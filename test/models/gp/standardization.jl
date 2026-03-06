@testset "Standardization" begin

    @testset "to_gp_format" begin
        v = [1.0, 2.0, 3.0]
        @test UncertaintyQuantification.to_gp_format(v) isa Vector
        M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        @test UncertaintyQuantification.to_gp_format(M) isa RowVecs
    end

    @testset "Input transformer" begin
        @testset "IdentityTransformChoice" begin
            data = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_input_transform(data, :x, IdentityTransformChoice())
            x = UncertaintyQuantification.transform(data, transformer)
            @test x ≈ data.x
        end
        @testset "ZScoreTransformChoice" begin
            data = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_input_transform(data, :x, ZScoreTransformChoice())
            x = UncertaintyQuantification.transform(data, transformer)
            @test mean(x) ≈ 0.0 atol=1e-10
            @test std(x) ≈ 1.0 atol=1e-10
        end
        @testset "UnitRangeTransformChoice" begin
            data = DataFrame(x = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_input_transform(data, :x, UnitRangeTransformChoice())
            x = UncertaintyQuantification.transform(data, transformer)
            @test minimum(x) ≈ 0.0 atol=1e-10
            @test maximum(x) ≈ 1.0 atol=1e-10
        end
        @testset "StandardNormalTransformChoice with Symbol throws" begin
            data = DataFrame(x = [1.0, 2.0, 3.0])
            @test_throws ArgumentError UncertaintyQuantification.fit_input_transform(data, :x, StandardNormalTransformChoice())
        end
        @testset "Multivariate input gives RowVecs" begin
            data = DataFrame(x1 = [1.0, 2.0, 3.0], x2 = [4.0, 5.0, 6.0])
            transformer = UncertaintyQuantification.fit_input_transform(data, [:x1, :x2], IdentityTransformChoice())
            x = UncertaintyQuantification.transform(data, transformer)
            @test x isa RowVecs
        end
        @testset "Single input column gives Vector" begin
            data = DataFrame(x = [1.0, 2.0, 3.0])
            transformer = UncertaintyQuantification.fit_input_transform(data, :x, IdentityTransformChoice())
            x = UncertaintyQuantification.transform(data, transformer)
            @test x isa Vector
        end
    end

    @testset "Output transformer" begin
        @testset "IdentityTransformChoice" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, IdentityTransformChoice())
            y = UncertaintyQuantification.transform(data, transformer)
            @test y ≈ data.y
        end
        @testset "ZScoreTransformChoice" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, ZScoreTransformChoice())
            y = UncertaintyQuantification.transform(data, transformer)
            @test mean(y) ≈ 0.0 atol=1e-10
            @test std(y) ≈ 1.0 atol=1e-10
        end
        @testset "UnitRangeTransformChoice" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, UnitRangeTransformChoice())
            y = UncertaintyQuantification.transform(data, transformer)
            @test minimum(y) ≈ 0.0 atol=1e-10
            @test maximum(y) ≈ 1.0 atol=1e-10
        end
        @testset "StandardNormalTransformChoice throws" begin
            data = DataFrame(y = [1.0, 2.0, 3.0])
            @test_throws ArgumentError UncertaintyQuantification.fit_output_transform(data, :y, StandardNormalTransformChoice())
        end
    end

    @testset "Inverse transforms" begin
        @testset "NoTransform is identity" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, IdentityTransformChoice())
            y = UncertaintyQuantification.transform(data, transformer)
            @test UncertaintyQuantification.inverse_transform(y, transformer) ≈ data.y
            @test UncertaintyQuantification.variance_inverse_transform(y, transformer) ≈ y
        end
        @testset "ZScoreTransform round-trip" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, ZScoreTransformChoice())
            y_transformed = UncertaintyQuantification.transform(data, transformer)
            @test UncertaintyQuantification.inverse_transform(y_transformed, transformer) ≈ data.y atol=1e-10
        end
        @testset "UnitRangeTransform round-trip" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, UnitRangeTransformChoice())
            y_transformed = UncertaintyQuantification.transform(data, transformer)
            @test UncertaintyQuantification.inverse_transform(y_transformed, transformer) ≈ data.y atol=1e-10
        end
        @testset "Variance inverse transform - ZScore" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, ZScoreTransformChoice())
            σ² = only(transformer.transform.scale)^2
            var_transformed = ones(5)
            # Var[y] = σ² * Var[ỹ]
            @test UncertaintyQuantification.variance_inverse_transform(var_transformed, transformer) ≈ σ² * var_transformed atol=1e-10
        end
        @testset "Variance inverse transform - UnitRange" begin
            data = DataFrame(y = [1.0, 2.0, 3.0, 4.0, 5.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, UnitRangeTransformChoice())
            σ² = only(transformer.transform.scale)^2
            var_transformed = ones(5)
            @test UncertaintyQuantification.variance_inverse_transform(var_transformed, transformer) ≈ σ² * var_transformed atol=1e-10
        end
        @testset "Variance inverse is not the same as mean inverse" begin
            # Regression guard: make sure variance and mean inverse transforms are not accidentally swapped
            data = DataFrame(y = [10.0, 20.0, 30.0, 40.0, 50.0])
            transformer = UncertaintyQuantification.fit_output_transform(data, :y, ZScoreTransformChoice())
            var_transformed = ones(5)
            mean_inv = UncertaintyQuantification.inverse_transform(var_transformed, transformer)
            var_inv = UncertaintyQuantification.variance_inverse_transform(var_transformed, transformer)
            @test !(mean_inv ≈ var_inv)
        end
    end
    
end