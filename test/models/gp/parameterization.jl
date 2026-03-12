function is_of_type(
    exporting_module::Module,
    name::Symbol, 
    type::DataType
)
    obj = getfield(exporting_module, name)
    if obj isa DataType
        return obj <: type
    elseif obj isa UnionAll
        return obj.body <: type
    else
        return false
    end
end

function get_exported_types(
    exporting_module::Module,
    type::DataType
)
    exported_names = names(exporting_module; all=false)
    type_symbols = filter(n -> is_of_type(exporting_module, n, type), exported_names)
    types = map(sym -> getfield(exporting_module, sym), type_symbols)
    return filter(t -> !isabstracttype(t), types)
end

check_extract_parameters(type::Type) = hasmethod(UncertaintyQuantification.extract_parameters, Tuple{type})
check_apply_parameters(type::Type) = hasmethod(UncertaintyQuantification.apply_parameters, Tuple{type, Any})
check_implementation(type::Type) = check_extract_parameters(type) && check_apply_parameters(type)

@testset "Parameterization" begin

    @testset "Mean functions" begin
        @testset "ZeroMean" begin
            m = ZeroMean()
            @test isnothing(UncertaintyQuantification.extract_parameters(m))
            @test UncertaintyQuantification.apply_parameters(m, nothing) === m
        end
        @testset "ConstMean" begin
            m = ConstMean(2.5)
            θ = UncertaintyQuantification.extract_parameters(m)
            @test θ ≈ 2.5
            @test UncertaintyQuantification.apply_parameters(m, θ).c ≈ 2.5
        end
        @testset "CustomMean" begin
            m = CustomMean(x -> sum(x))
            @test isnothing(UncertaintyQuantification.extract_parameters(m))
            @test UncertaintyQuantification.apply_parameters(m, nothing) === m
        end
        @testset "Unimplemented Means" begin
            # Check if any means exported from KernelFunctions.jl are not handled by parameterization
            meanfunctions = get_exported_types(AbstractGPs, AbstractGPs.MeanFunction)
            unimplemented_meanfunctions = filter(!check_implementation, meanfunctions)
            if !isempty(unimplemented_meanfunctions)
                @error "Mean parameter handling not implemented for:\n " * 
                    join(string.(unimplemented_meanfunctions), "\n ")
            end
            @test isempty(unimplemented_meanfunctions)
        end
    end

    @testset "Kernel functions" begin
        @testset "Kernels without parameters" begin
            no_param_types = Base.uniontypes(UncertaintyQuantification.AllWithoutParameters)
            for K in no_param_types
                k = try
                if K == FunctionTransform
                    K(x->x)
                elseif K == GibbsKernel
                    K(lengthscale=1.0)
                elseif K == PiecewisePolynomialKernel
                    K(dim=1)
                elseif K == SelectTransform
                    K([1, 2])
                else
                    K()
                end
            catch e
                @warn "Could not construct $K, skipping: $e"
            end
                @test isnothing(UncertaintyQuantification.extract_parameters(k))
                @test UncertaintyQuantification.apply_parameters(k, nothing) === k
            end
        end

        @testset "Kernels with parameters" begin
            @testset "ConstantKernel" begin
                k = ConstantKernel(; c=3.0)
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                @test only(k2.c) ≈ 3.0
            end
            @testset "ScaleTransform" begin
                t = ScaleTransform(2.0)
                θ = UncertaintyQuantification.extract_parameters(t)
                t2 = UncertaintyQuantification.apply_parameters(t, ParameterHandling.value(θ))
                @test only(t2.s) ≈ 2.0
            end
            @testset "ARDTransform" begin
                t = ARDTransform([1.0, 2.0, 3.0])
                θ = UncertaintyQuantification.extract_parameters(t)
                t2 = UncertaintyQuantification.apply_parameters(t, ParameterHandling.value(θ))
                @test t2.v ≈ [1.0, 2.0, 3.0]
            end
            @testset "PeriodicKernel" begin
                k = PeriodicKernel(; r=[1.5])
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                @test only(k2.r) ≈ 1.5
            end
            @testset "ScaledKernel" begin
                k = 2.0 * SqExponentialKernel()  # ScaledKernel
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                @test only(k2.σ²) ≈ 2.0
            end
            @testset "RationalQuadraticKernel" begin
                k = RationalQuadraticKernel(; α=1.5)
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                @test only(k2.α) ≈ 1.5
            end
        end

        @testset "Composite kernels" begin
            @testset "KernelSum" begin
                k = SqExponentialKernel() + ConstantKernel(; c=2.0)
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                # first component has no params, second has c=2.0
                @test isnothing(θ[1])
                @test only(k2.kernels[2].c) ≈ 2.0
            end
            @testset "KernelProduct" begin
                k = SqExponentialKernel() * ConstantKernel(; c=3.0)
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                @test only(k2.kernels[2].c) ≈ 3.0
            end
            @testset "TransformedKernel" begin
                k = SqExponentialKernel() ∘ ScaleTransform(2.0)
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                @test only(k2.transform.s) ≈ 2.0
            end
            @testset "Nested composite: ScaledKernel with TransformedKernel" begin
                k = 4.0 * (SqExponentialKernel() ∘ ScaleTransform(2.0))
                θ = UncertaintyQuantification.extract_parameters(k)
                k2 = UncertaintyQuantification.apply_parameters(k, ParameterHandling.value(θ))
                @test only(k2.σ²) ≈ 4.0
                @test only(k2.kernel.transform.s) ≈ 2.0
            end
        end

        @testset "Unimplemented Kernels" begin
            # Check if any kernels and transforms exported from KernelFunctions.jl are not handled by parameterization
            transforms = get_exported_types(KernelFunctions, Transform)
            unimplemented_transforms = filter(!check_implementation, transforms)
            kernels = get_exported_types(KernelFunctions, Kernel)
            unimplemented_kernels = filter(!check_implementation, kernels)

            if !isempty(unimplemented_transforms)
                @error "Transform parameter handling not implemented for:\n " * 
                    join(string.(unimplemented_transforms), "\n ")
            end
            @test isempty(unimplemented_transforms)

            if !isempty(unimplemented_kernels)
                @error "Kernel parameter handling not implemented for:\n " * 
                    join(string.(unimplemented_kernels), "\n ")
            end
            @test isempty(unimplemented_kernels)
        end
    end

    @testset "GP" begin
        @testset "GP round-trip" begin
            gp = GP(ConstMean(1.0), SqExponentialKernel())
            θ = UncertaintyQuantification.extract_parameters(gp)
            gp2 = UncertaintyQuantification.apply_parameters(gp, ParameterHandling.value(θ))
            @test only(gp2.mean.c) ≈ 1.0
            @test gp2.kernel isa SqExponentialKernel
        end
        @testset "NoisyGP round-trip" begin
            gp = with_gaussian_noise(GP(ZeroMean(), SqExponentialKernel()), 0.01)
            θ = UncertaintyQuantification.extract_parameters(gp)
            gp2 = UncertaintyQuantification.apply_parameters(gp, ParameterHandling.value(θ))
            @test gp2.σ² ≈ 0.01
            @test gp2.gp.kernel isa SqExponentialKernel
        end
        @testset "NoisyGP noise variance is constrained positive" begin
            gp = with_gaussian_noise(GP(ZeroMean(), SqExponentialKernel()), 0.01)
            _, θ = UncertaintyQuantification.parameterize(gp)
            θ_flat, unflatten = ParameterHandling.flatten(θ)
            # Constraint should keep σ² > 0
            θ_flat_negative = fill(-1000.0, length(θ_flat))
            gp2 = UncertaintyQuantification.apply_parameters(gp, ParameterHandling.value(unflatten(θ_flat_negative)))
            @test gp2.σ² > 0
        end
    end

    @testset "parameterize" begin
        @testset "model(θ) recovers original object" begin
            gp = with_gaussian_noise(GP(ConstMean(2.0), SqExponentialKernel()), 0.05)
            model, θ = UncertaintyQuantification.parameterize(gp)
            gp2 = model(θ)
            @test gp2.σ² ≈ gp.σ²
            @test gp2.gp.mean.c ≈ gp.gp.mean.c
        end
        @testset "flatten/unflatten round-trip preserves values" begin
            gp = with_gaussian_noise(GP(ConstMean(1.5), SqExponentialKernel() ∘ ScaleTransform(2.0)), 0.1)
            _, θ = UncertaintyQuantification.parameterize(gp)
            θ_flat, unflatten = ParameterHandling.flatten(θ)
            @test ParameterHandling.value(unflatten(θ_flat)) == ParameterHandling.value(θ)
        end
    end

end
