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

@testset "GaussianProcessParameterHandling" begin
    transforms = get_exported_types(KernelFunctions, Transform)
    unimplemented_transforms = filter(!check_implementation, transforms)

    kernels = get_exported_types(KernelFunctions, Kernel)
    unimplemented_kernels = filter(!check_implementation, kernels)

    meanfunctions = get_exported_types(AbstractGPs, AbstractGPs.MeanFunction)
    unimplemented_meanfunctions = filter(!check_implementation, meanfunctions)

    @testset "KernelFunctions.Transform" begin
        @test isempty(unimplemented_transforms) || @error "Transform parameter handling not implemented for:\n "* join(string.(unimplemented_transforms), "\n ")
    end

    @testset "KernelFunctions.Kernel" begin
        @test isempty(unimplemented_kernels) || @error "Kernel parameter handling not implemented for:\n "* join(string.(unimplemented_kernels), "\n ")
    end

    @testset "AbstractGPs.MeanFunction" begin
        @test isempty(unimplemented_meanfunctions) || @error "Meanfunction parameter handling not implemented for:\n "* join(string.(unimplemented_meanfunctions), "\n ")
    end
end
