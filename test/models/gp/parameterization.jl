exported_names = names(KernelFunctions; all=false)

function is_kernel_type(sym)
    obj = getfield(KernelFunctions, sym)   # get type or value
    if obj isa DataType
        return obj <: Kernel
    elseif obj isa UnionAll
        return obj.body <: Kernel
    else
        return false
    end
end

# Filter the symbols
kernel_symbols = filter(is_kernel_type, exported_names)

# Map to actual type objects
kernel_types = [getfield(KernelFunctions, sym) for sym in kernel_symbols]

function is_transform_type(sym)
    obj = getfield(KernelFunctions, sym)
    if obj isa DataType
        return obj <: Transform
    elseif obj isa UnionAll
        return obj.body <: Transform
    else
        return false
    end
end

transform_symbols = filter(is_transform_type, exported_names)
transform_types = getfield.(Ref(KernelFunctions), transform_symbols)

function check_implementation(t::Vector{Type})
    extract = map(tᵢ -> hasmethod(UncertaintyQuantification.extract_parameters, Tuple{tᵢ}), t)
    return extract
end

transform_types[.!check_implementation(transform_types)]

kernel_types[.!check_implementation(kernel_types)]