function gradient(
        models::Union{UQModel, Vector{<:UQModel}},
        inputs::Union{Symbol, Vector{Symbol}},
        df::DataFrame,
        output::Symbol;
        backend::AbstractADType = any(isa.(wrap(models), ExternalModel)) ? AutoFiniteDiff() : AutoMooncake()
    )
    models = wrap(models)
    inputs = wrap(inputs)

    constants = setdiff(propertynames(df), inputs)

    out = copy(df[:, inputs])

    function f(x, c...)
        r = DataFrame([inputs..., constants...] .=> [x..., collect(c)...])
        evaluate!(models, r)

        return only(r[1, output])
    end

    for (i, row) in enumerate(eachrow(df))
        x = collect(row[inputs])
        c = Constant.(collect(row[constants]))

        g = DifferentiationInterface.gradient(f, backend, x, c...)
        out[i, inputs] .= g
    end

    return out
end

function gradient_in_standard_normal_space(
        models::Union{UQModel, Vector{<:UQModel}},
        inputs::Union{UQInput, Vector{<:UQInput}},
        df::DataFrame,
        output::Symbol;
        backend::AbstractADType = any(isa.(wrap(models), ExternalModel)) ? AutoFiniteDiff() : AutoMooncake()
    )

    models = wrap(models)
    inputs = wrap(inputs)

    input_names = names(inputs)

    constants = setdiff(propertynames(df), input_names)

    out = copy(df[:, input_names])

    function f(x, c...)
        r = DataFrame([input_names..., constants...] .=> [x..., collect(c)...])
        to_physical_space!(inputs, r)
        evaluate!(models, r)
        return only(r[1, output])
    end

    for (i, row) in enumerate(eachrow(df))
        x = collect(row[input_names])
        c = Constant.(collect(row[constants]))
        g = DifferentiationInterface.gradient(f, backend, x, c...)
        out[i, input_names] .= g
    end

    return out
end
