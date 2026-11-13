function gradient(
        models::Union{UQModel, Vector{<:UQModel}},
        inputs::Union{Symbol, Vector{Symbol}},
        df::DataFrame,
        output::Symbol;
        backend::AbstractADType = any(isa.(wrap(models), ExternalModel)) ? AutoFiniteDiff() : AutoMooncake()
    )
    models = wrap(models)
    inputs = wrap(inputs)

    constants = setdiff(vcat(getproperty.(models, :inputs)...), inputs)

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
        models::Vector{<:UQModel},
        inputs::Vector{<:UQInput},
        reference::DataFrame,
        output::Symbol;
        fdm::FiniteDifferencesMethod = CentralFiniteDifferences(3),
    )
    samples = copy(reference)

    random_names = names(
        filter(
            i ->
            isa(i, RandomVariable{<:UnivariateDistribution}) ||
                isa(i, JointDistribution),
            inputs,
        ),
    )

    function f(x)
        samples[:, random_names] .= x
        to_physical_space!(inputs, samples)

        evaluate!(models, samples)

        return samples[:, output][1]
    end

    reference = Matrix{Float64}(samples[:, random_names])

    g = _grad(f, fdm, reference)

    return (; zip(random_names, g)...)
end

function _grad(f::Function, fdm::CentralFiniteDifferences, x)
    return grad(central_fdm(fdm.order, fdm.derivative), f, x)[1]
end

function _grad(f::Function, fdm::ForwardFiniteDifferences, x)
    return grad(forward_fdm(fdm.order, fdm.derivative), f, x)[1]
end

function _grad(f::Function, fdm::BackwardFiniteDifferences, x)
    return grad(backward_fdm(fdm.order, fdm.derivative), f, x)[1]
end
