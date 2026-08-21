const _sobol_table_types = [Symbol[], Float64[], Float64[], Float64[], Float64[]]
const _sobol_table_header = [
    "Variables", "FirstOrder", "FirstOrderStdError", "TotalEffect", "TotalEffectStdError",
]

function sobolindices(
        models::Vector{<:UQModel},
        inputs::Vector{<:UQInput},
        outputs::Vector{Symbol},
        sim::AbstractMonteCarlo,
    )
    sim_double_samples = double_samples(sim)

    samples = sample(inputs, sim_double_samples)
    samples = samples[shuffle(1:size(samples, 1)), :]

    random_names = names(filter(i -> isa(i, RandomUQInput), inputs))

    evaluate!(models, samples)
    indices = Dict(
        [
            (name, DataFrame(_sobol_table_types, _sobol_table_header)) for name in outputs
        ]
    )

    A = samples[1:(sim.n), :]
    B = samples[(sim.n + 1):end, :]

    fA = Matrix(A[:, outputs])
    fB = Matrix(B[:, outputs])
    fA = fA .- mean(fA; dims = 1)
    fB = fB .- mean(fB; dims = 1)

    VY = var([fA; fB]; dims = 1)

    for name in random_names
        ABi = select(A, Not(name))
        ABi[:, name] = B[:, name]

        evaluate!(models, ABi)

        for (j, qty) in enumerate(outputs)
            ABi[:, qty] .-= mean(ABi[:, qty])

            # First order effects
            first_order = x -> mean(fB[:, j] .* (x .- fA[:, j])) / VY[j] # Saltelli 2009
            bs = bootstrap(first_order, ABi[:, qty], BasicSampling(1000))
            Sᵢ = first_order(ABi[:, qty])
            σSᵢ = stderror(bs)[1]

            # Total effects
            total_effect = x -> (1 / (2 * sim.n)) * sum((fA[:, j] .- x) .^ 2) / VY[j] # Saltelli 2009
            bs = bootstrap(total_effect, ABi[:, qty], BasicSampling(1000))
            Sₜ = total_effect(ABi[:, qty])
            σSₜ = stderror(bs)[1]

            push!(indices[qty], [name, Sᵢ, σSᵢ, Sₜ, σSₜ])
        end
    end
    return length(outputs) > 1 ? indices : indices[outputs[1]]
end

function sobolindices(pce::PolynomialChaosExpansion)
    random_names = names(pce.inputs)
    indices = DataFrame()
    indices.Variables = random_names
    indices.FirstOrder =
        [
        sum(pce.y[findall(α -> α[i] != 0 && sum(α) == α[i], pce.Ψ.α)] .^ 2) for
            i in 1:length(random_names)
    ] ./ var(pce)
    indices.TotalEffect =
        [
        sum(pce.y[findall(α -> α[i] != 0, pce.Ψ.α)] .^ 2) for
            i in 1:length(random_names)
    ] ./ var(pce)

    return indices
end

function sobolindices(
        models::Vector{<:UQModel},
        inputs::UQInput,
        outputs::Vector{Symbol},
        sim::AbstractMonteCarlo,
    )
    return sobolindices(models, [inputs], outputs, sim)
end

function sobolindices(
        models::Vector{<:UQModel},
        inputs::Vector{<:UQInput},
        outputs::Symbol,
        sim::AbstractMonteCarlo,
    )
    return sobolindices(models, inputs, [outputs], sim)
end

function sobolindices(
        models::UQModel, inputs::Vector{<:UQInput}, outputs::Symbol, sim::AbstractMonteCarlo
    )
    return sobolindices([models], inputs, [outputs], sim)
end

function sobolindices(
        models::Vector{<:UQModel}, inputs::UQInput, outputs::Symbol, sim::AbstractMonteCarlo
    )
    return sobolindices(models, [inputs], [outputs], sim)
end

function sobolindices(
        models::UQModel, inputs::UQInput, outputs::Symbol, sim::AbstractMonteCarlo
    )
    return sobolindices([models], [inputs], [outputs], sim)
end
