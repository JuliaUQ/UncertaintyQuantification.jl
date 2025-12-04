struct PolynomialChaosExpansion <: UQModel
    y::Vector{Float64}
    Ψ::PolynomialChaosBasis
    output::Symbol
    inputs::Vector{<:UQInput}
end

struct LeastSquares
    sim::AbstractMonteCarlo
end

"""
    WeightedApproximateFetekePoints(sim::AbstractMonteCarlo; fadd=10, fmult=2)

Struct for performing weighted approximate Feteke points (wafp) subsampling of a Monte-Carlo sampler for use in generating a 
`PolynomialChaosExpansion`. Given a `PolynomialChaosBasis` of dimension `N`, and a Monte-Carlo sampler with `M` samples, generates
a subsample of size `max(N,min(N+fadd,N+fmult,M))` biased towards maximizing the determinant of the Gramian typically requiring less 
than `M` model evaluations. Follows procedure described in [burkEfficientSampling](@cite). 
"""
struct WeightedApproximateFetekePoints
    sim::AbstractMonteCarlo
    fadd::Integer
    fmult::Integer
    function WeightedApproximateFetekePoints(sim::AbstractMonteCarlo; fadd=10, fmult=2)
        return new(sim, fadd, fmult)
    end
end

struct GaussQuadrature end

function polynomialchaos(
    inputs::Union{<:UQInput,Vector{<:UQInput}},
    model::Union{<:UQModel,Vector{<:UQModel}},
    Ψ::PolynomialChaosBasis,
    outputs::Union{Symbol,<:AbstractVector{Symbol}},
    ls::LeastSquares,
)
    inputs = wrap(inputs)
    outputs = wrap(outputs)

    samples = sample(inputs, ls.sim)
    evaluate!(model, samples)

    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    random_names = names(random_inputs)

    to_standard_normal_space!(random_inputs, samples)
    x = map_to_bases(Ψ, Matrix(samples[:, random_names]))

    Np = length(Ψ.α)
    n = ls.sim.n
    A = Matrix{Float64}(undef, n, Np)
    for i in 1:n
        A[i,:] .= evaluate(Ψ, view(x,i,:))
    end

    pces = PolynomialChaosExpansion[]
    mses = Float64[]

    for output in outputs
        y = A \ samples[:, output]

        ϵ = samples[:, output] - A * y
        mse = mean(ϵ .^ 2)
        push!(mses, mse)
        pce = PolynomialChaosExpansion(y, Ψ, output, random_inputs)
        push!(pces, pce)
    end

    to_physical_space!(random_inputs, samples)

    if length(outputs) == 1
        return pces[1], samples, mses[1]
    else
        return pces, samples, mses
    end
end

function polynomialchaos(
    inputs::Union{<:UQInput,Vector{<:UQInput}},
    model::Union{<:UQModel,Vector{<:UQModel}},
    Ψ::PolynomialChaosBasis,
    outputs::Union{Symbol,<:AbstractVector{Symbol}},
    wafp::WeightedApproximateFetekePoints
)
    inputs = wrap(inputs)
    outputs = wrap(outputs)

    samples = sample(inputs, wafp.sim)
    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    random_names = names(random_inputs)
    to_standard_normal_space!(random_inputs, samples)
    x = map_to_bases(Ψ, Matrix(samples[:, random_names]))

    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)


    Np = length(Ψ.α)
    n = wafp.sim.n
    rest = min((Np-1) * wafp.fmult, wafp.fadd, n - Np)
    rest = max(rest, 0)
    
    A = Matrix{Float64}(undef, n, Np)
    for i in 1:n
        A[i,:] .= evaluate(Ψ, x[i,:])
    end
    w = norm.(eachrow(A)) .^ (-2.0)
    B = A' .* reshape(w .^ (1/2), 1, :)
    _, _, p = qr(B, ColumnNorm())
    pout = zeros(Int, Np + rest)
    pout[1:Np] .= p[1:Np]
    Ginv = inv(B[:,p] * B[:,p]')
    for i in 1:rest
        val, j = findmax(j -> B[:,p[j]]' * Ginv * B[:,p[j]], Np+i:n)
        pout[Np+i] = p[j]
        Ginv .-= ((Ginv * B[:,p[j]]) * (B[:,p[j]]' * Ginv)) ./ (1 + val)
    end
    
    samples = samples[pout,:]
    to_physical_space!(random_inputs, samples)
    w = w[pout]
    evaluate!(model, samples)

    A = A[pout,:]
    W = Diagonal(w)
    AtWA = factorize(A' * W * A)
    
    pces = PolynomialChaosExpansion[]
    mses = Float64[]

    for output in outputs
        y = AtWA \ (A' * W * samples[:, output])
        ϵ = samples[:, output] - A * y
        mse = mean(ϵ .^ 2)
        push!(mses, mse)
        pce = PolynomialChaosExpansion(y, Ψ, output, random_inputs)
        push!(pces, pce)
    end

    if length(outputs) == 1
        return pces[1], samples, mses[1]
    else
        return pces, samples, mses
    end
end

function polynomialchaos(
    inputs::Union{<:UQInput,Vector{<:UQInput}},
    model::Union{<:UQModel,Vector{<:UQModel}},
    Ψ::PolynomialChaosBasis,
    outputs::Union{Symbol,<:AbstractVector{Symbol}},
    _::GaussQuadrature,
)
    inputs = wrap(inputs)
    outputs = wrap(outputs)

    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    deterministic_inputs = filter(i -> isa(i, DeterministicUQInput), inputs)
    random_names = names(random_inputs)

    nodes = mapreduce(
        n -> [n...]', vcat, Iterators.product(quadrature_nodes.(Ψ.p + 1, Ψ.bases)...)
    )
    weights = map(prod, Iterators.product(quadrature_weights.(Ψ.p + 1, Ψ.bases)...))

    samples = DataFrame(map_from_bases(Ψ, nodes), random_names)
    to_physical_space!(random_inputs, samples)

    if !isempty(deterministic_inputs)
        DataFrames.hcat!(samples, sample(deterministic_inputs, size(nodes, 1)))
    end

    evaluate!(model, samples)

    pces = PolynomialChaosExpansion[]

    for output in outputs
        y = mapreduce(
            (x, w, f) -> f * w * evaluate(Ψ, collect(x)),
            +,
            eachrow(nodes),
            weights,
            samples[:, output],
        )
        pce = PolynomialChaosExpansion(y, Ψ, output, random_inputs)
        push!(pces, pce)
    end

    if length(outputs) == 1
        return pces[1], samples
    else
        return pces, samples
    end
end

function evaluate!(pce::PolynomialChaosExpansion, df::DataFrame)
    data = df[:, names(pce.inputs)]
    to_standard_normal_space!(pce.inputs, data)

    data = map_to_bases(pce.Ψ, Matrix(data))

    out = map(row -> dot(pce.y, evaluate(pce.Ψ, collect(row))), eachrow(data))
    df[!, pce.output] = out
    return nothing
end

function sample(pce::PolynomialChaosExpansion, n::Integer)
    samps = hcat(sample.(n, pce.Ψ.bases)...)
    out = map(row -> dot(pce.y, evaluate(pce.Ψ, collect(row))), eachrow(samps))

    samps = DataFrame(map_from_bases(pce.Ψ, samps), names(pce.inputs))
    to_physical_space!(pce.inputs, samps)

    samps[!, pce.output] = out
    return samps
end

mean(pce::PolynomialChaosExpansion) = pce.y[1]
var(pce::PolynomialChaosExpansion) = sum(pce.y[2:end] .^ 2)
