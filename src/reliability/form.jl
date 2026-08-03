"""
    FORM(n::Integer=10,tol::Real=1e-3,fdm::FiniteDifferencesMethod=CentralFiniteDifferences(3))

used to perform the first order reliability method using the HLRF algorithm with `n` iterations and tolerance `tol`. Gradients are estimated through `fdm`.

# References

[rackwitzStructuralReliability1978](@cite)
"""
struct FORM <: AbstractSimulation
    n::Integer
    tol::Real
    fdm::FiniteDifferencesMethod

    function FORM(
        n::Integer=10,
        tol::Real=1e-3;
        fdm::FiniteDifferencesMethod=CentralFiniteDifferences(3),
    )
        return new(n, tol, fdm)
    end
end

"""
    probability_of_failure(models::Union{Vector{<:UQModel},UQModel},performance::Function),inputs::Union{Vector{<:UQInput},UQInput},sim::FORM)

Perform a reliability analysis using the first order reliability method (FORM), see [`FORM`](@ref).
Returns the estimated probability of failure `pf`, the reliability index `β` and the design point `dp`.

## Examples
```
pf, β, dp = probability_of_failure(model, performance, inputs, sim)
```
"""
function probability_of_failure(
    models::Union{Vector{<:UQModel},UQModel},
    performance::Function,
    inputs::Union{Vector{<:UQInput},UQInput},
    sim::FORM,
)
    models, inputs = wrap.([models, inputs])

    if isimprecise(inputs, models)
        error("You must use DoubleLoop or RandomSlicing with imprecise inputs or models.")
    end

    # create reference point in standard normal space origin
    random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
    y::Vector{Float64} = zeros(length(random_names))

    G = [models..., Model(performance, :performance)]

    sns = sample(inputs, 1)

    α = Vector{Float64}(undef, length(random_names))
    β::Float64 = 0.0
    h₀::Float64 = Inf

    # compute pf through HLRF algorithm
    for it in 1:(sim.n)
        sns[1, random_names] .= y

        H = gradient_in_standard_normal_space(G, inputs, sns, :performance; fdm=sim.fdm)

        H = map(n -> H[n], random_names)

        to_physical_space!(inputs, sns)
        evaluate!(G, sns)

        h = sns[1, :performance]

        α = H ./ norm(H)

        if it == 1
            h₀ = h
        end

        β = norm(y)

        y_next = -α * (β + (h / norm(H)))
        Δ = norm(y - y_next)

        if Δ < sim.tol
            break
        end

        y = y_next
    end

    pf = cdf(Normal(), (h₀ > 0 ? -1 : 1) * β)
    dp = DataFrame(random_names .=> y)
    to_physical_space!(inputs, dp)

    dp = (; zip(random_names, collect(dp[1, :]))...)
    α = (; zip(random_names, collect(α))...)
    return pf, β, dp, α
end
