"""
    MaximumAPosterioriBayesian(prior, optimmethod, x0; islog, lowerbounds, upperbounds)

Passed to [`bayesianupdating`](@ref) to estimate one or more maxima of the posterior distribution starting from `x0`. The optimization uses the method specified in `optimmethod`. Will calculate one estimation per point in x0. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms. `lowerbounds` and `upperbounds` specify optimization intervals.

Alternative constructors

```julia
    MaximumAPosterioriBayesian(prior, optimmethod, x0; islog) # `lowerbounds` = [-Inf], # `upperbounds` = [Inf]
    MaximumAPosterioriBayesian(prior, optimmethod, x0)  # `islog` = true
```
See also [`MaximumLikelihoodBayesian`](@ref), [`bayesianupdating `](@ref),  [`TransitionalMarkovChainMonteCarlo`](@ref).
"""
struct MaximumAPosterioriBayesian <: AbstractBayesianPointEstimate
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    x0::Vector{Vector{Float64}}
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}

    function MaximumAPosterioriBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        x0::Vector{Float64};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return MaximumAPosterioriBayesian(
            prior,
            [x0];
            islog=islog,
            lowerbounds=lowerbounds,
            upperbounds=upperbounds,
        )
    end

    function MaximumAPosterioriBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        x0::Vector{Vector{Float64}};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return new(prior, x0, islog, lowerbounds, upperbounds)
    end
end

"""
    MaximumLikelihoodBayesian(prior, x0; islog, lowerbounds, upperbounds)

Passed to [`bayesianupdating`](@ref) to estimate one or more maxima of the likelihood starting from `x0`. The optimization uses the method specified in `optimmethod`. Will calculate one estimation per point in x0. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms. `lowerbounds` and `upperbounds` specify optimization intervals.

Alternative constructors

```julia
    MaximumLikelihoodBayesian(prior, x0; islog) # `lowerbounds` = [-Inf], # `upperbounds` = [Inf]
    MaximumLikelihoodBayesian(prior, x0)  # `islog` = true
```
### Notes
The method uses `prior` only as information on which parameters are supposed to be optimized. The prior itself does not influence the result of the maximum likelihood estimate and can be given as a dummy distribution. For example, if two parameters `a` and `b` are supposed to be optimized, the prior could look like this

```julia
    prior = RandomVariable.(Uniform(0,1), [:a, :b])
```
See also [`MaximumAPosterioriBayesian`](@ref), [`bayesianupdating `](@ref),  [`TransitionalMarkovChainMonteCarlo`](@ref).
"""
struct MaximumLikelihoodBayesian <: AbstractBayesianPointEstimate

    ## !TODO Currently the prior is used to get information about model parameters, maybe there is a better way. In MLE the prior is not needed

    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    x0::Vector{Vector{Float64}}
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}

    function MaximumLikelihoodBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        x0::Vector{Float64};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return MaximumLikelihoodBayesian(
            prior,
            [x0];
            islog=islog,
            lowerbounds=lowerbounds,
            upperbounds=upperbounds,
        )
    end

    function MaximumLikelihoodBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        x0::Vector{Vector{Float64}};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return new(prior, x0, islog, lowerbounds, upperbounds)
    end
end

"""
    bayesianupdating(likelihood, models, pointestimate; prior, filtertolerance, optimizer, optimoptions)

Perform bayesian updating using the given `likelihood`, `models`  and any point estimation method [`AbstractBayesianPointEstimate`](@ref).

### Notes

Method can be called with an empty Vector of models, i.e.

    bayesianupdating(likelihood, [], pointestimate)

If `prior` is not given, the method will construct a prior distribution from the prior specified in `AbstractBayesianPointEstimate.prior`.

`likelihood` is a Julia function which must be defined in terms of a `DataFrame` of samples, and must evaluate the likelihood for each row of the `DataFrame`

For example, a loglikelihood based on normal distribution using 'Data':

```julia
likelihood(df) = [sum(logpdf.(Normal.(df_i.x, 1), Data)) for df_i in eachrow(df)]
```

If a model evaluation is required to evaluate the likelihood, a vector of `UQModel`s must be passed to `bayesianupdating`. For example if the variable `x` above is the output of a numerical model.

`filtertolerance` is a tolerance value to filter out multiple estimates of the same point. If the distance between two points is smaller than `filtertolerance`, one of them will be discarded. This is useful if the optimization method finds multiple local maxima that are very close to each other. If all points should be kept, `filtertolerance` can be set to 0.

For a general overview of the function, see [`bayesianupdating `](@ref).
"""
function bayesianupdating(
    likelihood::Function,
    models::Vector{<:UQModel},
    pointestimate::AbstractBayesianPointEstimate;
    prior::Union{Function,Nothing}=nothing,
    filtertolerance::Real=1e-6,
    optimizer::Optim.AbstractOptimizer=Optim.LBFGS(),
    optimoptions::Optim.Options=Optim.Options(),
)
    optimTarget = setupoptimizationproblem(prior, likelihood, models, pointestimate)
    result = optimize_pointestimate(optimTarget, pointestimate, optimizer, optimoptions)

    x = vcat(map(x -> push!(x.minimizer, -x.minimum), result))

    varnames = [names(pointestimate.prior)..., name(pointestimate)]

    df = DataFrame([varname => Float64[] for varname in varnames])

    foreach(row -> push!(df, row), x)

    # filter only if filtertolerance is greater than 0, otherwise keep all points
    # also helps if users set tolerance to something negative
    if filtertolerance > 0
        filterresults!(df, varnames, filtertolerance)
    end

    return df
end

# function to set up the optimization problem for MAP estimate and to generate a prior function if none is given.
function setupoptimizationproblem(
    prior::Union{Function,Nothing},
    likelihood::Function,
    models::Vector{<:UQModel},
    mapestimate::MaximumAPosterioriBayesian,
)

    # if no prior is given, generate prior function from mapestimate.prior
    if isnothing(prior)
        prior = if mapestimate.islog
            df -> vec(
                sum(
                    hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), mapestimate.prior)...);
                    dims=2,
                ),
            )
        else
            df -> vec(
                prod(
                    hcat(map(rv -> pdf.(rv.dist, df[:, rv.name]), mapestimate.prior)...);
                    dims=2,
                ),
            )
        end
    end

    target = if mapestimate.islog
        df -> prior(df) .+ likelihood(df)
    else
        df -> log.(prior(df)) .+ log.(likelihood(df))
    end

    optimTarget = x -> begin
        input = DataFrame(x', names(mapestimate.prior))

        if !isempty(models)
            evaluate!(models, input)
        end
        target(input)[1] * -1
    end

    return optimTarget
end

# function to generate the optimization problem for MLE. Note that the prior is not used, but for reasons of multiple dispatch it needs to be included
function setupoptimizationproblem(
    prior::Union{Function,Nothing},
    likelihood::Function,
    models::Vector{<:UQModel},
    mlestimate::MaximumLikelihoodBayesian,
)
    target = if mlestimate.islog
        df -> likelihood(df)
    else
        df -> log.(likelihood(df))
    end

    optimTarget = x -> begin
        input = DataFrame(x', names(mlestimate.prior))

        if !isempty(models)
            evaluate!(models, input)
        end
        target(input)[1] * -1
    end

    return optimTarget
end

# actual optimization procedure based on the point estimation method and given parameters
function optimize_pointestimate(
    optimTarget::Function, pointestimate::AbstractBayesianPointEstimate, optimizer::Optim.AbstractOptimizer, optimoptions::Optim.Options
)

    if all(isinf, pointestimate.upperbounds) && all(isinf, pointestimate.lowerbounds)
        optvalues = map(x -> optimize(optimTarget, x, optimizer, optimoptions), pointestimate.x0)
    else
        optvalues = map(
            x -> optimize(
                optimTarget,
                pointestimate.lowerbounds,
                pointestimate.upperbounds,
                x,
                Fminbox(optimizer),
                optimoptions,
            ),
            pointestimate.x0,
        )
    end
end

"""
    LaplaceEstimateBayesian(prior, optimmethod, x0; islog, lowerbounds, upperbounds)

Estimates means and covariances of a mixture of Gaussians to approximate the posterior density. Passed to [`bayesianupdating`](@ref) to estimate one or more maxima of the posterior starting from `x0`. The optimization uses the method specified in `optimmethod`. Will calculate one estimation per point in x0, these are then filtered, s.t. multiple estimates of the same point are discarded. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms. Also specifies whether the posterior is given as log-function. `lowerbounds` and `upperbounds` specify optimization intervals.

Alternative constructors

```julia
    LaplaceEstimateBayesian(prior, optimmethod, x0; islog) # `lowerbounds` = [-Inf], # `upperbounds` = [Inf]
    LaplaceEstimateBayesian(prior, optimmethod, x0)  # `islog` = true
```
### Notes
The method makes use of the [`MaximumAPosterioriBayesian`](@ref) method to estimate the maximum a posteriori (MAP) estimate, and then calculates the Hessian of the posterior at the MAP estimate to construct a Gaussian approximation of the posterior distribution.

See also [`MaximumAPosterioriBayesian`](@ref), [`bayesianupdating `](@ref),  [`TransitionalMarkovChainMonteCarlo`](@ref).
"""
struct LaplaceEstimateBayesian <: AbstractBayesianPointEstimate

    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    x0::Vector{Vector{Float64}}
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}

    function LaplaceEstimateBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        x0::Vector{Float64};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return LaplaceEstimateBayesian(
            prior,
            [x0];
            islog=islog,
            lowerbounds=lowerbounds,
            upperbounds=upperbounds,
        )
    end

    function LaplaceEstimateBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        x0::Vector{Vector{Float64}};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return new(prior, x0, islog, lowerbounds, upperbounds)
    end
end

"""
    bayesianupdating(likelihood, models, lpestimate; prior, filtertolerance, optimizer, optimoptions, adbackend)

Perform bayesian updating with Laplace estimation using the given `likelihood`, `models`  and the MAP estimation [`MaximumAPosterioriBayesian`](@ref). Laplace estimation is basically an extension of the MAP estimation, where the Hessian of the posterior is calculated at the MAP estimate and used to construct a Gaussian approximation of the posterior distribution. Returns a `JointDistribution` built from the estimated mean values and covariances.
The Hessian is estimated using a backend defined from `DiffereniationInteface.jl` and can be changed using `ADTypes`.

### Notes

Method can be called with an empty Vector of models, i.e.

    bayesianupdating(likelihood, [], pointestimate)

If `prior` is not given, the method will construct a prior distribution from the prior specified in `AbstractBayesianPointEstimate.prior`.

`likelihood` is a Julia function which must be defined in terms of a `DataFrame` of samples, and must evaluate the likelihood for each row of the `DataFrame`

For example, a loglikelihood based on normal distribution using 'Data':

```julia
likelihood(df) = [sum(logpdf.(Normal.(df_i.x, 1), Data)) for df_i in eachrow(df)]
```

If a model evaluation is required to evaluate the likelihood, a vector of `UQModel`s must be passed to `bayesianupdating`. For example if the variable `x` above is the output of a numerical model.

`filtertolerance` is a tolerance value to filter out multiple estimates of the same point. If the distance between two points is smaller than `filtertolerance`, one of them will be discarded. This is useful if the optimization method finds multiple local maxima that are very close to each other.

For a general overview of the function, see [`bayesianupdating`](@ref).
"""
function bayesianupdating(
    likelihood::Function,
    models::Vector{<:UQModel},
    lpestimate::LaplaceEstimateBayesian;
    prior::Union{Function,Nothing}=nothing,
    filtertolerance::Real=1e-6,
    optimizer::Optim.AbstractOptimizer=Optim.LBFGS(),
    optimoptions::Optim.Options=Optim.Options(),
    adbackend::ADTypes.AbstractADType=AutoFiniteDiff()
)
    mapestimate = MaximumAPosterioriBayesian(
        lpestimate.prior,
        lpestimate.x0;
        islog=lpestimate.islog,
        lowerbounds=lpestimate.lowerbounds,
        upperbounds=lpestimate.upperbounds,
    )

    optimTarget = setupoptimizationproblem(prior, likelihood, models, mapestimate)

    results = bayesianupdating(
        likelihood,
        models,
        mapestimate;
        prior=prior,
        filtertolerance=filtertolerance,
        optimizer=optimizer,
        optimoptions=optimoptions,
    )

    vars = Matrix(results[:,names(lpestimate.prior)])
    
    # `hessian` from DifferentiationInterface.jl
    hess = [inv(hessian(optimTarget, adbackend, var)) for var in eachrow(vars)]
    # the call to Hermitian is needed to tell Julia that the matrix is Hermitian. Otherwise MvNormal will complain if the (co-)variances are small.
    Σ = Hermitian.(hess)

    postvalues = lpestimate.islog ? exp.(results[:,name(mapestimate)]) : results[:,name(mapestimate)]
    weights =  postvalues ./ sum(postvalues)

    μ = Matrix(results[:, names(lpestimate.prior)])

    mixture = MixtureModel([MvNormal(μ[k, :], Σ[k]) for k in 1:size(μ, 1)], weights)
    return JointDistribution(mixture, names(lpestimate.prior))

end

# filter the DataFrame to only include the variables specified in `variables`
function filterresults!(df::DataFrame, variables::Vector{Symbol}, tolerance::Real=1e-6)
    
    filtered_df = Matrix(select(df, variables))
    n_points = size(filtered_df, 1)

    rem = falses(n_points, n_points)

    for i=1:n_points, j=1:i
        if i == j
            rem[i, j] = false
        else
            rem[i, j] = norm(filtered_df[i,:] - filtered_df[j,:]) < tolerance
        end
    end
    mask = vec(any(rem, dims=2))
    deleteat!(df, mask)
    
end

name(pe::MaximumAPosterioriBayesian) = pe.islog ? :logMAP : :MAP  
name(pe::MaximumLikelihoodBayesian) = pe.islog ? :logMLE : :MLE  