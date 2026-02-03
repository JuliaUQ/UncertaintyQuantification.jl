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
    optimmethod::String
    x0::Vector{Vector{Float64}}
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}
    valname::String

    function MaximumAPosterioriBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        optimmethod::String,
        x0::Vector{Float64};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return MaximumAPosterioriBayesian(
            prior,
            optimmethod,
            [x0];
            islog=islog,
            lowerbounds=lowerbounds,
            upperbounds=upperbounds,
        )
    end

    function MaximumAPosterioriBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        optimmethod::String,
        x0::Vector{Vector{Float64}};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        valname = islog ? "logMAP" : "MAP"
        return new(prior, optimmethod, x0, islog, lowerbounds, upperbounds, valname)
    end
end

"""
    MaximumLikelihoodBayesian(prior, optimmethod, x0; islog, lowerbounds, upperbounds)

Passed to [`bayesianupdating`](@ref) to estimate one or more maxima of the likelihood starting from `x0`. The optimization uses the method specified in `optimmethod`. Will calculate one estimation per point in x0. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms. `lowerbounds` and `upperbounds` specify optimization intervals.

Alternative constructors

```julia
    MaximumLikelihoodBayesian(prior, optimmethod, x0; islog) # `lowerbounds` = [-Inf], # `upperbounds` = [Inf]
    MaximumLikelihoodBayesian(prior, optimmethod, x0)  # `islog` = true
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
    optimmethod::String
    x0::Vector{Vector{Float64}}
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}
    valname::String

    function MaximumLikelihoodBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        optimmethod::String,
        x0::Vector{Float64};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return MaximumLikelihoodBayesian(
            prior,
            optimmethod,
            [x0];
            islog=islog,
            lowerbounds=lowerbounds,
            upperbounds=upperbounds,
        )
    end

    function MaximumLikelihoodBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        optimmethod::String,
        x0::Vector{Vector{Float64}};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        valname = islog ? "logMLE" : "MLE"
        return new(prior, optimmethod, x0, islog, lowerbounds, upperbounds, valname)
    end
end

"""
    bayesianupdating(likelihood, models, pointestimate; prior, filtertolerance)

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

`filtertolerance` is a tolerance value to filter out multiple estimates of the same point. If the distance between two points is smaller than `filtertolerance`, one of them will be discarded. This is useful if the optimization method finds multiple local maxima that are very close to each other.

For a general overview of the function, see [`bayesianupdating `](@ref).
"""
function bayesianupdating(
    likelihood::Function,
    models::Vector{<:UQModel},
    pointestimate::AbstractBayesianPointEstimate;
    prior::Union{Function,Nothing}=nothing,
    filtertolerance::Real=0,
)
    optimTarget = setupoptimizationproblem(prior, likelihood, models, pointestimate)
    result = optimize_pointestimate(optimTarget, pointestimate)

    x = vcat(map(x -> push!(x.minimizer, -x.minimum), result))

    names = collect(p.name for p in pointestimate.prior)
    names = push!(names, Symbol(pointestimate.valname))

    df = DataFrame([name => Float64[] for name in names])

    foreach(row -> push!(df, row), x)

    if filtertolerance > 0
        filterresults!(df, names, filtertolerance)
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
        names = collect(p.name for p in mapestimate.prior)
        input = DataFrame(x', names)

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
        names = collect(p.name for p in mlestimate.prior)
        input = DataFrame(x', names)

        if !isempty(models)
            evaluate!(models, input)
        end
        target(input)[1] * -1
    end

    return optimTarget
end

# actual optimization procedure based on the point estimation method and given parameters
function optimize_pointestimate(
    optimTarget::Function, pointestimate::AbstractBayesianPointEstimate
)
    method = getOptimMethod(pointestimate.optimmethod)

    if all(isinf, pointestimate.upperbounds) && all(isinf, pointestimate.lowerbounds)
        optvalues = map(x -> optimize(optimTarget, x, method), pointestimate.x0)
    else
        optvalues = map(
            x -> optimize(
                optimTarget,
                pointestimate.lowerbounds,
                pointestimate.upperbounds,
                x,
                Fminbox(method),
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
The method makes use of the [`MaximumAPosterioriBayesian`](@ref) method to estimate the maximum a posteriori (MAP) estimate, and then calculates the Hessian of the posterior at the MAP estimate to construct a Gaussian approximation of the posterior distribution. The Hessian currently is estimated by finite differences.

See also [`MaximumAPosterioriBayesian`](@ref), [`bayesianupdating `](@ref),  [`TransitionalMarkovChainMonteCarlo`](@ref).
"""
struct LaplaceEstimateBayesian <: AbstractBayesianPointEstimate

    prior::Vector{RandomVariable}
    optimmethod::String
    x0::Vector{Vector{Float64}}
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}

    function LaplaceEstimateBayesian(
        prior::Vector{RandomVariable},
        optimmethod::String,
        x0::Vector{Float64};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return LaplaceEstimateBayesian(
            prior,
            optimmethod,
            [x0];
            islog=islog,
            lowerbounds=lowerbounds,
            upperbounds=upperbounds,
        )
    end

    function LaplaceEstimateBayesian(
        prior::Vector{RandomVariable},
        optimmethod::String,
        x0::Vector{Vector{Float64}};
        islog::Bool=true,
        lowerbounds::Vector{Float64}=[-Inf],
        upperbounds::Vector{Float64}=[Inf],
    )
        return new(prior, optimmethod, x0, islog, lowerbounds, upperbounds)
    end
end

"""
    bayesianupdating(likelihood, models, lpestimate; prior, filtertolerance, fddist)

Perform bayesian updating with Laplace estimation using the given `likelihood`, `models`  and the MAP estimation [`MaximumAPosterioriBayesian`](@ref). Laplace estimation is basically an extension of the MAP estimation, where the Hessian of the posterior is calculated at the MAP estimate and used to construct a Gaussian approximation of the posterior distribution. Returns a `DataFrame` with the MAP estimates and estimated covariance matrices, as well as a function handle for the posterior pdf as a function of the input `DataFrame`.

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

For a general overview of the function, see [`bayesianupdating `](@ref).
"""
function bayesianupdating(
    likelihood::Function,
    models::Vector{<:UQModel},
    lpestimate::LaplaceEstimateBayesian;
    prior::Union{Function,Nothing}=nothing,
    filtertolerance::Real=1e-6,
    fddist::Float64=1e-3,
)
    mapestimate = MaximumAPosterioriBayesian(
        lpestimate.prior,
        lpestimate.optimmethod,
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
    )

    vars = Matrix(results[:,names(lpestimate.prior)])

    # !TODO use some package for this, i.e. ForwardDiff.jl, Zygote.jl, etc.
    # Could then also be flexible between AD and FD, and also could track variable names
    hess = [inv(fd_hessian(optimTarget, var, fddist)) for var in eachrow(vars)]
    # the call to Hermitian is needed to tell Julia that the matrix is Hermitian. Otherwise MvNormal will complain if the (co-)variances are small.
    results.invhessian = Hermitian.(hess)

    postvalues = lpestimate.islog ? exp.(results[:,Symbol(mapestimate.valname)]) : results[:,Symbol(mapestimate.valname)]
    weights =  postvalues ./ sum(postvalues)

    postpdf = df -> map(row -> begin
        # calculate the posterior pdf for each row in df
        means = Matrix(results[:, names(lpestimate.prior)])
        vars = collect(row[names(lpestimate.prior)])
        pdfs = [weights[i] * pdf(MvNormal(means[i, :], results.invhessian[i]), vec(vars))
                for i in 1:size(means, 1)]
        return lpestimate.islog ? log.(sum(pdfs)) : sum(pdfs)
    end, eachrow(df))

    # !TODO: use Gaussian mixture model as return value
    return results, postpdf

end

function fd_hessian(fun, x::AbstractVector, dx::Real)
    N = length(x)
    hess = zeros(N, N)

    f0 = fun(x)

    for i in 1:N
        for j in 1:i
            if i == j
                dxF = copy(x)
                dxB = copy(x)
                dxF[i] += dx
                dxB[i] -= dx

                fF = fun(dxF)
                fB = fun(dxB)

                hess[i, i] = (fF - 2*f0 + fB) / dx^2
            else
                dxFdyF = copy(x)
                dxBdyB = copy(x)
                dxFdyB = copy(x)
                dxBdyF = copy(x)

                dxFdyF[i] += dx; dxFdyF[j] += dx
                dxBdyB[i] -= dx; dxBdyB[j] -= dx
                dxFdyB[i] += dx; dxFdyB[j] -= dx
                dxBdyF[i] -= dx; dxBdyF[j] += dx

                f1F = fun(dxFdyF)
                f1B = fun(dxBdyB)
                f2F = fun(dxFdyB)
                f2B = fun(dxBdyF)

                hij = (f1F + f1B - f2F - f2B) / (4 * dx^2)
                hess[i, j] = hij
                hess[j, i] = hij
            end
        end
    end

    return hess
end
"""
    getOptimMethod(method::String)

Function to return the optimization method based on a string input. Used to reduce the amount of packages that the user needs to import. Currently supported methods are (L-)BFGS and NelderMead.
"""
function getOptimMethod(method::String)

    if method == "LBFGS"
        method = LBFGS()
    elseif method == "BFGS"
        method = BFGS()
    elseif method == "NelderMead"
        method = NelderMead()
    else
        error(
            "Optimization method $(method) is not supported in UncertaintyQuantification.jl. Currently supported methods are '(L-)BFGS' and 'NelderMead'.",
        )
    end

    return method
end

function filterresults!(df::DataFrame, variables::Vector{Symbol}, tolerance::Real=1e-6)
    # filter the DataFrame to only include the variables specified in `variables`
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