"""
    CABO(; initial_points = 12, lower_bound_iterations = 13, upper_bound_iterations = 4,
           aleatory_samples = 1024, posterior_draws = 128,
           epistemic_candidates = nothing, grid_points = 7,
           learn_hyperparameters = true, optimizer = ..., σ² = 1e-8, jitter = 1e-10)

Collaborative and Adaptive Bayesian Optimization for imprecise failure probability.
One Gaussian process approximates the performance function in the augmented
aleatory/epistemic space. Each iteration maximizes expected improvement on failure
probability over the epistemic candidates, then uses [`DeviationNumber`](@ref) to
select one model evaluation. All observations are retained through both phases.

Supports independent continuous [`RandomVariable`](@ref)s, continuous parametric
[`ProbabilityBox`](@ref) marginals, [`IntervalVariable`](@ref)s and [`Parameter`](@ref)s.
At least one random variable and one nondegenerate epistemic interval are required.

# Keyword Arguments
- `initial_points`: Initial Latin-hypercube evaluations. Defaults to `12`.
- `lower_bound_iterations`: Additional evaluations targeting the lower bound. Defaults to `13`.
- `upper_bound_iterations`: Additional evaluations targeting the upper bound. Defaults to `4`.
- `aleatory_samples`: Size of the common shifted Sobol pool. Defaults to `1024`.
- `posterior_draws`: Correlated latent GP draws per setting. Defaults to `128`.
- `epistemic_candidates`: Optional matrix with one setting per row. Columns follow
  input order, with p-box parameters in distribution field order (e.g. `μ, σ`).
  Constant interval coordinates remain in this matrix and the returned locations.
- `grid_points`: Values per varying interval in the automatic grid, including
  endpoints. Defaults to `7`; automatic grids are limited to 256 settings.
- `learn_hyperparameters`: Fit GP hyperparameters on every iteration. Defaults to `true`.
- `optimizer`: Defaults to `MaximumLikelihoodEstimation(NelderMead(),
  Optim.Options(iterations = 250); restarts = 0)`.
- `σ²`: GP observation variance. Defaults to `1e-8`.
- `jitter`: Numerical variance for latent sampling. Defaults to `1e-10`; increased
  within a bounded range if needed, and recorded in [`CABODiagnostics`](@ref).

This implementation uses fixed budgets and a posterior-mean incumbent on the
candidate set. Endpoints are surrogate estimates, not certified continuous-domain
bounds. Dense draws cost O(aleatory_samples^3) per setting; posterior uncertainty
conditions on the integration pool and fitted hyperparameters.

# References
[hong2023collaborative](@cite)
"""
struct CABO
    initial_points::Int
    lower_bound_iterations::Int
    upper_bound_iterations::Int
    aleatory_samples::Int
    posterior_draws::Int
    epistemic_candidates::Union{Nothing, Matrix{Float64}}
    grid_points::Int
    learn_hyperparameters::Bool
    optimizer::MaximumLikelihoodEstimation
    σ²::Float64
    jitter::Float64

    function CABO(;
            initial_points::Int = 12,
            lower_bound_iterations::Int = 13,
            upper_bound_iterations::Int = 4,
            aleatory_samples::Int = 1024,
            posterior_draws::Int = 128,
            epistemic_candidates = nothing,
            grid_points::Int = 7,
            learn_hyperparameters::Bool = true,
            optimizer::MaximumLikelihoodEstimation = MaximumLikelihoodEstimation(
                NelderMead(), Optim.Options(; iterations = 250); restarts = 0
            ),
            σ²::Real = 1.0e-8,
            jitter::Real = 1.0e-10,
        )

        initial_points >= 2 || throw(ArgumentError("initial_points must be at least 2"))
        lower_bound_iterations >= 0 && upper_bound_iterations >= 0 || throw(ArgumentError("iteration budgets must be nonnegative"))
        aleatory_samples >= 2 || throw(ArgumentError("aleatory_samples must be at least 2"))
        posterior_draws >= 2 || throw(ArgumentError("posterior_draws must be at least 2"))
        grid_points >= 2 || throw(ArgumentError("grid_points must be at least 2"))
        isfinite(σ²) && σ² > 0 || throw(ArgumentError("σ² must be finite and positive"))
        isfinite(jitter) && jitter > 0 || throw(ArgumentError("jitter must be finite and positive"))

        candidates = if isnothing(epistemic_candidates)
            nothing
        elseif epistemic_candidates isa AbstractMatrix{<:Real}
            Matrix{Float64}(epistemic_candidates)
        else
            throw(ArgumentError("epistemic_candidates must be a real matrix, one candidate per row"))
        end

        return new(
            initial_points, lower_bound_iterations, upper_bound_iterations,
            aleatory_samples, posterior_draws, candidates, grid_points,
            learn_hyperparameters, optimizer, Float64(σ²), Float64(jitter)
        )
    end
end

"""
    CABODiagnostics

Returned as the fourth value of `probability_of_failure(..., CABO();
return_diagnostics = true)`. Fields:

- `gp`: final GP, whose inputs are the augmented coordinates in `training_data`.
- `training_data`: all augmented observations, with performance in `:_cabo_g`.
- `samples`: corresponding physical inputs and model outputs, in evaluation order.
- `calls`: number of evaluated input rows (one complete model chain per row).
- `history`: adaptive phase, cumulative call count, candidate index and EI.
- `epistemic_coordinates`: `(input_name, parameter_name)` for each candidate column;
  interval variables use parameter name `:value`.
- `epistemic_candidates`: matrix used for search and final reporting.
- `aleatory_samples`: fixed pool in augmented coordinates; precise random variables
  use physical values, while p-box marginals use uniform quantiles.
- `probability_draws`: final failure probabilities, candidates × posterior draws.
- `sampling_jitter`: largest numerical sampling variance used during the run.

Augmented column names `:_cabo_a1`, ... follow random-input order; `:_cabo_e1`, ...
follow `epistemic_coordinates`. Posterior draws describe uncertainty conditional
on the fixed integration pool and fitted GP hyperparameters.
"""
struct CABODiagnostics
    gp::GaussianProcess
    training_data::DataFrame
    samples::DataFrame
    calls::Int
    history::DataFrame
    epistemic_coordinates::Vector{Tuple{Symbol, Symbol}}
    epistemic_candidates::Matrix{Float64}
    aleatory_samples::DataFrame
    probability_draws::Matrix{Float64}
    sampling_jitter::Float64
end

function _cabo_parameter_names(pbox::ProbabilityBox{T}) where {T}
    return [p for p in fieldnames(T) if pbox.parameters[p] isa Interval]
end

function _cabo_input_map(inputs)
    inputs = UQInput[wrap(inputs)...]
    supported = i -> i isa Parameter || i isa IntervalVariable ||
        i isa RandomVariable{<:ContinuousUnivariateDistribution} ||
        i isa RandomVariable{<:ProbabilityBox{<:ContinuousUnivariateDistribution}}
    all(supported, inputs) || throw(
        ArgumentError(
            "CABO supports independent continuous random variables, parametric p-boxes, intervals and parameters"
        )
    )
    length(unique(names(inputs))) == length(inputs) || throw(ArgumentError("input names must be unique"))
    all(i -> !(i isa Parameter) || isfinite(i.value), inputs) ||
        throw(ArgumentError("parameter values must be finite"))
    random_inputs = filter(i -> i isa RandomVariable, inputs)
    isempty(random_inputs) && throw(ArgumentError("CABO needs at least one random variable"))
    imprecise = filter(isimprecise, inputs)
    isempty(imprecise) && throw(ArgumentError("CABO needs epistemic uncertainty in its inputs"))
    lower, upper = map(b -> Float64.(b), bounds(inputs))
    all(isfinite, lower) && all(isfinite, upper) ||
        throw(ArgumentError("CABO requires finite epistemic bounds"))
    any(lower .< upper) || throw(ArgumentError("CABO needs a nondegenerate epistemic interval"))
    coordinates = Tuple{Symbol, Symbol}[]
    for input in imprecise
        if input isa IntervalVariable
            push!(coordinates, (input.name, :value))
        else
            append!(coordinates, [(input.name, p) for p in _cabo_parameter_names(input.dist)])
        end
    end
    a = [Symbol("_cabo_a", i) for i in eachindex(random_inputs)]
    e = [Symbol("_cabo_e", i) for i in eachindex(lower)]
    # Fixed interval coordinates are reported, but are not GP dimensions.
    features = [a; e[lower .< upper]]
    return (; inputs, random_inputs, imprecise, lower, upper, coordinates, a, e, features)
end

function _cabo_candidates(layout, method)
    candidates = if isnothing(method.epistemic_candidates)
        varying = findall(layout.lower .< layout.upper)
        big(method.grid_points)^length(varying) <= 256 ||
            throw(ArgumentError("automatic CABO grid exceeds 256 settings; supply epistemic_candidates"))
        grid = full_factorial_matrix(fill(method.grid_points, length(varying)))
        settings = repeat(layout.lower', size(grid, 1))
        for (j, column) in enumerate(varying)
            lower, upper = layout.lower[column], layout.upper[column]
            settings[:, column] = clamp.(lower .+ (upper - lower) .* grid[:, j], lower, upper)
        end
        settings
    else
        copy(method.epistemic_candidates)
    end
    size(candidates, 2) == length(layout.e) || throw(ArgumentError("wrong number of epistemic candidate columns"))
    size(candidates, 1) > 0 || throw(ArgumentError("epistemic_candidates must not be empty"))
    all(isfinite, candidates) || throw(ArgumentError("epistemic_candidates must be finite"))
    all(layout.lower' .<= candidates .<= layout.upper') || throw(ArgumentError("epistemic_candidates are outside input bounds"))
    # Validate each precise distribution before spending any model evaluations.
    for row in eachrow(candidates)
        map_to_precise_inputs(collect(row), layout.imprecise)
    end
    return unique(candidates; dims = 1)
end

function _cabo_aleatory_table(layout, q)
    data = DataFrame(q, layout.a)
    for (name, input) in zip(layout.a, layout.random_inputs)
        if !(input.dist isa ProbabilityBox)
            data[!, name] = quantile.(Ref(input.dist), data[!, name])
        end
    end
    return data
end

function _cabo_validate_design(q, n, d, n_aleatory)
    q isa AbstractMatrix{<:Real} && size(q) == (n, d) ||
        throw(ArgumentError("CABO design must be a real $n × $d matrix"))
    all(isfinite, q) && all(0 .<= q .<= 1) ||
        throw(ArgumentError("CABO designs must be finite and lie in the unit cube"))
    all(0 .< q[:, 1:n_aleatory] .< 1) ||
        throw(ArgumentError("aleatory quantiles must lie strictly between 0 and 1"))
    size(unique(q; dims = 1), 1) == n || throw(ArgumentError("CABO designs must not contain duplicate rows"))
    return Matrix{Float64}(q)
end

function _cabo_design(layout, q::AbstractMatrix)
    data = _cabo_aleatory_table(layout, q[:, 1:length(layout.a)])
    for (j, name) in enumerate(layout.e)
        data[!, name] = layout.lower[j] .+ (layout.upper[j] - layout.lower[j]) .* q[:, length(layout.a) + j]
    end
    return data
end

function _cabo_design(rng, layout, n)
    q = QuasiMonteCarlo.sample(n, length(layout.a) + length(layout.e), LatinHypercubeSample(; rng = rng))
    return _cabo_design(layout, permutedims(q))
end

function _cabo_aleatory(rng, layout, n)
    d = length(layout.a)
    q = mod.(QuasiMonteCarlo.sample(n, d, SobolSample()) .+ rand(rng, d), 1)
    return _cabo_aleatory_table(layout, permutedims(clamp.(q, eps(), 1 - eps())))
end

function _cabo_to_physical(layout, augmented)
    all(isfinite, Matrix(augmented[:, [layout.a; layout.e]])) ||
        throw(ArgumentError("CABO augmented inputs must be finite"))
    physical = DataFrame()
    # Precise inputs already have their physical values in the augmented data.
    aleatory_column = Dict(i.name => c for (i, c) in zip(layout.random_inputs, layout.a))
    for input in layout.inputs
        physical[!, input.name] = if isimprecise(input)
            zeros(nrow(augmented))
        elseif input isa Parameter
            fill(input.value, nrow(augmented))
        else
            copy(augmented[!, aleatory_column[input.name]])
        end
    end
    for (i, row) in enumerate(eachrow(augmented))
        for input in map_to_precise_inputs(collect(row[layout.e]), layout.imprecise)
            physical[i, input.name] = input isa Parameter ? input.value :
                quantile(input, row[aleatory_column[input.name]])
        end
    end
    all(isfinite, Matrix(physical)) || throw(ArgumentError("CABO produced nonfinite physical inputs"))
    return physical
end

function _cabo_evaluate!(data, layout, models, performance)
    physical = _cabo_to_physical(layout, data)
    evaluate!(models, physical)
    values = performance(physical)
    values isa AbstractVector{<:Real} && length(values) == nrow(data) ||
        throw(ArgumentError("performance must return one real value per input row"))
    all(isfinite, values) || throw(ArgumentError("performance must return finite values"))
    data[!, :_cabo_g] = Float64.(values)
    return physical
end

function _cabo_fit_gp(data, layout, method, rng)
    # Reinitialize each fit: a previous optimum can be singular with added points.
    return GaussianProcess(
        data, :_cabo_g, layout.features;
        mean = ConstMean(mean(data._cabo_g)),
        kernel = max(var(data._cabo_g), 1.0) * (SqExponentialKernel() ∘ ARDTransform(ones(length(layout.features)))),
        σ² = method.σ²,
        learn_hyperparameters = method.learn_hyperparameters,
        optimizer = method.optimizer,
        rng = rng
    )
end

function _cabo_probability_draws(gp, layout, candidates, aleatory, method, rng)
    probabilities = Matrix{Float64}(undef, size(candidates, 1), method.posterior_draws)
    sampling_jitter = method.jitter
    limit = max(method.jitter, 1.0e-4 * max(var(gp.data[!, gp.output]), 1.0))
    for (i, setting) in enumerate(eachrow(candidates))
        points = crossjoin(aleatory, sample(Parameter.(setting, layout.e)))
        jitter = method.jitter
        while true
            try
                sample!(rng, gp, points, method.posterior_draws; latent = true, jitter = jitter)
                break
            catch err
                err isa PosDefException || rethrow()
                jitter * 10 <= limit || throw(
                    ArgumentError(
                        "CABO latent covariance is numerically indefinite; increase σ² or adjust the GP optimizer"
                    )
                )
                jitter *= 10
            end
        end
        sampling_jitter = max(sampling_jitter, jitter)
        for j in 1:method.posterior_draws
            probabilities[i, j] = count(draws -> draws[j] < 0, points[!, gp.output]) / nrow(points)
        end
    end
    return probabilities, sampling_jitter
end

function _cabo_next_point(gp, data, layout, candidates, aleatory, ei)
    for index in sortperm(ei; rev = true)
        pool = crossjoin(aleatory, sample(Parameter.(candidates[index, :], layout.e)))
        pool = antijoin(pool, data; on = [layout.a; layout.e])
        isempty(pool) && continue
        return _find_next_point(gp, pool, DeviationNumber()), index
    end
    throw(ArgumentError("CABO exhausted its candidate pool; increase aleatory_samples or add epistemic candidates"))
end

function _cabo_check_design(data, aleatory, candidates, layout, method)
    all(c -> std(data[!, c]) > 0, layout.features) ||
        throw(ArgumentError("initial design must vary in every nonconstant augmented coordinate"))
    nrow(unique(data)) == nrow(data) || throw(ArgumentError("initial design maps to duplicate augmented inputs"))
    aleatory_keys = Set(Tuple(row) for row in eachrow(aleatory))
    epistemic_keys = Set(Tuple(row) for row in eachrow(candidates))
    overlap = count(eachrow(data)) do row
        Tuple(row[layout.a]) in aleatory_keys && Tuple(row[layout.e]) in epistemic_keys
    end
    method.lower_bound_iterations + method.upper_bound_iterations <=
        length(aleatory_keys) * size(candidates, 1) - overlap ||
        throw(ArgumentError("CABO iteration budget exceeds the candidate pool; increase aleatory_samples or add epistemic candidates"))
    return nothing
end

function _cabo_expected_improvement(probabilities, minimize)
    means = vec(mean(probabilities; dims = 2))
    best = minimize ? minimum(means) : maximum(means)
    improvement = minimize ? best .- probabilities : probabilities .- best
    return vec(mean(max.(improvement, 0); dims = 2))
end
