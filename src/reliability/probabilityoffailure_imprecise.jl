"""
    DoubleLoop(lb::AbstractSimulation, ub::AbstractSimulation)

Used to estimate imprecise reliability with the *double loop* Monte Carlo method. 

Wraps two simulation objects — one for lower-bound (`lb`) and one for upper-bound (`ub`).

The two simulations can differ in simulation type, complexity, or accuracy settings, since estimating the lower bound often requires more simulation effort.

This approach runs an optimisation loop over interval parameters (outer loop) and computes reliability bounds in an inner loop using the `lb` and `ub` simulation methods.

Use `DoubleLoop(sim::AbstractSimulation)` for creating a `DoubleLoop` with same simulation method for both bounds.
"""
struct DoubleLoop
    lb::AbstractSimulation
    ub::AbstractSimulation
end

"""
    DoubleLoop(sim::AbstractSimulation)

Construct a [`DoubleLoop`](@ref) where the same simulation method is used for both 
lower and upper bounds.
"""
function DoubleLoop(sim::AbstractSimulation)
    return DoubleLoop(sim, deepcopy(sim))
end

"""
    RandomSlicing(lb::AbstractSimulation, ub::AbstractSimulation)

Used to estimate imprecise reliability with *random slicing* Monte Carlo method, sometimes known as interval Monte Carlo.

Wraps two simulation objects — one for lower-bound (`lb`) and one for upper-bound (`ub`). 

The two simulations can differ in simulation type, complexity, or accuracy settings, since estimating the lower bound often requires more simulation effort.

In this approach, the `lb` and `ub` simulation methods generate random intervals from the imprecise variables. These intervals are then propagated through the model via optimisation-based interval propagation, yielding lower and upper bounds on the reliability estimate.

Use `RandomSlicing(sim::AbstractSimulation)` for creating a `RandomSlicing` with same simulation method for both bounds.

# References

[alvarez2018estimation](@cite)
"""
struct RandomSlicing
    lb::AbstractSimulation
    ub::AbstractSimulation
end

"""
    RandomSlicing(sim::AbstractSimulation)

Construct a [`RandomSlicing`](@ref) where the same simulation method is used for both 
lower and upper bounds.
"""
function RandomSlicing(sim::AbstractSimulation)
    return RandomSlicing(sim, deepcopy(sim))
end

"""
    probability_of_failure(
        models::Union{Vector{<:UQModel}, UQModel},
        performance::Function,
        inputs::Union{Vector{<:UQInput}, UQInput},
        dl::DoubleLoop
    )

Perform an **imprecise reliability analysis** using the *double loop* Monte Carlo method.

The inputs must include at least one imprecise variable.

# Returns
- **`pf_bounds`**: An [`Interval`](@ref) giving the lower and upper bounds on the probability of failure.  
- **`result_lb`**: The outputs of the reliability simulation that achieved the lower bound.  
- **`result_ub`**: The outputs of the reliability simulation that achieved the upper bound. 

If the lower and upper bounds are equal, returns only the scalar probability of failure.

See [`DoubleLoop`](@ref) for details of the random slicing configuration.
"""
function probability_of_failure(
        models::Union{Vector{<:UQModel}, UQModel},
        performance::Function,
        inputs::Union{Vector{<:UQInput}, UQInput},
        dl::DoubleLoop,
    )
    inputs, models = wrap.([inputs, models])
    @assert isimprecise(inputs, models)

    imprecise_inputs = filter(x -> isimprecise(x), inputs)
    precise_inputs = filter(x -> !isimprecise(x), inputs)

    imprecise_models = filter(m -> isimprecise(m), models)
    precise_models = filter(m -> !isimprecise(m), models)

    model_names = names(models)

    function pf_low(x)
        p = collect(x)
        mc_inputs = [precise_inputs..., map_to_precise_inputs(p, imprecise_inputs)...]

        mc_models = if !isempty(imprecise_models)
            [precise_models..., map_to_precise_models(p, imprecise_models)...]
        else
            models
        end
        if !isempty(imprecise_models)
            order_models!(model_names, mc_models)
        end

        mc_pf, _, _ = probability_of_failure(mc_models, performance, mc_inputs, dl.lb)
        return mc_pf
    end

    function pf_high(x)
        p = collect(x)
        mc_inputs = [precise_inputs..., map_to_precise_inputs(p, imprecise_inputs)...]

        mc_models = if !isempty(imprecise_models)
            [precise_models..., map_to_precise_models(p, imprecise_models)...]
        else
            models
        end
        if !isempty(imprecise_models)
            order_models!(model_names, mc_models)
        end

        mc_pf, _, _ = probability_of_failure(mc_models, performance, mc_inputs, dl.ub)
        return mc_pf
    end

    lb_in, ub_in = float.(bounds(inputs))
    lb_m, ub_m = bounds(imprecise_models)
    lb = [lb_in..., lb_m...]
    ub = [ub_in..., ub_m...]

    x0 = middle.(lb, ub)

    result_lb = minimize(
        isa(dl.lb, FORM) ? OrthoMADS(length(x0)) : RobustOrthoMADS(length(x0)),
        x -> pf_low(x),
        x0;
        lowerbound = lb,
        upperbound = ub,
        min_mesh_size = 1.0e-13,
    )

    result_ub = minimize(
        isa(dl.ub, FORM) ? OrthoMADS(length(x0)) : RobustOrthoMADS(length(x0)),
        x -> -pf_high(x),
        x0;
        lowerbound = lb,
        upperbound = ub,
        min_mesh_size = 1.0e-13,
    )

    pf = Interval(result_lb.f, -result_ub.f)
    # We only return the values of the inputs leading to the lower and upper bound of the pf
    x_lb = result_lb.x[1:length(lb_in)]
    x_ub = result_ub.x[1:length(lb_in)]

    return pf, x_lb, x_ub
end

function bounds(inputs::AbstractVector{<:UQInput})
    imprecise_inputs = filter(x -> isimprecise(x), inputs)

    b = bounds.(imprecise_inputs)
    lb = vcat(getindex.(b, 1)...)
    ub = vcat(getindex.(b, 2)...)
    return lb, ub
end

function bounds(models::AbstractVector{<:UQModel})
    imprecise_models = filter(m -> isimprecise(m), models)

    b = bounds.(imprecise_models)
    lb = vcat(getindex.(b, 1)...)
    ub = vcat(getindex.(b, 2)...)
    return lb, ub
end

function map_to_precise_inputs(x::AbstractVector, inputs::AbstractVector{<:UQInput})
    precise_inputs = UQInput[]
    for i in inputs
        if isa(i, IntervalVariable)
            push!(precise_inputs, map_to_precise(popfirst!(x), i))
        elseif isa(i, RandomVariable{<:ProbabilityBox})
            d = count(x -> isa(x, Interval), values(i.dist.parameters))
            p = [popfirst!(x) for _ in 1:d]
            push!(precise_inputs, map_to_precise(p, i))
        elseif isa(i, JointDistribution)
            precise_marginals = map(i.m) do rv
                if isimprecise(rv)
                    d = count(x -> isa(x, Interval), values(rv.dist.parameters))
                    p = [popfirst!(x) for _ in 1:d]
                    return map_to_precise(p, rv)
                else
                    return rv
                end
            end
            push!(precise_inputs, JointDistribution(i.d, precise_marginals))
        end
    end
    return precise_inputs
end

function map_to_precise_models(x::AbstractVector, models::AbstractVector{<:UQModel})
    precise_models = UQModel[]
    for m in models
        if isa(m, IntervalPredictorModel)
            lbfm = LinearBasisFunctionModel(
                m.b, [popfirst!(x) for _ in 1:length(m.b)], m.inputs, m.out
            )
            push!(precise_models, lbfm)
        end
    end
    return precise_models
end

function order_models!(names::AbstractVector{Symbol}, models::AbstractVector{<:UQModel})
    for (i, n) in enumerate(names)
        j = findfirst(m -> UncertaintyQuantification.name(m) == n, models)
        # swap models
        if i != j
            models[i], models[j] = models[j], models[i]
        end
    end
    return
end

"""
    probability_of_failure(
        models::Union{Vector{<:UQModel}, UQModel},
        performance::Function,
        inputs::Union{Vector{<:UQInput}, UQInput},
        rs::RandomSlicing
    )

Perform an **imprecise reliability analysis** using the *random slicing* Monte Carlo method

The inputs must include at least one imprecise variable.  

# Returns
- **`pf_bounds`**: An [`Interval`](@ref) giving the lower and upper bounds on the probability of failure.  
- **`result_lb`**: The outputs of the reliability simulation that achieved the lower bound.  
- **`result_ub`**: The outputs of the reliability simulation that achieved the upper bound.

See [`RandomSlicing`](@ref) for details of the random slicing configuration.
"""
function probability_of_failure(
        models::Union{Vector{<:UQModel}, UQModel},
        performance::Function,
        inputs::Union{Vector{<:UQInput}, UQInput},
        rs::RandomSlicing,
    )
    inputs, models = wrap.([inputs, models])
    @assert isimprecise(inputs, models)

    sns_inputs = mapreduce(transform_to_sns_input, vcat, inputs)

    models = [models..., Model(x -> performance(x), :g_slice)]

    sm_min = SlicingModel(models, inputs, false)

    out_ub = probability_of_failure(sm_min, df -> df.g_slice, sns_inputs, rs.ub)

    sm_max = SlicingModel(models, inputs, true)

    out_lb = probability_of_failure(sm_max, df -> df.g_slice, sns_inputs, rs.lb)

    # If sim is not FORM, transform samples back to physical space
    typeof(rs.lb) != FORM && to_physical_space!(inputs, out_lb[3])
    typeof(rs.ub) != FORM && to_physical_space!(inputs, out_ub[3])

    return Interval(out_lb[1], out_ub[1]), out_lb[2:end], out_ub[2:end]
end

function transform_to_sns_input(i::UQInput)
    if isa(i, RandomVariable)
        return RandomVariable(Normal(), i.name)
    elseif isa(i, JointDistribution)
        return RandomVariable.(Normal(), names(i))
    elseif isa(i, Parameter)
        return i
    end

    return UQInput[]
end

"""
    probability_of_failure(
        models, performance, inputs, method::CABO;
        rng = Random.default_rng(), return_diagnostics = false,
        initial_design = nothing, aleatory_design = nothing
    )

Estimate parametric imprecise failure probability, with failure defined by
`performance(samples) < 0`. Evaluate `models` on physical inputs and fit one GP
to the resulting performance values. Each adaptive iteration evaluates exactly
one new input row; the total budget is `initial_points + lower_bound_iterations +
upper_bound_iterations`. A row runs the complete model chain, so this count is not
the number of individual component-model invocations.

Returns `(pf::Interval, xmin::Vector, xmax::Vector)`, where locations follow the
epistemic ordering documented for [`CABO`](@ref). Both endpoints use the final GP
and all observations. With `return_diagnostics = true`, append a
[`CABODiagnostics`](@ref) object as the fourth return value.

The supplied RNG controls designs, GP optimization restarts and posterior draws.
Models themselves are expected to be deterministic. No quadrature reference or
knowledge of the model's analytical solution is used.

Optional `initial_design` and `aleatory_design` matrices supply unit-cube designs
instead of generating them. `initial_design` has `initial_points` rows, with random
input quantiles first (input order), followed by normalized epistemic coordinates
in the candidate-column order. `aleatory_design` has `aleatory_samples` rows and
one quantile column per random input. Aleatory quantiles must be strictly in (0,1);
epistemic coordinates may include 0 and 1. Designs must have distinct rows.
"""
function probability_of_failure(
        models::Union{Vector{<:UQModel}, UQModel},
        performance::Function,
        inputs::Union{Vector{<:UQInput}, UQInput},
        method::CABO;
        rng::AbstractRNG = Random.default_rng(),
        return_diagnostics::Bool = false,
        initial_design = nothing,
        aleatory_design = nothing,
    )
    inputs, models = wrap.([inputs, models])
    any(isimprecise, models) && throw(ArgumentError("CABO currently supports precise models only"))
    layout = _cabo_input_map(inputs)
    candidates = _cabo_candidates(layout, method)
    na = length(layout.a)
    data = if isnothing(initial_design)
        _cabo_design(rng, layout, method.initial_points)
    else
        q = _cabo_validate_design(initial_design, method.initial_points, na + length(layout.e), na)
        _cabo_design(layout, q)
    end
    aleatory = if isnothing(aleatory_design)
        _cabo_aleatory(rng, layout, method.aleatory_samples)
    else
        q = _cabo_validate_design(aleatory_design, method.aleatory_samples, na, na)
        _cabo_aleatory_table(layout, q)
    end
    _cabo_check_design(data, aleatory, candidates, layout, method)

    samples = _cabo_evaluate!(data, layout, models, performance)
    gp = _cabo_fit_gp(data, layout, method, rng)
    probabilities, sampling_jitter = _cabo_probability_draws(gp, layout, candidates, aleatory, method, rng)
    history = DataFrame(phase = Symbol[], call = Int[], candidate = Int[], expected_improvement = Float64[])

    for (phase, budget) in ((:minimum, method.lower_bound_iterations), (:maximum, method.upper_bound_iterations))
        for _ in 1:budget
            # EI selects the epistemic setting; DeviationNumber selects the aleatory point.
            ei = _cabo_expected_improvement(probabilities, phase == :minimum)
            point, index = _cabo_next_point(gp, data, layout, candidates, aleatory, ei)
            append!(samples, _cabo_evaluate!(point, layout, models, performance))
            append!(data, point)

            gp = _cabo_fit_gp(data, layout, method, rng)
            probabilities, jitter = _cabo_probability_draws(gp, layout, candidates, aleatory, method, rng)
            sampling_jitter = max(sampling_jitter, jitter)
            push!(history, (phase, nrow(data), index, ei[index]))
        end
    end

    means = vec(mean(probabilities; dims = 2))
    imin, imax = argmin(means), argmax(means)
    result = (Interval(means[imin], means[imax]), candidates[imin, :], candidates[imax, :])
    if return_diagnostics
        diagnostics = CABODiagnostics(
            gp, data, samples, nrow(data), history,
            layout.coordinates, candidates, aleatory, probabilities, sampling_jitter
        )
        return (result..., diagnostics)
    end
    return result
end
