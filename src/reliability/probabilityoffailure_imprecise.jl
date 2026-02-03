
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
    models::Union{Vector{<:UQModel},UQModel},
    performance::Function,
    inputs::Union{Vector{<:UQInput},UQInput},
    dl::DoubleLoop,
)
    @assert isimprecise(inputs)

    inputs = wrap(inputs)
    imprecise_inputs = filter(x -> isimprecise(x), inputs)
    precise_inputs = filter(x -> !isimprecise(x), inputs)

    function pf_low(x)
        imprecise_inputs_x = map_to_precise_inputs(x, imprecise_inputs)
        mc_inputs = [precise_inputs..., imprecise_inputs_x...]
        mc_pf, _, _ = probability_of_failure(models, performance, mc_inputs, dl.lb)
        return mc_pf
    end

    function pf_high(x)
        imprecise_inputs_x = map_to_precise_inputs(x, imprecise_inputs)
        mc_inputs = [precise_inputs..., imprecise_inputs_x...]
        mc_pf, _, _ = probability_of_failure(models, performance, mc_inputs, dl.ub)
        return mc_pf
    end

    lb, ub = float.(bounds(inputs))
    x0 = middle.(lb, ub)

    result_lb = minimize(
        isa(dl.lb, FORM) ? OrthoMADS(length(x0)) : RobustOrthoMADS(length(x0)),
        x -> pf_low(x),
        x0;
        lowerbound=lb,
        upperbound=ub,
        min_mesh_size=1e-13,
    )

    result_ub = minimize(
        isa(dl.ub, FORM) ? OrthoMADS(length(x0)) : RobustOrthoMADS(length(x0)),
        x -> -pf_high(x),
        x0;
        lowerbound=lb,
        upperbound=ub,
        min_mesh_size=1e-13,
    )

    pf_lb = result_lb.f
    pf_ub = -result_ub.f

    if pf_lb == pf_ub
        return pf_ub
    else
        return Interval(pf_lb, pf_ub), result_lb.x, result_ub.x
    end
end

function bounds(inputs::AbstractVector{<:UQInput})
    imprecise_inputs = filter(x -> isimprecise(x), inputs)

    b = bounds.(imprecise_inputs)
    lb = vcat(getindex.(b, 1)...)
    ub = vcat(getindex.(b, 2)...)
    return lb, ub
end

function map_to_precise_inputs(x::AbstractVector, inputs::AbstractVector{<:UQInput})
    precise_inputs = UQInput[]
    params = collect(x)
    for i in inputs
        if isa(i, IntervalVariable)
            push!(precise_inputs, map_to_precise(popfirst!(params), i))
        elseif isa(i, RandomVariable{<:ProbabilityBox})
            d = count(x -> isa(x, Interval), values(i.dist.parameters))
            p = [popfirst!(params) for _ in 1:d]
            push!(precise_inputs, map_to_precise(p, i))
        elseif isa(i, JointDistribution)
            precise_marginals = map(i.m) do rv
                if isimprecise(rv)
                    d = count(x -> isa(x, Interval), values(rv.dist.parameters))
                    p = [popfirst!(params) for _ in 1:d]
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
    models::Union{Vector{<:UQModel},UQModel},
    performance::Function,
    inputs::Union{Vector{<:UQInput},UQInput},
    rs::RandomSlicing,
)
    @assert isimprecise(inputs)

    inputs = wrap(inputs)

    sns_inputs = mapreduce(transform_to_sns_input, vcat, inputs)

    models = [wrap(models)..., Model(x -> performance(x), :g_slice)]

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
