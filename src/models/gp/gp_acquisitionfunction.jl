#===
The type `AbstractGaussianProcessAcquisitionFunction` is used to later be able to implement
acquisition functions for other surrogates, e.g., PCE.

Some common acquisition (learning) functions for adaptive GPs for different kind of applications

### General Active Learning
Goal: Get good fit of the GP (globally)
- `MaximumVariance()` simply adds a new point based on the maximum variance
Some other learning functions are implemented based on Fuhg et al. (2021) and they provide a
tradeoff between exploitation (refine local extrema) and exploration (reduce global variance).
Specifically, there are the more exploration-based functions:
- `MaximinDistance()`
- `ExpectedImprovementForGlobalFit`
There are more learning functions that can be implemented (https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling).

### Bayesian Optimization
Goal: Refine the global minimum of the function
- `ExpectedImprovement()`
- `ProbabilityOfImprovement()`
- `UpperConfidenceBound()`

### Reliability Analysis
Goal: Refine limit state surface, `g(x) = 0`,
- `DeviationNumber()` (`U`-function from AK-MCS)
- `ExpectedFeasibility()` (`EFF` from EGRA)

### References
Fuhg, J. N., Fau, A., & Nackenhorst, U. (2021). State-of-the-Art and Comparative Review
of Adaptive Sampling Methods for Kriging. Archives of Computational Methods in Engineering,
28(4), 2689–2747. https://doi.org/10.1007/s11831-020-09474-6
===#

"""
    MaximumVariance()

Selects the candidate with the largest posterior variance.

### Reference
Sacks, J., Welch, W. J., Mitchell, T. J., & Wynn, H. P. (1989). Design and
Analysis of Computer Experiments. Statistical Science, 4(4), 409–423.
https://doi.org/10.1214/ss/1177012413
"""
struct MaximumVariance <: AbstractGaussianProcessAcquisitionFunction end

function _find_next_point(gp::GaussianProcess, candidates::DataFrame, ::MaximumVariance)
    input = propertynames(candidates)
    candidates = copy(candidates)
    _, var_col = _mean_var_cols(gp)

    evaluate!(gp, candidates; mode = :var)
    ind = argmax(candidates[:, var_col])
    return candidates[[ind], Cols(input)]
end

"""
    ExpectedImprovement(ξ=0.0)

Selects the candidate maximizing expected improvement over the current best
(minimum) observed training output. `ξ` controls the exploration/exploitation
trade-off (higher `ξ` favors exploration).

### Reference
Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient Global
Optimization of Expensive Black-Box Functions. Journal of Global
Optimization, 13(4), 455–492. https://doi.org/10.1023/A:1008306431147
"""
Base.@kwdef struct ExpectedImprovement <: AbstractGaussianProcessAcquisitionFunction
    ξ::Float64 = 0.0
end

function _find_next_point(gp::GaussianProcess, candidates::DataFrame, ei::ExpectedImprovement)
    input = propertynames(candidates)
    candidates = copy(candidates)
    mean_col, var_col = _mean_var_cols(gp)

    evaluate!(gp, candidates; mode = :mean_and_var)
    μ = candidates[:, mean_col]
    σ = sqrt.(candidates[:, var_col])
    f_best = minimum(gp.training_data[:, gp.output])

    improvement = f_best .- μ .- ei.ξ
    z = improvement ./ σ
    ei_values = improvement .* cdf.(Normal(), z) .+ σ .* pdf.(Normal(), z)
    ei_values[σ .<= 0] .= 0.0

    ind = argmax(ei_values)
    return candidates[[ind], Cols(input)]
end

"""
    ProbabilityOfImprovement(ξ=0.0)

Selects the candidate maximizing the probability of improving over the current
best (minimum) observed training output.

### Reference
Kushner, H. J. (1964). A New Method of Locating the Maximum Point of an
Arbitrary Multipeak Curve in the Presence of Noise. Journal of Basic
Engineering, 86(1), 97–106. https://doi.org/10.1115/1.3653121
"""
Base.@kwdef struct ProbabilityOfImprovement <: AbstractGaussianProcessAcquisitionFunction
    ξ::Float64 = 0.0
end

function _find_next_point(gp::GaussianProcess, candidates::DataFrame, poi::ProbabilityOfImprovement)
    input = propertynames(candidates)
    candidates = copy(candidates)
    mean_col, var_col = _mean_var_cols(gp)

    evaluate!(gp, candidates; mode = :mean_and_var)
    μ = candidates[:, mean_col]
    σ = sqrt.(candidates[:, var_col])
    f_best = minimum(gp.training_data[:, gp.output])

    z = (f_best .- μ .- poi.ξ) ./ σ
    pi_values = cdf.(Normal(), z)
    pi_values[σ .<= 0] .= 0.0

    ind = argmax(pi_values)
    return candidates[[ind], Cols(input)]
end


"""
    UpperConfidenceBound(κ=2.0)

Selects the candidate minimizing `μ(x) - κ·σ(x)` (lower confidence bound for a
minimization objective). `κ` controls the exploration weight.

### Reference
Cox, D. D., & John, S. (1992). A Statistical Method for Global Optimization.
Proceedings of the 1992 IEEE International Conference on Systems, Man, and
Cybernetics, 1241–1246. https://doi.org/10.1109/ICSMC.1992.271617
"""
Base.@kwdef struct UpperConfidenceBound <: AbstractGaussianProcessAcquisitionFunction
    κ::Float64 = 2.0
end

function _find_next_point(gp::GaussianProcess, candidates::DataFrame, ucb::UpperConfidenceBound)
    input = propertynames(candidates)
    candidates = copy(candidates)
    mean_col, var_col = _mean_var_cols(gp)

    evaluate!(gp, candidates; mode = :mean_and_var)
    μ = candidates[:, mean_col]
    σ = sqrt.(candidates[:, var_col])

    lcb = μ .- ucb.κ .* σ
    ind = argmin(lcb)
    return candidates[[ind], Cols(input)]
end

"""
    DeviationNumber(threshold=0.0)

U-function: selects the candidate minimizing `|μ(x) - threshold| / σ(x)`, i.e. the
point closest to the limit state and with the highest uncertainty.

### Reference
Echard, B., Gayton, N., & Lemaire, M. (2011). AK-MCS: An active learning
reliability method combining Kriging and Monte Carlo Simulation. Structural Safety, 33(2),
145–154. https://doi.org/10.1016/j.strusafe.2011.01.002
"""
Base.@kwdef struct DeviationNumber <: AbstractGaussianProcessAcquisitionFunction
    threshold::Float64 = 0.0
    stopping::Float64 = 2.0
end

function _find_next_point(gp::GaussianProcess, candidates::DataFrame, dn::DeviationNumber)
    next_point, _ = _find_next_point_stopping(gp, candidates, dn)
    return next_point
end

function _find_next_point_stopping(gp::GaussianProcess, candidates::DataFrame, dn::DeviationNumber)
    input = propertynames(candidates)
    candidates = copy(candidates)
    mean_col, var_col = _mean_var_cols(gp)

    evaluate!(gp, candidates; mode = :mean_and_var)
    μ = candidates[:, mean_col]
    σ = sqrt.(candidates[:, var_col])

    u = abs.(μ .- dn.threshold) ./ σ
    val, ind = findmin(u)

    return candidates[[ind], Cols(input)], (val >= dn.stopping)
end

function _mean_var_cols(gp::GaussianProcess)
    return Symbol(gp.output, "_mean"), Symbol(gp.output, "_var")
end

"""
    ExpectedFeasibility(threshold=0.0, epsilon_factor=2.0)

Selects the candidate with the largest expected feasibility function (EFF)
near the limit state `G(x) = threshold`. The feasibility half-width is set
per candidate as `epsilon_factor * σ(x)`.

### Reference
Bichon, B. J., Eldred, M. S., Swiler, L. P., Mahadevan, S., & McFarland,
J. M. (2008). Efficient Global Reliability Analysis for Nonlinear Implicit
Performance Functions. AIAA Journal, 46(10), 2459-2468.
https://doi.org/10.2514/1.34321
"""
Base.@kwdef struct ExpectedFeasibility <: AbstractGaussianProcessAcquisitionFunction
    threshold::Float64 = 0.0
    epsilon_factor::Float64 = 2.0
    stopping::Float64 = 0.001
end

function _find_next_point(
        gp::GaussianProcess,
        candidates::DataFrame,
        eff::ExpectedFeasibility,
    )
    next_point, _ = _find_next_point_stopping(gp, candidates, eff)
    return next_point
end

function _find_next_point_stopping(
        gp::GaussianProcess,
        candidates::DataFrame,
        eff::ExpectedFeasibility,
    )
    input = propertynames(candidates)
    candidates = copy(candidates)
    mean_col, var_col = _mean_var_cols(gp)

    evaluate!(gp, candidates; mode = :mean_and_var)
    μ = candidates[:, mean_col]
    σ = sqrt.(candidates[:, var_col])

    eff_values = zeros(length(μ))
    positive_σ = σ .> 0

    μ_active = μ[positive_σ]
    σ_active = σ[positive_σ]
    ε = eff.epsilon_factor .* σ_active

    z = (eff.threshold .- μ_active) ./ σ_active
    z_lower = (eff.threshold .- ε .- μ_active) ./ σ_active
    z_upper = (eff.threshold .+ ε .- μ_active) ./ σ_active

    eff_values[positive_σ] =
        (μ_active .- eff.threshold) .* (
        2 .* cdf.(Normal(), z) .-
            cdf.(Normal(), z_lower) .-
            cdf.(Normal(), z_upper)
    ) .-
        σ_active .* (
        2 .* pdf.(Normal(), z) .-
            pdf.(Normal(), z_lower) .-
            pdf.(Normal(), z_upper)
    ) .+
        ε .* (
        cdf.(Normal(), z_upper) .-
            cdf.(Normal(), z_lower)
    )

    val, ind = findmax(eff_values)

    return candidates[[ind], Cols(input)], (val <= eff.stopping)
end

"""
    MaximinDistance()

Space-filling score that selects the candidate farthest from every
existing training point (maximizes the nearest-neighbor distance).

### Reference
Johnson, M. E., Moore, L. M., & Ylvisaker, D. (1990). Minimax and Maximin
Distance Designs. Journal of Statistical Planning and Inference, 26(2),
131–148. https://doi.org/10.1016/0378-3758(90)90122-B
"""
struct MaximinDistance <: AbstractGaussianProcessAcquisitionFunction end

function _find_next_point(gp::GaussianProcess, candidates::DataFrame, ::MaximinDistance)
    input = propertynames(candidates)
    X = Matrix(gp.training_data[:, input])
    Xc = Matrix(candidates[:, input])

    distances = [minimum(norm(Xc[i, :] - X[j, :]) for j in axes(X, 1)) for i in axes(Xc, 1)]
    ind = argmax(distances)
    return candidates[[ind], Cols(input)]
end

"""
    ExpectedImprovementForGlobalFit()

EIGF score rewards candidates far (in output) from their nearest
training observation, adjusted by local posterior variance.

### Reference
Lam, C. Q. (2008). Sequential Adaptive Designs in Computer Experiments for
Response Surface Model Fit. PhD dissertation, The Ohio State University.
"""
struct ExpectedImprovementForGlobalFit <: AbstractGaussianProcessAcquisitionFunction end

function _find_next_point(gp::GaussianProcess, candidates::DataFrame, ::ExpectedImprovementForGlobalFit)
    input = propertynames(candidates)
    candidates = copy(candidates)
    mean_col, var_col = _mean_var_cols(gp)
    evaluate!(gp, candidates; mode = :mean_and_var)

    μ = candidates[:, mean_col]
    σ² = candidates[:, var_col]
    X = Matrix(gp.training_data[:, input])
    Xc = Matrix(candidates[:, input])
    y = gp.training_data[:, gp.output]

    nearest = [argmin([norm(Xc[i, :] - X[j, :]) for j in axes(X, 1)]) for i in axes(Xc, 1)]
    eigf = abs2.(μ .- y[nearest]) .+ σ²

    ind = argmax(eigf)
    return candidates[[ind], Cols(input)]
end
