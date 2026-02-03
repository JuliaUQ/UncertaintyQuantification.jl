# Reliability Analysis

In the context of structural engineering, engineering design, and risk assessment, the term reliability is used to describe the ability of system to perform its intended function under varying conditions over time.

There, the state of a system is identified by its *performance function* (also sometimes refered to as limit state function) ``g(\boldsymbol{x})`` such that:

```math
g(\boldsymbol{x}) =
\begin{cases}
    > 0 & \text{safe\ domain}\\
    \leq 0 & \text{failure \ domain}\\
\end{cases}.
```

Then the *probability of failure* is defined as the likelihood of the system being in the failed state, given as

```math
p_f = \int_{g(\boldsymbol{x}) \leq 0} f_{\boldsymbol{X}}(\boldsymbol{x}) \mathrm{d} \boldsymbol{x}.
```

Here, ``f_{\boldsymbol{X}}(\boldsymbol{x})`` denotes the joint probability density function (PDF) of the input ``\boldsymbol{X}``.

## Definition of the Input, Model and Performance

The first step of the implementation of a reliability analysis in `UncertaintyQuantification.jl` is the definition of the probabilistic input and the model which is shown exemplarily.

Here we use a modification of the first example presented in [papaioannouCombination2021](@cite) which uses a quadratic performance function and a probabilistic input containing of two standard normal random variables ``\boldsymbol{X} = [X_1, X_2]`` with ``X_i \sim \mathcal{N}(0,1)``.
The model is given as

```math
y(\boldsymbol{X}) = 0.1(X_1 - X_2)^2 - \frac{1}{\sqrt{2}} (X_1 + X_2).
```

Then, the performance function is defined as

```math
g(\boldsymbol{X}) = y(\boldsymbol{X}) + 4.
```

The probabilistic input is implemented as

```@example reliability
using UncertaintyQuantification, DataFrames, Plots # hide
using Random # hide
Random.seed!(42) # hide
x = RandomVariable.(Normal(), [:x1, :x2])
```

Next we define the model for the response ``y(\boldsymbol{X})`` as

```@example reliability
y = Model(df -> 0.1*(df.x1 - df.x2).^2 - 1/sqrt(2) * (df.x1 + df.x2), :y)
nothing # hide
```

where the first input is the function ``y`` (which must accept a `DataFrame`) and the second argument is the `Symbol` for the output variable. With the help of the model, we can define the performance function ``g`` which again takes a `DataFrame` as an input:

```@example reliability
g(df) = df.y .+ 4
nothing # hide
```

## Approximation Methods

### First Order Reliability Method

The First Order Reliability Method (FORM) [rackwitzStructuralReliability1978](@cite) estimates the failure probability by finding a linear approximation of the performance function at the so-called *design point* ``\boldsymbol{U}^*``.
The design point represents the point on the surface of the performance function ``g(\boldsymbol{X}) = 0`` that is closest to the origin in the standard normal space.

That distance from the design point to the origin is referred to as the *reliability index* given as ``\beta^* = ||\boldsymbol{U}^*||``.
Due to the transformation to the standard normal space, the probability of failure is simply given as

```math
\hat{p}_{f, \mathrm{FORM}} = \Phi(-\beta^*)
```

where ``\Phi`` denotes the standard normal CDF.

In addition to the ``\beta^*``, the location of the design point is specified by the *important direction* defined as:

```math
\boldsymbol{\alpha}^* = \frac{\boldsymbol{U}^*}{||\boldsymbol{U}^*||}.
```

In `UncertaintyQuantification.jl` a FORM analysis can be performed calling `probability_of_failure(model, performance, input, simulation)` where `FORM()` is passed as the simulation method:

```@example reliability
pf_form, β, dp = probability_of_failure(y, g, x, FORM())

println("Probability of failure: $pf_form")
```

## Simulation Methods

### Monte Carlo Simulation

Monte Carlo Simulation (MCS) offers an approximation of the failure probability using stochastic simulation.

It utilizes an indicator function of the failure domain

```math
\mathbb{I}[g(\boldsymbol{x})] =
\begin{cases}
    0 & \text{when} \ g(\boldsymbol{x}) > 0\\
    1 & \text{when} \ g(\boldsymbol{x}) \leq 0\\
\end{cases}.
```

This allows for the failure probability to be interpreted as the expected value of the indicator function

```math
p_f = \int_{\boldsymbol{X}} \mathbb{I}[g(\boldsymbol{x})]\ f_{\boldsymbol{X}}(\boldsymbol{x}) \mathrm{d} \boldsymbol{x} = \mathbb{E}\big[\mathbb{I}[g(\boldsymbol{x})]\big].
```

The Monte Carlo estimate of the failure probability is given as

```math
p_f \approx \hat{p}_f = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[g(\boldsymbol{x}_i)]
```

where ``\{\boldsymbol{x}_i\}_{i=1}^N`` represents a set of ``N`` samples drawn from the input PDF ``f_{\boldsymbol{X}}(\boldsymbol{x})``.
The variance of the estimator is given as

```math
\operatorname{Var}[\hat{p}_f] = \frac{\hat{p}_f (1-\hat{p}_f)}{N}.
```

In `UncertaintyQuantification.jl` we can perform a Monte Carlo Simulation by defining the analysis as `MonteCarlo(n)`  where `n` is the number of samples:

```@example reliability
mc = MonteCarlo(10^7)
nothing # hide
```

Then the reliability analysis is performed by calling `probability_of_failure(model, performance, input, simulation)`.

```@example reliability
pf_mc, std_mc, samples = probability_of_failure(y, g, x, mc)

println("Probability of failure: $pf_mc")
println("Coefficient of variation: $(std_mc/pf_mc)")
```

### Importance Sampling

Based on the standard MCS method, a class of advanced method exist that have to goal to accelerate the estimation of the failure probability by requiring fewer model calls.

Importance Sampling [melchersImportanceSampling1989](@cite) introduces a second density that is *biased* in a way that it generates more samples in the failure domain.
Typically such a density is constructed around the design point obtained in a preceding FORM analysis.

In order to perform a reliability analysis using Importance Sampling, we again have to specify the number of samples and then can `probability_of_failure()`.

```@example reliability
is = ImportanceSampling(1000)
pf_is, std_is, samples = probability_of_failure(y, g, x, is)

println("Probability of failure: $pf_is")
println("Coefficient of variation: $(std_is/pf_is)")
```

### Radial Based Importance Sampling

Radial based importance sampling (RBIS) [harbitzEfficientSamplingMethod1986](@cite) increases the efficiency of the Monte Carlo simulation by sampling in standard normal space and excluding a β-sphere where no failures occur from the sampling domain. Here, `β` is the reliability index obtained from a preliminary analysis like FORM. The probability of failure is then estimated as

```math
p_f \approx \hat{p}_f = (1- \chi^2_k(\beta^2) \frac{1}{N} \sum_{i=1}^N \mathbb{I}[g(\boldsymbol{x}_i)],
```

where ``\chi^2_k`` is the CDF of the Chi-squared distribution with ``k`` degrees of freedom and ``k`` is the number of random variables.

If no `β` or `β=0.0` is passed to the [`RadialBasedImportanceSampling`](@ref) constructor, a FORM analysis will automatically be performed.

```@example reliability
rbis = RadialBasedImportanceSampling(1000)
pf_rbis, std_rbis, samples = probability_of_failure(y, g, x, rbis)

println("Probability of failure: $pf_rbis")
println("Coefficient of variation: $(std_rbis/pf_rbis)")

scatter(samples.x1, samples.x2; xlabel="x1", ylabel="x2", aspect_ratio=:equal) # hide
savefig("rbis-samples.svg"); nothing # hide
```

A scatter plot clearly shows the exclusion of the β-sphere.
![RBIS samples](rbis-samples.svg)

### Line Sampling

Another advanced Monte Carlo method for reliability analysis is Line Sampling [koutsourelakisReliability2004](@cite).
Its main idea is to use parallel lines for sampling rather than points.

Therefore first the problem is transformed into the standard normal space to make use of the invariance of rotation.
The important direction ``\boldsymbol{\alpha}`` is determined, e.g., using FORM or the gradient at the origin.
Then, samples are generated and projected onto the hyperplane orthogonal to ``\boldsymbol{\alpha}``.
From each point on the hyperplane, a line is drawn parallel to ``\boldsymbol{\alpha}`` and its intersection with the performance function is determined using root finding based on a spline interpolation scheme, giving the set of distances ``\{\beta^{(i)}\}_{i=1}^N`` from the hyperplane to the intersection with the performance function.
Due to working in the standard normal space, the *failure probability along each line* is given as

```math
p_{f, \mathrm{line}}^{(i)} = \Phi(-\beta^{(i)})
```

Finally, the probability of failure is obtained as the mean of the failure probabilities along the lines

```math
\hat{p}_{f,\mathrm{LS}} = \frac{1}{N} \sum_{i=1}^N p_{f, \mathrm{line}}^{(i)}.
```

The variance of ``\hat{p}_{f,\mathrm{LS}}`` is given by the variance of the line failure probabilities:

```math
\operatorname{Var}[\hat{p}_{f,\mathrm{LS}}] = \frac{1}{N(N-1)} \sum_{i=1}^N \Big(p_{f, \mathrm{line}}^{(i)} - \hat{p}_{f,\mathrm{LS}}\Big)^2.
```

Similar to standard MCS, we have to pass ``N`` to the Line Sampling method. However, here we pass the number of lines.
Optionally, we can pass a vector of the points along each line that are used to evaluate the performance function and a predetermined direction ``\boldsymbol{\alpha}``:

```@example reliability
ls = LineSampling(100, collect(0.5:0.5:8.0))
pf_ls, std_ls, samples = probability_of_failure([y], g, x, ls)

println("Probability of failure: $pf_ls")
println("Coefficient of variation: $(std_ls/pf_ls)")
```

### Advanced Line Sampling

Advanced Line Sampling [deangelisAdvances2015](@cite) is a further enhancement of the standard line sampling methods due to two main features:

1. The important direction ``\boldsymbol{\alpha}`` is adapted once a more probable point is found
2. The lines are processed sorted by proximity of the points on the hyperplane.

Especially the second point enables the use of an iterative root finder using Newton's method.

The definition of the `AdvancedLineSampling` simulation method is similar to that of regular Line Sampling.
The number of lines has to be given to the constructor and we can optionally give the number of points along the line which is only used to find the starting point of the iterative root search.

```@example reliability
als = AdvancedLineSampling(100, collect(0.5:0.5:10))
pf_als, std_als, samples = probability_of_failure([y], g, x, als)

println("Probability of failure: $pf_als")
println("Coefficient of variation: $(std_als/pf_als)")
```

For `AdvancedLineSampling`, we can also define the (initial) direction and options of the iterative root finding, i.e., the `tolerance`, `stepsize` of the gradient and `maxiterations`.

!!! note "Parallelism"
    We note that Advanced Line Sampling is a serial algorithm, although much fewer samples (order of magnitude) are required. If a large amount of parallel compute is available, standard Line Sampling may be more attractive, which is "embarrassingly" parallel like Monte Carlo.

## Subset Simulation

Subset simulation [auEstimationSmallFailure2001](@cite) is an advanced simulation technique for the estimation of small failure probabilities.
This approach involves decomposing the problem into a sequence of conditional probabilities that are estimated using Markov Chain Monte Carlo.

We create the [`SubSetSimulation`](@ref) object and compute the probability of failure using a standard Gaussian proposal PDF. The value for the target probability of failure at each intermediate level is set to ``0.1`` which is generally accepted as the optimal value.

```@example reliability
subset = SubSetSimulation(1000, 0.1, 10, Normal())
pf_sus, std_sus, samples = probability_of_failure(y, g, x, subset)

println("Probability of failure: $pf_sus")
println("Coefficient of variation: $(std_sus/pf_sus)")
```

Alternatively, instead of using the standard Subset simulation algorithm (which internally uses Markov Chain Monte Carlo), we can use [`SubSetInfinity`](@ref) to compute the probability of failure, see [auRareEventSimulation2016](@cite). Here we use a standard deviation of ``0.5`` to create the proposal samples for the next level.

```@example reliability
subset = SubSetInfinity(1000, 0.1, 10, 0.5)
pf_sus, std_sus, samples = probability_of_failure(y, g, x, subset)

println("Probability of failure: $pf_sus")
println("Coefficient of variation: $(std_sus/pf_sus)")
```

## Imprecise Reliability Analysis

if *epistemic* uncertainty is considered in the input variables it must also be propagated through the analysis, converting the probability of failure into an interval itself. The goal of the reliability analysis is then to find the lower bound ``\underline{p}_f`` and upper bound ``\overline{p}_f`` of the probability of failure such that ``p_f \in [\underline{p}*f, \overline{p}*f]``. Formally, this can be defined as

```math
    \underline{p}*f = \min*{\vartheta \in \mathbf{\theta}} \int \mathbb{I}[g(x)]f*{\mathbf{X}}(x, \vartheta) dx
```

and

```math
\overline{p}*f = \max*{\vartheta \in \mathbf{\theta}} \int  \mathbb{I}[g(x)]f*{\mathbf{X}}(x, \vartheta) dx,
```

where ``\theta`` represents the epistemic parameters characterizing the inputs.

The following two sections provide two possible solutions to this challenging problem.

### Double-loop Monte Carlo Simulation

The simplest way to solve the imprecise reliability analysis is by double-loop simulation. The name refers to the need for two loops in comparison to a standard reliability analysis. The outer-loop essentially solves two optimisation problems over the parameter space of the epistemic inputs to find the combinations that minimize and maximize the probability of failure. The inner-loop requires a reliability calculation, with the current combination of parameters under consideration being mapped to precise inputs. In practice, [`IntervalVariable`](@ref) is mapped to a [`Parameter`](@ref) while a [`ProbabilityBox`](@ref) nested inside a [`RandomVariable`](@ref) yields a regular `RandomVariable`. Therefore, the double-loop simulation treats p-boxes as parametric. Then, a comprehensive reliability analysis using these purely *aleatory* inputs is carried out. This repeated analysis in the inner-loop makes the double-loop simulation computationally demanding. If a Monte Carlo simulation is applied in the inner-loop this is known as double-loop Monte Carlo simulation.

Special attention must be paid to the type of optimisation algorithm used, as random sampling in the inner-loop leads to non-smooth objective functions. Here we have chosen to apply mesh adaptive direct search (MADS) algorithms which are specifically designed for such cases [abramsonOrthoMADSDeterministicMADS2009](@cite). However, exploration of alternative methods is part of the ongoing development.

Estimating the probability of failure is effectively separated into two independent problems, one for each bound. This provides the ability to apply different types of analyses. For example, using a larger number of samples for the lower bound.

As an example, consider the function

```math
     f(x_1,x_2) = x_1 +x_2,
```

where ``x_1 \in [-1,1]`` is an interval and ``x_2 \sim N(0, [1,2])`` is distributed as a Gaussian p-box. The associated performance function is

```math
     g(x) = 9 + f(x),
```

i.e., failures are ``f(x) \leq -9``. The analytical solution to this problem is known to be

```math
    \underline{p}_f = \Phi(-9; 1, 1)
```

and

```math
     \overline{p}_f = \Phi(-9; -1, 2),
```

where ``\Phi(x; \mu, \sigma)`` is the Gaussian CDF with mean ``\mu`` and standard deviation ``\sigma``. The follow code shows how to set up this problem in *UncertaintyQuantification.jl*.

```@example reliability
     x1 = IntervalVariable(-1,1,:x1)
     x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(1,2,))), :x2)
     f = Model(df -> df.x1 .+ df.x2, :f)
     return nothing # hide
```

Then, the relibility analysis is again run through the  [`probability_of_failure`](@ref) function. As the final argument we we pass the inner simulation wrapped inside a [`DoubleLoop`](@ref). For this simple example we apply the first order reliability method (FORM), as this can reliably estimate the small failure probability of the lower bound which is \(\approx 7.6e-24\). The bounds on the ```p_\text{f}``` can be obtained from the output interval through `pf.lb` and `pf.ub`. The epistemic uncertainty is propagated correctly and matches the analytical solution. On top of the probability of failure, the double-loop analysis also returns the parameters that lead to the lower and upper bound. Here, they are correctly identified as ``([1,1]`` and ``[-1,2]``.

```@example reliability
     pf, x_lb, x_ub = probability_of_failure(f, df -> 9 .+ df.f, [x1,x2], DoubleLoop(FORM()))
```

### Random Slicing

An alternative method for computing probability bounds for reliability problems is based on random-set theory, as outlined by [alvarez2018estimation](@cite). We colloquially call this "random slicing", as will become apparent. As opposed to double-loop Monte Carlo, random slicing is a *distribution-free* or a *non-parametric* technique, as it does not make use of distribution parameters or family. For this reason, it is slightly more general, but can provide wider probability intervals in certain simulations.

In random slicing, we make use of the fact that a p-box is a random-set [fersonConstructingProbabilityBoxes2015](@cite) to simulate random realisations (random intervals) from the inverses of the bounding CDFs

```math
     \Gamma(\alpha)=[\overline{F}^{-1}(\alpha), \underline{F}^{-1}(\alpha)],
```

where ``\alpha \sim U(0,1)`` is a sample from a uniform distribution. This interval can be visualised as a horizontal cut (or slice) of the p-box at an ``\alpha \in [0,1]``. This interval is then propagated through the model ``f`` or performance function ``g`` using an optimiser or surrogate model,

```math
     \underline{g}(\alpha) = \min_{x \in \Gamma(\alpha)}g(x),
```

```math

     \overline{g}(\alpha) = \max_{x \in \Gamma(\alpha)}g(x).
```

In *UncertaintyQuantification.jl* the intervals are also propagated using MADS. In the multivariate case, we can combine two correlated intervals using a Cartesian product

```math
     [\overline{F}_X^{-1}(\alpha_X), \underline{F}_X^{-1}(\alpha_X)] \times [\overline{F}_Y^{-1}(\alpha_Y), \underline{F}_Y^{-1}(\alpha_Y)],
```

where ``(\alpha_X, \alpha_Y) \sim C_{XY}`` are samples of the copula between ``X`` and ``Y``.

The reliability analysis can be written as thus:

```math
     \underline{p}*{\text{f}} = \int_U \mathbb{I}[\overline{g}(\alpha)] dC,(\alpha)
```

```math
     \overline{p}*{\text{f}} = \int_U \mathbb{I}[\underline{g}(\alpha)] dC.(\alpha)
```

In some sense, the two loops from the double-loop method have been reversed, where now the outer-loop handles the random (aleatory) component, and the inner-loop handles the interval propagation (epistemic). Describing the analysis this way essentially gives two separate reliability calculations, with ``\underline{g}`` and ``\overline{g}`` as the two target performance functions. Rosenblatt transformations may be used to associate a standard normal distribution to the copula $C$, and one may then use any standard reliability method to compute the performance bounds.

The software implementation is such that this imprecise reliability method can be coupled to any simulation method (FORM, line sampling, etc.) in a straightforward way. As with the double-loop, one can apply different simulations for each bound if desired.

The problem setup for random slicing is identical to that of the double-loop. The only difference is that a [`RandomSlicing`](@ref) instance is passed instead of [`DoubleLoop](@ref).

Here, the lower bound is again estimated using FORM, while we apply subset simulation to obtain the upper bound. The result for the lower bound matches the analytical solution perfectly. The upper bound is estimated accurately as ``\overline{p}_f \approx 3.884325e-5``. Note, that in addition to the probability of failure, random slicing also returns other outputs of the underlying simulations, such as the coefficient of variation and the evaluated samples for potential post-processing.

```@example reliability
     subset = SubSetSimulation(2000, 0.1, 10, Normal())
     pf, out_lb, out_ub = probability_of_failure(f, df -> 9 .+ df.f, [x1,x2], RandomSlicing(FORM(), subset))
```

As outlined by [alvarez2018estimation], other forms of random-sets can in principle be evaluated with this method, such as possibility distributions [dubois1990consonant](@cite) or general Dempster–Shafer structures [shafer1976mathematical](@cite). However, careful consideration of multivariate extensions of these structures must be taken [schmelzer2023random](@cite). For this reason, we restrict ourselves to distributions, intervals, and p-boxes for the time being.
