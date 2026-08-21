#===
## Signal Analysis
### First passage analysis of stochastic signals generated from a stochastic process approximated by the spectral representation method

In this example we will perform a first passage analysis of a stochastic process generated from a spectral representation model. 
The spectral representation method is a technique to approximate a stochastic process by a linear combination of sinusoidal functions. 
It is used to generate stochastic signals which can, for example, represent the ground motion of an earthquake. 

First passage analysis is a method to estimate the probability of a limit state being exceeded by a stochastic process. The limit state function is usually of the 
structure "limit state = capacity - demand", where the capacity is a constant value and the demand is the maximum absolute value of the stochastic signals. 
The probability of failure is then estimated by the fraction of the number of times the capacity is exceeded by the demand. For more information on the theory see [Reliability-Analysis](@ref).

For parallel execution, see the example in [OpenSees supported beam parallel](@ref)
===#

#md using UncertaintyQuantification # hide
#jl using UncertaintyQuantification

# Frequency discretization for the Power Spectral Density Function (PSD)
ω = collect(0:0.6:150)

# Definition of Clough Penzien PSD with prescribed parameters
cp_psd = CloughPenzien(ω, 0.1, 0.8π, 0.6, 8π, 0.6)

# Ground motion model
gm = SpectralRepresentation(cp_psd, collect(0:0.02:10), :gm)
gm_model = StochasticProcessModel(gm)

# Capacity value for the limit state function
capacity = Parameter(21, :cap) #estimation for a capacity value which for this gm_model results in pf ~ [1.0e-6, 2.0e-5]

# Limit state function
function limitstate(df)
    return df.cap - map(x -> maximum(abs.(x)), df.gm)
end

models = [gm_model]
inputs = [gm, capacity]

# Compute probability of failure using standard Monte Carlo
mc = MonteCarlo(10^6)

# Simple Monte Carlo simulation with 10^6 samples to estimate a failure probability (``pf \approx [1.0e-6, 2.0e-5]``)

#md # ```julia
#md # mc_pf, mc_std, mc_samples = probability_of_failure(models, limitstate, inputs, mc)
#md # ```

#jl @time mc_pf, mc_std, mc_samples = probability_of_failure(models, limitstate, inputs, mc)
#jl println("Monte Carlo probability of failure $mc_pf ($(size(mc_samples, 1)) model evaluations)")

# Compute probability of failure using Subset Sampling
#md # ```julia
#md # subset = UncertaintyQuantification.SubSetSimulation(2000, 0.1, 10, Uniform(-0.5, 0.5))
#md # subset_pf, subset_std, subset_samples = probability_of_failure(models, limitstate, inputs, subset)
#md # ```

#jl subset = UncertaintyQuantification.SubSetSimulation(2000, 0.1, 10, Uniform(-0.5, 0.5))
#jl subset_pf, subset_std, subset_samples = probability_of_failure(models, limitstate, inputs, subset)
#jl println("Subset Simulation probability of failure: $subset_pf ($(size(subset_samples, 1)) model evaluations)")

# Compute probability of failure using conditional Subset Sampling
#md # ```julia
#md # subset_inf = UncertaintyQuantification.SubSetInfinity(2000, 0.1, 10, 0.5)
#md # subset_pf_inf, subset_std_inf, subset_samples_inf = probability_of_failure(models, limitstate, inputs, subset_inf)
#md # ```

#jl subset_inf = UncertaintyQuantification.SubSetInfinity(2000, 0.1, 10, 0.5)
#jl subset_pf_inf, subset_std_inf, subset_samples_inf = probability_of_failure(models, limitstate, inputs, subset_inf)
#jl println("Subset infinity probability of failure: $subset_pf_inf ($(size(subset_samples_inf, 1)) model evaluations)")

# Compute probability of failure using adaptive Subset Sampling
#md # ```julia
#md # subset_adap = UncertaintyQuantification.SubSetInfinityAdaptive(2000, 0.1, 10, 10, 0.6, 1.0)
#md # subset_pf_adap, subset_std_adap, subset_samples_adap = probability_of_failure(models, limitstate, inputs, subset_adap)
#md # ```

#jl subset_adap = UncertaintyQuantification.SubSetInfinityAdaptive(2000, 0.1, 10, 10, 0.6, 1.0)
#jl subset_pf_adap, subset_std_adap, subset_samples_adap = probability_of_failure(models, limitstate, inputs, subset_adap)
#jl println("Subset infinity adaptive probability of failure: $subset_pf_adap ($(size(subset_samples_adap, 1)) model evaluations)")
