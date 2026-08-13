using UncertaintyQuantification

ω = collect(0:0.6:150)

cp_psd = CloughPenzien(ω, 0.1, 0.8π, 0.6, 8π, 0.6)

gm = SpectralRepresentation(cp_psd, collect(0:0.02:10), :gm)
gm_model = StochasticProcessModel(gm)

capacity = Parameter(21, :cap) #estimation for a capacity value which for this gm_model results in pf ~ [1.0e-6, 2.0e-5]

function limitstate(df)
    return df.cap - map(x -> maximum(abs.(x)), df.gm)
end

models = [gm_model]
inputs = [gm, capacity]

mc = MonteCarlo(10^6)

@time mc_pf, mc_std, mc_samples = probability_of_failure(models, limitstate, inputs, mc)
println("Monte Carlo probability of failure $mc_pf ($(size(mc_samples, 1)) model evaluations)")

subset = UncertaintyQuantification.SubSetSimulation(2000, 0.1, 10, Uniform(-0.5, 0.5))
subset_pf, subset_std, subset_samples = probability_of_failure(models, limitstate, inputs, subset)
println("Subset Simulation probability of failure: $subset_pf ($(size(subset_samples, 1)) model evaluations)")

subset_inf = UncertaintyQuantification.SubSetInfinity(2000, 0.1, 10, 0.5)
subset_pf_inf, subset_std_inf, subset_samples_inf = probability_of_failure(models, limitstate, inputs, subset_inf)
println("Subset infinity probability of failure: $subset_pf_inf ($(size(subset_samples_inf, 1)) model evaluations)")

subset_adap = UncertaintyQuantification.SubSetInfinityAdaptive(2000, 0.1, 10, 10, 0.6, 1.0)
subset_pf_adap, subset_std_adap, subset_samples_adap = probability_of_failure(models, limitstate, inputs, subset_adap)
println("Subset infinity adaptive probability of failure: $subset_pf_adap ($(size(subset_samples_adap, 1)) model evaluations)")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
