using UncertaintyQuantification

function run_sim(N_dimensions, pf_target, N_samples, N_runs=50)
    inputs = RandomVariable.(Normal(), [Symbol("x$i") for i in 1:N_dimensions])

    y = Model(df -> sum(eachcol(df[:, names(inputs)])), :f)

    fail_limit = quantile(Normal(0, sqrt(N_dimensions)), 1 - pf_target)

    function limitstate(df)
        return fail_limit .- reduce(vcat, df.f)
    end

    subset_MH = SubSetSimulation(N_samples, 0.1, 20, Uniform(-0.5, 0.5))
    subset_CS = SubSetInfinity(N_samples, 0.1, 20, 0.5)
    subset_aCS = SubSetInfinityAdaptive(N_samples, 0.1, 20, Integer(floor(N_samples * 0.1)))

    pf_MH = [probability_of_failure(y, limitstate, inputs, subset_MH)[1] for _ in 1:N_runs]

    pf_CS = [probability_of_failure(y, limitstate, inputs, subset_CS)[1] for _ in 1:N_runs]

    pf_aCS = [
        probability_of_failure(y, limitstate, inputs, subset_aCS)[1] for _ in 1:N_runs
    ]

    return pf_MH, pf_CS, pf_aCS
end
