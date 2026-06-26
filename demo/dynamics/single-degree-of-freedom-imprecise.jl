using Distributed

@everywhere begin
    using UncertaintyQuantification
    using DataFrames
    using DifferentialEquations
    using DelimitedFiles
    using Dierckx

    w = vec(readdlm("w.csv"))
    centers = readdlm("centers.csv")
    ensemble = readdlm("ensemble.csv")

    # SDOF parameters
    #m = IntervalVariable(0.5, 1.0, :m)
    #k = IntervalVariable(2, 3, :k)
    #c = IntervalVariable(0, 1, :c)
    m = Parameter(1, :m)
    k = Parameter(3, :k)
    c = Parameter(0.5, :c)

    psd = ImprecisePSD(w, ensemble, GaussianRadialBasis(centers, 0.2))
    gm = SpectralRepresentation(psd, collect(0:0.05:10), :gm)
    gm_model = StochasticProcessModel(gm)

    # counter for model calls
    global n_calls::Integer = 0

    function sdof(df::DataFrame)
        global n_calls += size(df, 1)
        solver = Tsit5()

        return pmap(eachrow(df)) do s
            gm_itp = Spline1D(gm.time, s.gm; k=1)

            function f(dy, y, _, t)
                dy[1] = y[2]

                dy[2] = -s.c / s.m * y[2] - s.k / s.m * y[1] + gm_itp(t)
                return nothing
            end

            prob = ODEProblem(f, [0.0, 0.0], (gm.time[1], gm.time[end]))

            sol = solve(prob, solver; saveat=gm.time)

            return maximum(abs.(sol[1, :]))
        end
    end

    displacement = Model(sdof, :d)

    inputs = [gm, m, k, c]

    models = [gm_model, displacement]

    function g(df)
        return map(eachrow(df)) do s
            1.5 - s.d
        end
    end

    N = 10
end

@time pf, x_lb, x_ub = probability_of_failure(models, g, inputs, DoubleLoop(MonteCarlo(N)));

println("Double Loop pf: $pf ($n_calls model calls)")

@time pf, out_lb, out_ub = probability_of_failure(
    models, g, inputs, RandomSlicing(MonteCarlo(N))
);

println("RandomSlicing pf: $pf ($n_calls model calls)")
