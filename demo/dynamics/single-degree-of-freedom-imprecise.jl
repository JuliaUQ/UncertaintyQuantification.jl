using UncertaintyQuantification
using Dierckx
using DelimitedFiles
using DifferentialEquations

w = vec(readdlm("w.csv"))
centers = readdlm("centers.csv")
ensemble = readdlm("ensemble.csv")

# SDOF parameters
m = IntervalVariable(0.5, 1.0, :m)
k = IntervalVariable(2, 3, :k)
c = IntervalVariable(0, 1, :c)

psd = ImprecisePSD(w, ensemble, GaussianRadialBasis(centers, 0.2))
gm = SpectralRepresentation(psd, collect(0:0.02:10), :gm)
gm_model = StochasticProcessModel(gm)

# counter for model calls
global n_calls::Integer = 0

function sdof(df)
    return map(eachrow(df)) do s
        global n_calls += 1
        gm_interp = Spline1D(gm.time, s.gm; k=1)

        function f(dy, y, _, t)
            dy[1] = y[2]

            return dy[2] = -s.c / s.m * y[2] - s.k / s.m * y[1] + gm_interp(t)
        end

        prob = ODEProblem(f, [0.0, 0.0], (gm.time[1], gm.time[end]))

        sol = solve(prob, Tsit5())

        return sol[1, :]
    end
end

displacement = Model(sdof, :d)

inputs = [gm, m, k, c]

models = [gm_model, displacement]

function g(df)
    return map(eachrow(df)) do s
        1.5 - maximum(abs.(s.d))
    end
end

pf, x_lb, x_ub = probability_of_failure(models, g, inputs, DoubleLoop(MonteCarlo(10000)))

@show pf, n_calls