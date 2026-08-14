using UncertaintyQuantification
using Dierckx

using DifferentialEquations

m = Parameter(0.5, :m)

k = Parameter(2, :k)
c = Parameter(0, :c)

ω = collect(0:0.6:150)

cp = CloughPenzien(ω, 0.1, 0.8π, 0.6, 8π, 0.6)

# Ground motion
gm = SpectralRepresentation(cp, collect(0:0.02:10), :gm)

gm_model = StochasticProcessModel(gm)

function sdof(df)
    return map(eachrow(df)) do s
        gm_interp = Spline1D(gm.time, s.gm; k = 1)

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

mc_pf, mc_std, mc_samples, = probability_of_failure(models, g, inputs, MonteCarlo(10000))

ss_pf, ss_std, ss_samples = probability_of_failure(
    models, g, inputs, SubSetInfinityAdaptive(2000, 0.1, 10, 10)
)
