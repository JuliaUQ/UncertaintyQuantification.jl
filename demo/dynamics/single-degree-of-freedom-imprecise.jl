using UncertaintyQuantification
using Dierckx
using JLD2
using DifferentialEquations

@load "ensemble.jld2"

m = Parameter(0.5, :m)

k = Parameter(2, :k)
c = Parameter(0, :c)

b = GaussianRadialBasis(centers, 0.2)

psd = ImprecisePSD(w, ensemble, b)
# Ground motion
gm = SpectralRepresentation(psd, collect(0:0.02:10), :gm)

gm_model = StochasticProcessModel(gm)

function sdof(df)
    return map(eachrow(df)) do s
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
        0.1 - maximum(abs.(s.d))
    end
end

pf, x_lb, x_ub = probability_of_failure(models, g, inputs, DoubleLoop(MonteCarlo(1000)))
