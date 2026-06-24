using UncertaintyQuantification
using DataFrames
using Base.Threads
using DifferentialEquations
using DelimitedFiles
using Interpolations

@show Threads.nthreads()

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

# epsd = EmpiricalPSD(w, ensemble[1, :])

psd = ImprecisePSD(w, ensemble, GaussianRadialBasis(centers, 0.2))
gm = SpectralRepresentation(psd, collect(0:0.02:10), :gm)
gm_model = StochasticProcessModel(gm)

# counter for model calls
global n_calls::Integer = 0

function sdof(df::DataFrame)
    global n_calls += size(df, 1)
    solver = AutoTsit5(Rosenbrock23())

    X = zeros(length(gm.time), size(df, 1))

    Threads.@threads for (i, s) in collect(enumerate(eachrow(df)))
        gm_itp = linear_interpolation(gm.time, df[i, :gm])

        function f(dy, y, _, t)
            dy[1] = y[2]

            dy[2] = @views -df[i, :c] / df[i, :m] * y[2] - df[i, :k] / df[i, :m] * y[1] +
                gm_itp(t)
            return nothing
        end

        prob = ODEProblem(f, [0.0, 0.0], (gm.time[1], gm.time[end]))

        sol = solve(prob, solver; saveat=gm.time)

        X[:, i] = sol[1, :]
    end
    return collect(eachcol(X))
end

displacement = Model(sdof, :d)

inputs = [gm, m, k, c]

models = [gm_model, displacement]

function g(df)
    return map(eachrow(df)) do s
        1.5 - maximum(abs.(s.d))
    end
end

@time pf, x_lb, x_ub = probability_of_failure(
    models, g, inputs, DoubleLoop(MonteCarlo(1000))
)

@show pf, n_calls

# rmprocs(workers())
