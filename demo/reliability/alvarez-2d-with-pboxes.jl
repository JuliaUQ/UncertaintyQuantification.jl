using UncertaintyQuantification

mc = MonteCarlo(10^4)
LHC = LatinHypercubeSampling(10^4)
ss = SubSetSimulation(2000, 0.1, 10, Uniform(-0.2, 0.2))
ss_infinity = SubSetInfinity(2000, 0.1, 10, 0.5)
ss_infinity_adaptive = SubSetInfinityAdaptive(2000, 0.1, 10, 10)
LS = LineSampling(200)
IS = ImportanceSampling(10^4)

## Alvarez model, from 10^6 interval MC he gets pf = [2.590 × 10−4, 0.503]

X1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 2), :σ => 1)), :X1)
X2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-2, 1), :σ => 2)), :X2)

inputs = [X1, X2]
models = Model(df -> df.X1 .^ 2 .+ df.X2, :g)
performance(df) = 7 .- df.g

rs = RandomSlicing(ss_infinity_adaptive, mc)

pf, out_lb, out_ub = probability_of_failure(models, performance, inputs, rs)

pf_dl = probability_of_failure(models, performance, inputs, DoubleLoop(mc))
