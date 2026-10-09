using UncertaintyQuantification, QuasiMonteCarlo

x = RandomVariable.(Uniform(-π, π), [:x1, :x2, :x3])
a = Parameter(7, :a)
b = Parameter(0.05, :b)

inputs = [x; a; b]

ishigami = Model(
    df -> sin.(df.x1) .+ df.a .* sin.(df.x2) .^ 2 .+ df.b .* (df.x3 .^ 4) .* sin.(df.x1), :y
)

data = UncertaintyQuantification.sample(inputs, QuasiMonteCarloSampling(512, SobolSample()))
evaluate!(ishigami, data)

phs = PolyharmonicSpline(data, 2, :y)

si = sobolindices([ishigami], inputs, :y, MonteCarlo(8192))
si_phs = sobolindices([phs], inputs, :y, QuasiMonteCarloSampling(8192, SobolSample()))

println("First order sobol indices of the ishigami function: $(si.FirstOrder)")
println(
    "First order sobol indices of the polyharmonic spline meta mode: $(si_phs.FirstOrder)"
)
