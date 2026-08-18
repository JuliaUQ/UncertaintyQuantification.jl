using UncertaintyQuantification

x = RandomVariable.(Uniform(-π, π), [:x1, :x2, :x3])
a = Parameter(7, :a)
b = Parameter(0.05, :b)

inputs = [x; a; b]

ishigami = Model(
    df -> sin.(df.x1) .+ df.a .* sin.(df.x2) .^ 2 .+ df.b .* (df.x3 .^ 4) .* sin.(df.x1), :y
)

data = sample(inputs, SobolSampling(500))
evaluate!(ishigami, data)

phs = PolyharmonicSpline(data, 2, :y)

si = sobolindices([ishigami], inputs, :y, MonteCarlo(10000))
si_phs = sobolindices([phs], inputs, :y, SobolSampling(10000))

println("First order sobol indices of the ishigami function: $(si.FirstOrder)")
println(
    "First order sobol indices of the polyharmonic spline meta mode: $(si_phs.FirstOrder)"
)
