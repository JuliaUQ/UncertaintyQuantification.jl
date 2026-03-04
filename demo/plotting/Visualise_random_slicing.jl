using UncertaintyQuantification, DataFrames, Plots, ColorSchemes
using UncertaintyQuantification: lo, hi

X1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(0, 2), :σ => 4)), :X1)
X2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(0, 2), :σ => Interval(5, 5.5))), :X2)

function limitstate(df)
    return  300 .- 2* df.X2 .^2 + df.X1 .+ df.X2 .^3 - 3 *df.X1.^2 .+ df.X1 .*df.X2
end

model = Model(limitstate, :g)

inputs =[X1; X2]

@time samples = sample(inputs, SobolSampling(1000))

propagate_intervals!(model, samples)

gs = samples.g

safe = lo.(gs) .>= 0
fail = hi.(gs) .<= 0
unsure = lo.(gs) .<= 0 .&& 0 .<= hi.(gs)

N_grid = 1000

xs_physical = range(-15, 12, length = N_grid)
ys_physical = range(-10, 10, length = N_grid)

XY_physical = [[x; y] for x in xs_physical, y in ys_physical]
XY_physical = reduce(hcat, XY_physical[:])

samples_physical = DataFrame(:X1 => XY_physical[1,:], :X2 => XY_physical[2,:])

evaluate!(model, samples_physical)
contour(xs_physical, ys_physical, samples_physical.g .<= 0, color = :red, linewidth = 2, levels = 1, label = "g", colorbar =false)


plot(samples.X1[safe], samples.X2[safe], alpha = 0.2, color = "blue", label = "safe")
plot!(samples.X1[fail], samples.X2[fail], alpha = 0.2, color = "red", label = "failed")
plot!(samples.X1[unsure], samples.X2[unsure], alpha = 0.2, color = "yellow", label = "indetermined")
contour!(xs_physical, ys_physical, samples_physical.g .<= 0, color = :red, linewidth = 2, levels = 1, label = "g", colorbar =false)


### Subset viz, with tighter dists

X1_ = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(0, 2), :σ => 2)), :X1)
X2_ = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => 1.5)), :X2)

inputs2 = [X1_, X2_]

ss = SubSetInfinity(1000, 0.1, 10, 0.5)
@time pf_ss, outputs_ss1, outputs_ss2 = probability_of_failure(limitstate, inputs2, RandomSlicing(ss))

samples_lo = outputs_ss1[2]
N_levels = maximum(samples_lo.level)

colour = colormap("RdBu", N_levels)

p1 = plot(samples_lo.X1[samples_lo.level .== 1], samples_lo.X2[samples_lo.level .==1], color = colour[1], label ="level1", alpha = 0.2)
for i = 2:N_levels
    plot!(p1, samples_lo.X1[samples_lo.level .==i], samples_lo.X2[samples_lo.level .==i], color = colour[i], label ="level$i", alpha = 0.2)
end
contour!(p1, xs_physical, ys_physical, samples_physical.g .<= 0, color = :red, linewidth = 2, levels = 1, label = true, colorbar =false, title = "lower bound")

## Lower bound

samples_lo = outputs_ss2[2]
N_levels = maximum(samples_lo.level)

# colour = colormap("RdBu", N_levels)

p2 = plot(samples_lo.X1[samples_lo.level .== 1], samples_lo.X2[samples_lo.level .==1], color = colour[1], label ="level1", alpha = 0.2)
for i = 2:N_levels
    plot!(p2, samples_lo.X1[samples_lo.level .==i], samples_lo.X2[samples_lo.level .==i], color = colour[i], label ="level$i", alpha = 0.2)
end
contour!(p2, xs_physical, ys_physical, samples_physical.g .<= 0, color = :red, linewidth = 2, levels = 1, label = true, colorbar =false, title = "upper bound")

# Side by side
p = plot(p1, p2, layout = (1, 2), size = (1200, 500))