using UncertaintyQuantification, Plots

X1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 2), :σ => 1)), :X1)
X2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 2), :σ => 2)), :X2)
X3 = IntervalVariable(-1, 2, :s)
h = RandomVariable(Normal(0.24, 0.01), :h) # height

plot(X1)    # p-box
plot(X1, color = "red") # red p-box
plot(X3)    # Interval
plot(h)     # distribution 

inputs = [X1, X2, X3, h]

plot(inputs)    # Everything together

samples = sample(inputs, 200)

plot(X1)
plot!(samples.X1)    # Samples ecdf samples of X1

plot(samples.X1[1], samples.X2[1])  # Plots 2D box

plot(samples.X1, samples.X2)        # Plots bivariate random sets of X1 and X2
