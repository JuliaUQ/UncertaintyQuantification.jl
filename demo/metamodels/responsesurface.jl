using UncertaintyQuantification

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])

himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)

design = FullFactorial([5, 5])

training_data = sample(x, design)
evaluate!(himmelblau, training_data)
rs = ResponseSurface(training_data, :y, 4)

test_data = sample(x, 1000)
evaluate!(rs, test_data)

p_data = test_data[:, [:x1, :x2]]
evaluate!(himmelblau, p_data)

mse = mean((p_data.y .- test_data.y) .^ 2)
println("MSE is:  $mse")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
