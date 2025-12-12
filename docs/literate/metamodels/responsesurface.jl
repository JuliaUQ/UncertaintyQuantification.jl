#===
# Metamodels
## Design Of Experiments

Design Of Experiments (DOE) offers various designs that can be used for creating a model of a given system. The core idea is to evaluate significant points of the system in order to obtain a sufficient model while keeping the effort to achieve this relatively low.
Depending on the parameters, their individual importance and interconnections, different designs may be adequate.

The ones implemented here are `TwoLevelFactorial`, `FullFactorial`, `FractionalFactorial`, `CentralComposite` and `BoxBehnken`.
===#

#===
## Response Surface

A Response Surface is a structure used for modeling.
    It can be trained by providing it with evaluated points of a function.
    It will then, using polynomial regression, compute a model of that function.
===#

#===
## Example

In this example, we will model the following test function (known as Himmelblau's function)

in the range ``x1, x2 ∈ [-5, 5]``. It is defined as

```math
f(x1, x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2.
```
===#

#md using Plots #hide
#md a = range(-5, 5; length=1000)   #hide
#md b = range(5, -5; length=1000)   #hide
#md himmelblau_f(x1, x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2 #hide
#md s1 = surface(a, b, himmelblau_f; plot_title="Himmelblau's function")   #hide
#md savefig(s1, "himmelblau.svg"); nothing # hide

# ![](himmelblau.svg)
#===
At first we need to create an array of random variables, that will be used when evaluating the points that our design produces.
It will also define the range of the function we want the design to fit.
This is also a good time to declare the function that we are working with.
===#

using UncertaintyQuantification

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])

himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)

#===
Our next step is to chose the design we want to use and if required, set the parameters to the values we need or want.
In this example, we are using a `FullFactorial` design:
===#

design = FullFactorial([5, 5])

#===
After that, we call the sample function with our design.
This produces a matrix containing the points of our design fitted to the range defined via the RandomVariables.
Wer then evaluate the function we want to model in these points and use the resulting data to train a `ResponseSurface`.
The `ResponseSurface` uses regression to fit a polynomial function to the given datapoints.
That functions degree is set as an Integer in the constructor.
===#

#md # !!! note
#md #     The choice of the degree and the design and its parameters may be crucial to obtaining a sufficient model.

training_data = sample(x, design)
evaluate!(himmelblau, training_data)
basis = MonomialBasis(2, 4)
rs = LinearBasisFunctionModel(training_data, :y, basis)

test_data = sample(x, 1000)
evaluate!(rs, test_data)

#===
To evaluate the `ResponseSurface`use `evaluate!(rs::ResponseSurface, data::DataFrame)` with the `DataFrame` containing the points you want to evaluate.

The model in this case has an mse of about 1e-26 and looks like this in comparison to the original:
===#

#md f(x1, x2) = sum(rs.monomials([x1, x2]) .* rs.β) #hide
#md s2 = surface(a, b, f; plot_title="Response Surface", plot_titlefontsize=16) #hide
#md surface(s1, s2; layout=(1, 2), legend=false, size=(800, 400))  #hide
#md savefig("himmelblau-comparison.svg"); nothing # hide

# ![](himmelblau-comparison.svg)

#jl p_data = test_data[:, [:x1, :x2]]
#jl evaluate!(himmelblau, p_data)

#jl mse = mean((p_data.y .- test_data.y) .^ 2)
#jl println("MSE is:  $mse")
