# Metamodels

## Design Of Experiments

Design Of Experiments (DOE) offers various designs that can be used for creating a model of a given system. The core idea is to evaluate significant points of the system in order to obtain a sufficient model while keeping the effort to achieve this relatively low. Depending on the parameters, their individual importance and interconnections, different designs may be adequate.

The ones implemented here are `TwoLevelFactorial`, `FullFactorial`, `FractionalFactorial`, `CentralComposite`, `BoxBehnken` and `PlackettBurman`.

## Response Surface

A Response Surface is a simple polynomial surrogate model. It can be trained by providing it with evaluated points of a function or any of the aforementioned experimental designs.

## Interval Predictor Model

An interval predictor model (IPM)[crespoIntervalPredictorModels2016](@cite) is a function that returns an interval instead of a precise value for the dependent variable given as

```math
    I_y(x,P) = \{y = p^T \varphi (x), p \in P\},
```

where ``\varphi(x)`` is an arbitrary basis and the uncertainty set ``P`` is defined as

```math
    P = \{p : \underline{p} \leq p \leq \overline{p} \}.
```

Using the *defining vertices*  of P ``\underline{p}`` and ``\overline{p}`` the IPM results as

```math
I_y(x,P) = \left[\underline{y}(x,\overline{p},\underline{p}), \overline{y}(x,\overline{p},\underline{p})\right],
```

with

```math
    \underline{y}(x,\overline{p},\underline{p}) = \overline{p}^T\left(\frac{\varphi(x) - |\varphi(x)|}{2}\right) + \underline{p}^T\left(\frac{\varphi(x) + |\varphi(x)|}{2}\right)
```

and

```math
    \overline{y}(x,\overline{p},\underline{p}) = \overline{p}^T\left(\frac{\varphi(x) + |\varphi(x)|}{2}\right) + \underline{p}^T\left(\frac{\varphi(x) - |\varphi(x)|}{2}\right).
```

Here, ``\underline{y}`` and ``\overline{y}`` are the lower and upper bounds of the IPM.

The distance between the lower and upper bound given by

```math
\delta_y(x,\overline{p},\underline{p}) = (\overline{p} - \underline{p})^T|\varphi(x)|
```

is known as the *spread* of the IPM. The optimal defining vertices for a given data set are found by minimizing the average spread such that all data points fall into the IPM by solving the following convex constrained optimization problem.

```math
\{\underline{p},\overline{p}\} = \underset{u,v}{\operatorname{argmax}}\{\mathbb{E}_x\left[\delta_y(x,u,v\right] : \underline{y}(x_i,v,u)\leq y_i \leq\overline{y}(x_i,v,u),u \leq v\}
```

### Example

Consider the function

```math
    y(x) = x^2\cos(x) - \sin(3x)\exp(-x^2) - x - \cos(x^2) +xg,
```

where ``x \sim U(-5.5, 5.5`` and ``g \sim N(0,1)``. We generate a data sequence of ``N = 150`` points and fit an `IntervalPredictorModel` using a `MonomialBasis` of sixth degree.

```@example ipm
using UncertaintyQuantification # hide

x = RandomVariable(Uniform(-5.5, 5.5), :x)
data = sample(x, HaltonSampling(150))

m = Model(
        df ->
            df.x .^ 2 .* cos.(df.x) .- sin.(3 * df.x) .* exp.(-df.x .^ 2) .- df.x .-
            cos.(df.x .^ 2) .+ df.x .* randn(size(df, 1)),
        :y,
    )

evaluate!(m, data)

b = MonomialBasis(1,6)
ipm = IntervalPredictorModel(data, :y, b)
```

The following figure presents the bounds of the resulting IPM and the corresponding least squares solution. Note, that the least squares solution is not guaranteed to be between the bounds of the IPM.

```@example ipm
using Plots # hide
using DataFrames # hide
ls = LinearBasisFunctionModel(data, :y, b) # hide

ipm_data = DataFrame(x = range(-5.5, 5.5; length=1000)) # hide
ls_data = copy(ipm_data) # hide

evaluate!(ipm, ipm_data) # hide
evaluate!(ls, ls_data) # hide

scatter(data.x, data.y; markershape=:xcross, color=:red, label="data") # hide
plot!(ipm_data.x, getproperty.(ipm_data.y, :lb), linestyle=:dash, color=:black, label="IPM bounds") # hide
plot!(ipm_data.x, getproperty.(ipm_data.y, :ub), linestyle=:dash, color=:black, label="") # hide
plot!(ls_data.x, ls_data.y; color=:blue, label ="LS") # hide

savefig("ipm.svg"); nothing # hide

```

![IPM Plot](ipm.svg)

