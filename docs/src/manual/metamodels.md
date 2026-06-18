# Metamodels

## Linear Basis Function Models 

Linear basis function models are a simple class of metamodels that express the predicted output as a linear combination of basis functions evaluated for the input variables defined as

```math
y(\mathbf{x}) = \sum_{i=1}^{n} \beta_i \varphi_i(\mathbf{x}),
```

where ``\varphi_i(x)`` are nonlinear basis functions that map the input space into an intermediate feature space and ``\mathbf{\beta}`` represents the adjustable weights. Some commonly used basis functions are introduced next.

### Monomial Basis

Monomial basis functions are defined as the powers, or products of powers in the multivariate case, of the input variables with a total degree of less than or equal to ``d``. For example, the monomial basis in two variables of degree ``d=3`` in graded reverse lexicographic order is given by

```math
\varphi(x) = \left[1, x_2, x_1, x_2^2, x_1x_2, x_1^2, x_2^3, x_1x_2^2, x_1^2x_2, x_1³\right]
```

The construction of this [`MonomialBasis`](@ref) is presented next.

```@example monomials
using UncertaintyQuantification # hide
φ = MonomialBasis(2, 3)
```

By default the [`MonomialBasis`](@ref) includes the constant (zero degree) term. This behaviour can be changed by passing the `include_zero=false` keyword.

### Radial Basis

A radial basis function (RBF) is a real-valued function ``\varphi(||\mathbf{x}-\mathbf{c}||)`` that depends on the distance between the input ``\mathbf{x}`` and a fixed point ``\mathbf{c}``, called a *center*. These functions are multivariate but reduce to scalar functions of the radius ``r = ||\mathbf{x} - \mathbf{c}||``, hence the name **radial** basis function. The distance used is most commonly the euclidian norm. *UncertaintyQuantification* provides two types of RBFs.

#### Gaussian

```math

\varphi(r) = \exp(-\epsilon r)

```

here, ``\epsilon`` is a shape parameter. Gaussian radial basis functions are exposed through the [`GaussianRadialBasis`](@ref) struct.

#### Polyharmonic

```math
\varphi(r) = \begin{cases}
    r^k & \text{with} & k = 1,3,5,\ldots, \\
    r^k\ln(r) & \text{with} & k = 2,4,6,\ldots
\end{cases}
```

Note, that polyharmonic radial basis functions do not require a shape parameter. These RBs can be constructed using the [`PolyharmonicRadialBasis`](@ref) type.

### Least Squares

Despite the nonlinearity of the basis functions, the model remains linear in its parameters, which allows efficient estimation of the weights using ordinary least squares. Given ``m`` observations ``(\mathbf{x}_i, y_i)``, for ``i = 1,\ldots,m`` we construct the design matrix ``\Phi`` where each row contains the evaluated basis functions for one input:

```math
\Phi_{ij} = \varphi_j(\mathbf{x}_i), \text{for} i=1,\ldots,n \text{and} j=1,\ldots,m
```

The optimal weight vector is found by minimizing the sum-of-squares error

```math
E(\mathbf{\beta}) = \sum_{i=1}^{m} (y(\mathbf{x}_i, \mathbf{\beta}) - y_i)^2
```

yielding the closed-form solution via

```math
\mathbf{\beta} = (\Phi^{\top}\Phi)^{-1}\Phi^{\top}\mathbf{y},
```

where ``\mathbf{y}`` is the output vector.

### Example

Consider the function

```math
    y(x) = x\cos(x) ,
```

where ``x \sim U(-5, 5)``

We use [`HaltonSampling`](@ref) to sample 150 data points from the input variable, evaluate the model, and fit a [`LinearBasisFunctionModel`](@ref) using a [`MonomialBasis`](@ref) of degree ``d=9``.

```@example lbfm
using UncertaintyQuantification # hide
x = RandomVariable(Uniform(-5, 5), :x)
y = Model(
        df -> df.x .* cos.(df.x),
        :y,
    )
data = sample(x, HaltonSampling(150))
evaluate!(y, data)
lbfm = LinearBasisFunctionModel(data, :y, MonomialBasis(1, 9))
```

A plot comparing the resulting model to the data points is presented next.

```@example lbfm
using Plots # hide
using DataFrames # hide
scatter(data.x, data.y; markershape=:xcross, color=:red, label="data") # hide
plot_data = DataFrame(x = collect(range(-5.5, 5.5; length=1000))) # hide
evaluate!(lbfm, plot_data) # hide
plot!(plot_data.x, plot_data.y, color=:blue, label="model") # hide
savefig("lbfm.svg") # hide
```

![Linear Basis Function Model Plot](lbfm.svg)

### Response Surface

A linear basis function model constructed from a [`MonomialBasis`](@ref) is also known as a polynomial *Response Surface* [khuriResponseSurfaceMethodology2010](@cite). For this reason we provide a convenient alias [`ResponseSurface`](@ref). Using this alias the previous example can be adapted as follows.

```@example lbfm
rs = ResponseSurface(data, :y, 9)
```

### Design Of Experiments

Several experimental designs have been developed to efficiently estimate [`ResponseSurface`](@ref) models [khuriResponseSurfaceMethodology2010](@cite). Although designed for response surface methodology these designs can be used to fit any metamodel. However, for more complex models we suggest using [Quasi Monte Carlo](@ref) sampling schemes instead.

The designs implemented in *UncertaintyQuantification* are `TwoLevelFactorial`, `FullFactorial`, `FractionalFactorial`, `CentralComposite`, `BoxBehnken`, and `PlackettBurman`.

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

where ``x \sim U(-5.5, 5.5`` and ``g \sim N(0,1)``. We generate a data sequence of ``N = 150`` points and fit an `IntervalPredictorModel` using a [`MonomialBasis`](@ref) of sixth degree.

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

