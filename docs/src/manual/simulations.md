# Simulations

## Monte Carlo

The Monte-Carlo (MC) method is a method of sampling random numbers that dates back to 1777. The name was suggested by Nick Metropolis when MC was used while working on the Manhattan Project.
It is used in Random Number Generators which generally produce pseudo random numbers.
Extensive information on the Monte-Carlo method can be found in [zioMonteCarlo2013](@cite).

## Quasi Monte Carlo

For a comprehensive introduction to Quasi Monte Carlo, view [leobacherQuasiMonteCarlo2026](@cite). The following provides an essential overview.

Quasi Monte Carlo (QMC), is a method of producing samples similar to those generated via Monte Carlo (MC).
The difference being that QMC samples are generated deterministically in a way to ensure they are evenly distributed across the sampling space, not forming clutters or voids as MC samples might.
This makes QMC more efficient than MC for lots of applications, since fewer samples are needed in order to produce a sufficient density of samples throughout. There are multiple ways of QMC sampling methods which can be classified as either digital nets/sequences or lattices. 
Since the samples are not random, they are not suitable for many tasks. However, it is possible to randomize QMC samples by randomly shifting or scrambling each sample.

While the disjunction of a deterministic sampling and a randomization method that can be applied to the samples later is most common for QMC, there are also some sampling methods that directly produce quasi random samples.

This package relies on [`QuasiMonteCarlo.jl`](https://github.com/SciML/QuasiMonteCarlo.jl) which offers various QMC sampling methods as well as corresponding randomization methods. An example on how to use them is shown below.

The package gives access to these deterministic sampling algorithms: Lattices: `GridSample`, `LatticeRuleSample`, `GoldenSample` and `KroneckerSample`. Digital nets: `SobolSample`, `FaureSample`, `HaltonSample`.

!!! note 
    `KroneckerSample` should not be used with bases > 3. `GoldenSample` is a special version of a `KroneckerSample` and should be used with base 1 only. Otherwise, `LatticeRuleSample` is the preferred method. For more information on this and on the implementation of the other sampling algorithms, please consult the corresponding [documentation of QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/dev/samplers/).

!!! note
    The number of samples for digital nets should always be a power of the base, while lattices work best with large prime numbers as sample-sizes. Straying from these rules may worsen results significantly, since the samples will be imbalanced [owenQuasiMonteCarlo2021](@cite).

The samples can be randomized by shifting with `Shift` (Cranley-Patterson rotation [cranleyPattersonRandomization1976](@cite)). or by scrambling via `MatousekScramble`[matousekScramble1998](@cite), `OwenScramble`[owenScramble1995](@cite) or `DigitalShift`. The latter functions similarly to `Shift`, but takes the base that was used during sampling into account. 

!!! note
    For randomizing lattice-samples, `Shift` is recommended. The mentioned scrambling methods work well with samples from digital nets.

Further, `LatinHypercubeSample`[mcKayLatinHypercubeSampling1979](@cite) and `RandomizedHaltonSample`[owenRandomizedHalton2017](@cite) are the available randomized sampling algorithms.

To facilitate QMC-sampling in `UncertaintyQuantification.jl` one needs to first create one or more random variables to sample from as well as an instance of `QuasiMonteCarloSampling`, which takes the number of samples and the desired sampling algorithm from `QuasiMonteCarlo.jl`. Using the `sample`-function, the defined number of samples is created from the given variable(s).

```@setup qmc
    using UncertaintyQuantification
```

```@example qmc
    using QuasiMonteCarlo 
    x = RandomVariable(Uniform(), :x)
    qmc = QuasiMonteCarloSampling(128, LatinHypercubeSample())
    samples = UncertaintyQuantification.sample(x, qmc)
    nothing    # hide
```
It is of course possible to directly create the `QuasiMonteCarloSampling`-instance inside the `sample`-call, enabling a more efficient version of the example above which looks like this:

```@example qmc
    x = RandomVariable(Uniform(), :x)
    samples = UncertaintyQuantification.sample(x, QuasiMonteCarloSampling(128, LatinHypercubeSample()))
    nothing    # hide
```

!!! note
    When chosing `n`, be reminded that, for digital nets, `n` must be a power of the base that is used for creating the respective sequence. For `SobolSample` the base is always equal to 2 while for `FaureSample`, it depends on the number of input-variables, being the smallest prime number that is greater or equal to the number of variables. 

```@example qmc
    x = RandomVariable(Uniform(), :x)
    samples = QuasiMonteCarloSampling(128, QuasiMonteCarlo.SobolSample())
```

To emphasize the importance of randomization, take a look at the correlations that might occur using unrandomized qmc and how they are fixed by randomizing:

This is the 7th dimension plotted against the 8th in Faure Sampling, unrandomized vs. randomized via Owen Scramble:

```@setup plots
    using UncertaintyQuantification #hide
    using Plots #hide
    using QuasiMonteCarlo #hide

    x = RandomVariable.(Uniform(), [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8]) #hide
    samples = UncertaintyQuantification.sample(x, QuasiMonteCarloSampling(1331, FaureSample())) #hide
    p1 = plot(samples[:, 7], samples[:, 8],seriestype=:scatter, label="unrandomized") #hide
    rand_samples = UncertaintyQuantification.sample(x,QuasiMonteCarloSampling(1331, FaureSample(R=OwenScramble(base=11)))) #hide
    p2 = plot(rand_samples[:, 7], rand_samples[:, 8],seriestype=:scatter, label="randomized") #hide
    y = RandomVariable.(Uniform(), [:y1, :y2]) #hide
    y_samples = UncertaintyQuantification.sample(y, 1331) #hide
    p3 = plot(y_samples[:, 1], y_samples[:, 2], seriestype=:scatter, label="monte-carlo") #hide
```

```@example plots
    plot(p1, p2, p3; layout=(2, 2),  size=(800, 800)) # hide
    savefig("faure-sequence.svg"); nothing # hide
```

![](faure-sequence.svg)
