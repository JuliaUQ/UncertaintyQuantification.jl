```@raw html
---
authors:
  - name: Jasper Behrensdorf
    platform: github
    link: https://github.com/FriesischScott
  - name: Matteo Broggi
    platform: github
    link: https://github.com/teobros
  - name: Lukas Fritsch
    platform: github
    link: https://github.com/lukasfritsch
  - name: Ander Gray
    platform: github
    link: https://github.com/AnderGray
  - name: Jan Grashorn
    platform: github
    link: https://github.com/jgrashorn
  - name: Laurenz Knipper
    platform: github
    link: https://github.com/sitoryu
  - name: Max Luttmann
    platform: github
    link: https://github.com/mlsuh
  - name: Felix Mett
    platform: github
    link: https://github.com/felixmett
  - name: Andrea Perin
    platform: github
    link: https://github.com/andreaperin
  - name: Thomas Potthast
    platform: github
    link: https://github.com/potthastT
---
```

# UncertaintyQuantification.jl

A Julia package for uncertainty quantification.

## Authors

```@raw html

<Authors />
```

## Features

Current functionality includes:

* Simulation-based reliability analysis
  * Monte Carlo simulation
  * Quasi Monte Carlo simulation (Sobol, Halton)
  * (Advanced) Line Sampling
  * Subset Simulation
* Sensitivity analysis
  * Gradients
  * Sobol indices
* Metamodeling
  * Polyharmonic splines
  * Response Surface
  * Polynomial Chaos Expansion
* Bayesian Updating
* Third-party solvers
  * Connect to any solver by injecting random samples into source files
  * HPC interfacing with slurm
* Stochastic Dynamics
  * Power Spectral Density Estimation
  * Stochastic Process Generation

---

## Installation

To install the latest release through the Julia package manager run:

```julia
julia> ]add UncertaintyQuantification
julia> using UncertaintyQuantification
```

or install the latest development version with:

```julia
julia> ]add UncertaintyQuantification#master
julia> using UncertaintyQuantification
```

---

### Related packages

* [OpenCossan](https://github.com/cossan-working-group/OpenCossan): Matlab-based toolbox for uncertainty quantification and management
