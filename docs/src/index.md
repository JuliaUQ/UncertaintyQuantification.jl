# UncertaintyQuantification.jl

A Julia package for uncertainty quantification.

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
