<h1 align="center">
<img src="docs/src/assets/logo.svg" alt="alt text" width="60" height="60" align="center"> UncertaintyQuantification.jl
</h1>

<div align="center">

| Category | Badges |
|---|---|
| Build status | ![Build Status](https://github.com/JuliaUQ/UncertaintyQuantification.jl/actions/workflows/ci.yml/badge.svg?branch=master) [![Coverage Status](https://codecov.io/gh/JuliaUQ/UncertaintyQuantification.jl/branch/master/graph/badge.svg?token=LfslMAoWvA)](https://codecov.io/gh/JuliaUQ/UncertaintyQuantification.jl) |
| Documentation | [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliauq.github.io/UncertaintyQuantification.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliauq.github.io/UncertaintyQuantification.jl/dev) |
| Code quality | [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) |
| References | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3993816.svg)](https://doi.org/10.5281/zenodo.3993816) [![Static Badge](https://img.shields.io/badge/Paper-10.7712%2F120223.10347.19810-yellow)](https://www.eccomasproceedia.org/conferences/thematic-conferences/uncecomp-2023/10347)|

</div>

A Julia package for uncertainty quantification. Current functionality includes:

* Simulation-based reliability analysis
  * Monte Carlo simulation
  * Quasi Monte Carlo simulation (Sobol, Halton)
  * (Advanced) Line Sampling
  * Subset Simulation
  * Propagation of imprecise probabilites
* Sensitivity analysis
  * Gradients
  * Sobol indices
* Metamodeling
  * Polyharmonic splines
  * Response Surface Methodology
  * Radial basis function networks
  * Polynomial Chaos Expansion
  * Gaussian Processes
  * Interval Predictor Models
* Bayesian Updating
* Third-party solvers
  * Connect to any solver by injecting random samples into source files
  * HPC interfacing with slurm
* Stochastic Dynamics
  * Power Spectral Density Estimation
  * Stochastic Process Generation
