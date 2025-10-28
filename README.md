UncertaintyQuantification.jl
============================

![Build Status](https://github.com/JuliaUQ/UncertaintyQuantification.jl/workflows/CI/badge.svg)
[![Coverage Status](https://codecov.io/gh/JuliaUQ/UncertaintyQuantification.jl/branch/master/graph/badge.svg?token=LfslMAoWvA)](https://codecov.io/gh/JuliaUQ/UncertaintyQuantification.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3993816.svg)](https://doi.org/10.5281/zenodo.3993816)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliauq.github.io/UncertaintyQuantification.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliauq.github.io/UncertaintyQuantification.jl/dev)

A Julia package for uncertainty quantification. Current functionality includes:

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
