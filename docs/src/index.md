```@raw html
---
layout: home

hero:
  name: "Uncertainty\nQuantification.jl"
  text: Computational methods for uncertain systems
  tagline: Propagate uncertainties, assess reliability, and calibrate models with data.
  image:
    src: /logo.svg
    alt: UncertaintyQuantification.jl logo
  actions:
    - theme: brand
      text: Get Started
      link: /manual/gettingstarted
    - theme: alt
      text: Installation
      link: /#Installation
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaUQ/UncertaintyQuantification.jl

features:
  - title: Reliability Analysis
    details: Estimate failure probabilities with Monte Carlo and quasi Monte Carlo (Sobol, Halton), advanced line sampling, and subset simulation.
    link: /manual/reliability
  - title: Sensitivity Analysis
    details: Identify influential inputs using gradients and Sobol indices.
  - title: Metamodeling
    details: Approximate expensive models with polyharmonic splines, response surfaces, and polynomial chaos expansions.
    link: /manual/metamodels
  - title: Bayesian Updating
    details: Update uncertain model parameters using observed data and Bayesian inference.
    link: /manual/bayesianupdating
  - title: External Solvers and HPC
    details: Connect third-party solvers by injecting random samples into input files and run analyses on HPC clusters with Slurm.
    link: /manual/hpc
  - title: Stochastic Dynamics
    details: Estimate power spectral densities and generate stochastic processes for dynamic analyses.
    link: /manual/dynamics
---
```

## Installation

To install the latest release through the Julia package manager run:

```julia-repl
julia> ]add UncertaintyQuantification
julia> using UncertaintyQuantification
```

or install the latest development version with:

```julia-repl
julia> ]add UncertaintyQuantification#master
julia> using UncertaintyQuantification
```

---

### Related packages

* [OpenCossan](https://github.com/cossan-working-group/OpenCossan): Matlab-based toolbox for uncertainty quantification and management
