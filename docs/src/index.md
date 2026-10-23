```@raw html
---
layout: home

hero:
  name: "Uncertainty\nQuantification.jl"
  text: Computational Methods for Uncertain Systems
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
    details: Estimate (small) failure probabilities with advanced Monte Carlo methods.
    link: /manual/reliability
  - title: Imprecise Probabilities
    details: Model epistemic and mixed uncertainty using intervals and probability boxes.
    link: /manual/gettingstarted#Imprecise-Probabilities
  - title: Metamodeling
    details: Approximate expensive models with surrogates.
    link: /manual/metamodels
  - title: Bayesian Updating
    details: Update uncertain model parameters using observed data and Bayesian inference.
    link: /manual/bayesianupdating
  - title: External Solvers and HPC
    details: Connect third-party solvers into UQ workflows.
    link: /manual/hpc
  - title: Stochastic Dynamics
    details: Model and analyze dynamic systems under uncertainty.
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
