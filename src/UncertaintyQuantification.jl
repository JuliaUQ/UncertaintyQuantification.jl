module UncertaintyQuantification

using Bootstrap
using CovarianceEstimation
using DataFrames
using Dates
using Dierckx
using Distributed
using FastGaussQuadrature
using FiniteDifferences
using Format
using LinearAlgebra
using MeshAdaptiveDirectSearch
using Monomials
using Mustache
using Optim
using Primes
using QuadGK
using QuasiMonteCarlo
using Random
using Reexport
using Roots
using StatsBase

@reexport using Distributions

import Base: rand, names, copy, run, length
import Distributions: cdf, quantile, pdf, logpdf, minimum, maximum, insupport, mean, var
import Statistics: mean, var

abstract type UQInput end
abstract type DeterministicUQInput <: UQInput end
abstract type RandomUQInput <: UQInput end

"""
Abstract supertype for all model types
"""
abstract type UQModel end

abstract type Copula end

abstract type AbstractSimulation end
abstract type AbstractMonteCarlo <: AbstractSimulation end
abstract type AbstractQuasiMonteCarlo <: AbstractMonteCarlo end

"""
    AbstractBayesianMethod

Subtypes are used to dispatch to the different MCMC methods in [`bayesianupdating`](@ref).

Subtypes are:

- [`SingleComponentMetropolisHastings`](@ref)
- [`TransitionalMarkovChainMonteCarlo`](@ref)
"""
abstract type AbstractBayesianMethod end

"""
    AbstractBayesianPointEstimate

Subtypes are used to dispatch to the different point estimation methods in [`bayesianupdating`](@ref).

Subtypes are:

- [`MaximumAPosterioriBayesian`](@ref)
- [`MaximumLikelihoodBayesian`](@ref)
"""
abstract type AbstractBayesianPointEstimate end
abstract type AbstractDesignOfExperiments end

abstract type AbstractHPCScheduler end

# Types
export AbstractBayesianMethod
export AbstractBayesianPointEstimate
export AbstractDesignOfExperiments
export AbstractMonteCarlo
export AbstractPowerSpectralDensity
export AbstractStochasticProcess
export AbstractQuasiMonteCarlo
export AbstractSimulation
export Copula
export DeterministicUQInput
export RandomUQInput
export UQInput
export UQModel

# Structs
export AdvancedLineSampling
export EmpiricalDistribution
export BackwardFiniteDifferences
export BinnedData
export BoxBehnken
export CentralComposite
export CentralFiniteDifferences
export CloughPenzien
export DoubleLoop
export EmpiricalPSD
export ExternalModel
export SlurmInterface
export Extractor
export FaureSampling
export FORM
export ForwardFiniteDifferences
export FractionalFactorial
export FullFactorial
export GaussianCopula
export GaussianMixtureModel
export GaussQuadrature
export HaltonSampling
export HermiteBasis
export ImportanceSampling
export Interval
export IntervalVariable
export JointDistribution
export KanaiTajimi
export LatinHypercubeSampling
export LatticeRuleSampling
export LeastSquares
export WeightedApproximateFetekePoints
export LegendreBasis
export LineSampling
export SingleComponentMetropolisHastings
export MaximumAPosterioriBayesian
export MaximumLikelihoodBayesian
export Model
export MonteCarlo
export ParallelModel
export Parameter
export PlackettBurman
export PolynomialChaosBasis
export PolynomialChaosExpansion
export PolyharmonicSpline
export ProbabilityBox
export RadialBasedImportanceSampling
export RandomVariable
export RandomSlicing
export ResponseSurface
export ShinozukaDeodatis
export SobolSampling
export Solver
export SpectralRepresentation
export StochasticProcessModel
export SubSetInfinity
export SubSetInfinityAdaptive
export SubSetSimulation
export TransitionalMarkovChainMonteCarlo
export TwoLevelFactorial

# Methods
export bayesianupdating
export calc
export count_rvs
export dimensions
export distribution_parameters
export doe_samples
export evaluate
export evaluate!
export gradient
export gradient_in_standard_normal_space
export linear_binning
export mean
export multivariate_indices
export periodogram
export polynomialchaos
export probability_of_failure
export propagate_intervals!
export qmc_samples
export quadrature_nodes
export quadrature_weights
export rand
export sample
export sample_conditional_copula
export sobolindices
export kucherenkoindices
export kucherenkoindices_bin
export to_copula_space
export to_physical_space!
export to_standard_normal_space
export to_standard_normal_space!

include("util/binning.jl")
include("util/fourier-transform.jl")
include("util/wrap.jl")
include("util/imprecise.jl")
include("util/kde.jl")

include("inputs/empiricaldistribution.jl")
include("inputs/inputs.jl")
include("inputs/parameter.jl")

include("inputs/imprecise/interval.jl")
include("inputs/imprecise/p-box.jl")

include("inputs/randomvariables/randomvariable.jl")
include("inputs/randomvariables/distributionparameters.jl")
include("inputs/copulas/gaussian.jl")
include("inputs/gaussianmixtures.jl")
include("inputs/jointdistribution.jl")

include("dynamics/psd.jl")
include("inputs/stochasticprocesses/spectralrepresentation.jl")
include("inputs/stochasticprocesses/models.jl")

include("models/external/solver.jl")
include("models/external/extractor.jl")
include("models/external/externalmodel.jl")
include("models/model.jl")
include("models/imprecise/propagation.jl")
include("models/polyharmonicspline.jl")
include("models/responsesurface.jl")
include("models//slicingmodel.jl")

include("hpc/slurm.jl")

include("models/pce/pcebases.jl")
include("models/pce/polynomialchaosexpansion.jl")

include("modelupdating/bayesianMAP.jl")
include("modelupdating/bayesianupdating.jl")

include("sensitivity/finitedifferences.jl")
include("sensitivity/gradient.jl")

include("simulations/doe.jl")
include("simulations/linesampling.jl")
include("simulations/montecarlo.jl")
include("simulations/radialbasedimportancesampling.jl")
include("simulations/subset.jl")

include("reliability/form.jl")
include("simulations/importancesampling.jl")
include("reliability/probabilityoffailure.jl")
include("reliability/probabilityoffailure_imprecise.jl")
include("sensitivity/sobolindices.jl")
include("sensitivity/kucherenkoindices.jl")

end
