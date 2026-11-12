module UncertaintyQuantification

using ADTypes
using Bootstrap
using Clarabel
using Copulas
using CovarianceEstimation
using DataFrames
using Dates
using Dierckx
using DifferentiationInterface
using Distributed
using FastGaussQuadrature
using FiniteDifferences
using Format
using JuMP
using LinearAlgebra
using MeshAdaptiveDirectSearch
using Monomials
using Mooncake: Mooncake
using Mustache
using Optim
using ParameterHandling
using Primes
using QuadGK
using QuasiMonteCarlo
using Random
using Reexport
using Roots
using StatsBase
using TransportMaps
using RecipesBase

import NLSolversBase

@reexport using AbstractGPs
@reexport using Distributions

import AbstractGPs: sampleplot, sampleplot!
import Base: rand, names, copy, run, length, eltype
import Copulas: Copula
import DifferentiationInterface: derivative, jacobian, pullback, pushforward
import Distributions:
    cdf, quantile, pdf, logpdf, minimum, maximum, insupport, mean, var, sampler, std,
    median, kldivergence, mode, shape
import Statistics: mean, var, std
import TransportMaps: AbstractMapDensity, AbstractQuadratureWeights, logpdf, grad_logpdf, optimize!

abstract type UQInput end
abstract type DeterministicUQInput <: UQInput end
abstract type RandomUQInput <: UQInput end

"""
Abstract supertype for all model types
"""
abstract type UQModel end

abstract type AbstractBasis end

abstract type AbstractSimulation end
abstract type AbstractMonteCarlo <: AbstractSimulation end
abstract type AbstractMarkovChainMonteCarlo end

"""
	AbstractBayesianMethod

Subtypes configure Bayesian updating algorithms. General MCMC samplers and
`LaplaceEstimate` can also be passed to [`bayesianupdating`](@ref).

Subtypes are:

- [`TransitionalMarkovChainMonteCarlo`](@ref)
- [`MaximumAPosterioriBayesian`](@ref)
- [`TransportMapBayesian`](@ref)
"""
abstract type AbstractBayesianMethod end

"""
	AbstractBayesianPointEstimate

Subtypes are used to dispatch to the different point estimation methods in [`bayesianupdating`](@ref).

Subtypes are:

- [`MaximumAPosterioriBayesian`](@ref)
"""
abstract type AbstractBayesianPointEstimate <: AbstractBayesianMethod end
abstract type AbstractDesignOfExperiments end

abstract type AbstractHPCScheduler end

abstract type AbstractTransportMap <: ContinuousMultivariateDistribution end

abstract type AbstractAcquisitionFunction end
abstract type AbstractGaussianProcessAcquisitionFunction <: AbstractAcquisitionFunction end

# Types
export AbstractAcquisitionFunction
export AbstractGaussianProcessAcquisitionFunction
export AbstractBayesianMethod
export AbstractBayesianPointEstimate
export AbstractDesignOfExperiments
export AbstractMonteCarlo
export AbstractPowerSpectralDensity
export AbstractStochasticProcess
export AbstractSimulation
export AbstractTransportMap
export Copula
export DeterministicUQInput
export RandomUQInput
export UQInput
export UQModel

# Structs
export MetropolisHastings
export AdaptiveMetropolisHastings
export AdvancedLineSampling
export AdaptiveGaussianProcess
export EmpiricalDistribution
export BackwardFiniteDifferences
export LinearBasisFunctionModel
export BinnedData
export BoxBehnken
export CentralComposite
export CentralFiniteDifferences
export CloughPenzien
export DeviationNumber
export DoubleLoop
export EmpiricalPSD
export ExternalModel
export SlurmInterface
export ExpectedFeasibility
export ExpectedImprovement
export ExpectedImprovementForGlobalFit
export Extractor
export FORM
export ForwardFiniteDifferences
export FractionalFactorial
export FullFactorial
export GaussianMixtureModel
export GaussianProcess
export GaussQuadrature
export HermiteBasis
export ImportanceSampling
export Interval
export IntervalVariable
export IntervalPredictorModel
export JointDistribution
export KanaiTajimi
export LaplaceEstimate
export LeastSquares
export WeightedApproximateFetekePoints
export LegendreBasis
export LineSampling
export SingleComponentMetropolisHastings
export MaximumAPosterioriBayesian
export MaximumLikelihoodEstimate
export MaximumLikelihoodEstimation
export MaximinDistance
export MaximumVariance
export Model
export MonomialBasis
export MonteCarlo
export Parameter
export PlackettBurman
export PolynomialChaosBasis
export PolynomialChaosExpansion
export PolyharmonicRadialBasis
export PolyharmonicSpline
export ProbabilityBox
export ProbabilityOfImprovement
export RadialBasedImportanceSampling
export GaussianRadialBasis
export RandomVariable
export RandomSlicing
export ResponseSurface
export ShinozukaDeodatis
export Solver
export SpectralRepresentation
export StochasticProcessModel
export SubSetInfinity
export SubSetInfinityAdaptive
export SubSetSimulation
export TransitionalMarkovChainMonteCarlo
export TransportMap
export TransportMapFromSamples
export TransportMapBayesian
export TwoLevelFactorial
export UpperConfidenceBound
export QuasiMonteCarloSampling

# Methods
export bayesianupdating
export count_rvs
export dimensions
export distribution_parameters
export doe_samples
export double_samples
export evaluate
export evaluate!
export gaussian_likelihood
export gradient
export gradient_in_standard_normal_space
export isimprecise
export linear_binning
export laplaceestimate
export maximumlikelihood
export logpdf
export mapfromdensity
export mapfromsamples
export mean
export multivariate_indices
export optimize_hyperparameters
export pdf
export periodogram
export polynomialchaos
export probability_of_failure
export propagate_intervals!
export quadrature_nodes
export quadrature_weights
export rand
export refine!
export reliability
export sample
export sample!
export sobolindices
export to_physical_space!
export to_standard_normal_space!
export variancediagnostic

include("util/binning.jl")
include("util/fourier-transform.jl")
include("util/wrap.jl")
include("util/kde.jl")

include("inputs/empiricaldistribution.jl")
include("inputs/inputs.jl")
include("inputs/parameter.jl")

include("inputs/imprecise/interval.jl")
include("inputs/imprecise/p-box.jl")

include("inputs/randomvariables/randomvariable.jl")
include("inputs/randomvariables/distributionparameters.jl")
include("inputs/gaussianmixtures.jl")
include("inputs/jointdistribution.jl")
include("inputs/laplaceestimate.jl")
include("inputs/transportmaps.jl")

include("dynamics/psd.jl")
include("inputs/stochasticprocesses/spectralrepresentation.jl")
include("inputs/stochasticprocesses/models.jl")

include("models/basisfunctions/monomialbasis.jl")
include("models/basisfunctions/radialbasis.jl")
include("models/basisfunctions/basisfunctionmodels.jl")
include("models/external/solver.jl")
include("models/external/extractor.jl")
include("models/external/externalmodel.jl")
include("models/model.jl")
include("models/imprecise/propagation.jl")
include("models/polyharmonicspline.jl")
include("models/responsesurface.jl")
include("models/slicingmodel.jl")
include("models/gp/parameterization.jl")
include("models/gp/hyperparametertuning.jl")
include("models/gp/gaussianprocess.jl")
include("models/gp/adaptivegaussianprocess.jl")
include("models/gp/gp_acquisitionfunction.jl")
include("models/ipm.jl")
include("models/models.jl")

include("hpc/slurm.jl")

include("models/pce/pcebases.jl")
include("models/pce/polynomialchaosexpansion.jl")

include("sensitivity/finitedifferences.jl")
include("sensitivity/gradient.jl")

include("simulations/doe.jl")
include("simulations/linesampling.jl")
include("simulations/montecarlo.jl")
include("simulations/radialbasedimportancesampling.jl")
include("simulations/MCMC/singlecomponentmetropolishastings.jl")
include("simulations/MCMC/adaptivemetropolishastings.jl")
include("simulations/MCMC/metropolishastings.jl")
include("simulations/subset.jl")

#load after simulations to avoid circular dependency
include("modelupdating/bayesian/markovchainmontecarlo.jl")
include("modelupdating/bayesian/transitionalmarkovchainmontecarlo.jl")
include("modelupdating/bayesian/laplaceestimate.jl")
include("modelupdating/bayesian/maximumaposteriori.jl")
include("modelupdating/bayesian/transportmap.jl")
include("modelupdating/maximumlikelihood.jl")
include("modelupdating/bayesian/bayesianupdating.jl")
include("modelupdating/likelihoods.jl")

include("reliability/form.jl")
include("simulations/importancesampling.jl")
include("reliability/probabilityoffailure.jl")
include("reliability/probabilityoffailure_imprecise.jl")
include("sensitivity/sobolindices.jl")

include("util/imprecise.jl")

include("plotting/plot_recipes.jl")
end
