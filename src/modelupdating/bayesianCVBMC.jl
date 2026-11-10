struct CyclicVariationalBayesianMonteCarlo <:AbstractBayesianMethod
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    optimize::Optim.AbstractOptimizer
    optimi_options::Optim.Options
    islog::Bool
end

function bayesianupdating(
    prior::Function,
    likelihood::Function,
    models::Vector{<:UQModel},
    cvbmc::CyclicVariationalBayesianMonteCarlo
    )
end

### Helper functions
function temp(
    t::Float64,
    Ntot::Int,
    Ncyc::Int;
    S::Real=0.5,
    M::Int=5, 
    α::Real=0.2
)
    T = floor(α*Ntot/Ncyc)
    τ = mod(t-1, ceil(T/M))

    β_t = 0
    if τ<=S
        β_t = τ/S
    else 
        β_t = 1
    end

    return 1/β_t
end