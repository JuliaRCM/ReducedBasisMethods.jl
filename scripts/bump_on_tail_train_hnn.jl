
using Distances
using HDF5
using LaTeXStrings
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Plots
using ProgressMeter
using Random
using ReducedBasisMethods
using Zygote

using GeometricMachineLearning: get_batch


# HDF5 file containing training data
runid = "BoT_Np5e4_k_010_050_np_10_T25"
ppath = "../runs/$(runid)_projections.h5"
vpath = "../runs/$(runid)_vectorfield.h5"
hpath = "../runs/$(runid)_hnn.png"


# define some custom apply methods for Chain and Dense
# that use Tuples for parameters instead of NamedTuples
# and do not return a state but only the results of each
# layer and the whole chain
# splitting of Lux's return tuple of (result, state) as well
# as symbolic indexing of NamedTuples does not work when
# computing two derivatives with Zygote

@generated function Lux.applychain(layers::NamedTuple{fields}, x, ps::Tuple, st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    calls = [:(($(x_symbols[i + 1])) = Lux.apply(layers.$(fields[i]),
                                                $(x_symbols[i]),
                                                ps[$i],
                                                st.$(fields[i]))) for i in 1:N]
    push!(calls, :(return $(x_symbols[N + 1])))
    return Expr(:block, calls...)
end

@inline function Lux.apply(d::Dense{false}, x::AbstractVecOrMat, ps::Tuple, st::NamedTuple)
    return d.activation.(ps[1] * x)
end

@inline function Lux.apply(d::Dense{true}, x::AbstractVector, ps::Tuple, st::NamedTuple)
    return d.activation.(ps[1] * x .+ vec(ps[2]))
end


# Lux initialisation methods for Float64

function glorot_uniform64(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float64(gain) * sqrt(24.0 / sum(Lux._nfan(dims...)))
    return (rand(rng, Float64, dims...) .- 0.5) .* scale
end

zeros64(rng::AbstractRNG, args...; kwargs...) = zeros(rng, Float64, args...; kwargs...)


# Plot functions for diagnostics

function plot_network(H, total_loss; xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2, nsamples=100, filename=nothing)
    # #get offset of learned Hamiltonian
    # H̃₀ = H̃([0,0])

    # #plot contour lines of Hamiltonian
    # X = range(xmin, stop=xmax, length=nsamples)
    # Y = range(ymin, stop=ymax, length=nsamples)
    # plt_cnt = contour(X, Y, [H([x,y]) for x in X, y in Y], linewidth = 0, fill = true, levels = 7, c = cgrad(:default, rev = true))

    # #plot contour lines of learned Hamiltonian
    # contour!(plt_cnt, X, Y, [H̃([x,y]) - H̃₀ for x in X, y in Y], linecolor = :black)

    # #plot contours of error of Hamiltonian
    # plt_err = contourf(X, Y, [H̃([x,y]) - H̃₀ - H([x,y]) for x in X, y in Y])

    # plot total loss
    plt_loss = plot(total_loss, xguide="n(training)", yguide="Total Loss", legend=false, size=(1000,800))

    # l = @layout [
    #         grid(1,2)
    #         b{0.4h}
    # ]

    # plt = plot(plt_cnt, plt_err, plt_loss, layout = l)

    if filename !== nothing
        savefig(filename)
    end

    return plt_loss
end


# read reduced basis
h5open(ppath, "r") do file
    global rbasis = ReducedBasis(file)
end


# read E field training data

h5open(vpath, "r") do file
    global X = read(file["X"])
    global A = read(file["A"])
end

data   = [copy(X[:,i,j]) for (i,j) in Iterators.product(axes(X,2), axes(X,3))]
target = [copy(A[:,i,j]) for (i,j) in Iterators.product(axes(A,2), axes(A,3))]

@assert length(data) == length(target)
@assert length(data[begin]) == length(target[begin])


# Input normalization

function minmax_normalisation(x, xmin, xmax, u = -1.0, l = +1.0)
    (x .- xmin) ./ (xmax - xmin) .* (u-l) .+ l
end

# Xmin = minimum(X)
# Xmax = maximum(X)

# datanorm = [minmax_normalisation(d, Xmin, Xmax) for d in data]


function eigenvalue_normalisation(x, λ)
    x ./ λ
end

Λ = sqrt.(rbasis.Λₚ[1:length(data[begin])])

datanorm = [eigenvalue_normalisation(d, Λ) for d in data]


# set random generator seed
Random.seed!(42)

# learning rate
const η = .0001

# number of training runs
const nruns = 1000

# input dimension
const nin = length(data[begin])

# layer width
const ld = nin

# activation function
# const act = NNlib.relu
const act = tanh

# create model
model = Chain(Dense(nin, ld, act; init_weight=glorot_uniform64, init_bias=zeros64),
              Dense(ld,  ld, act; init_weight=glorot_uniform64, init_bias=zeros64),
              Dense(ld,  1; init_weight=glorot_uniform64, init_bias=zeros64, bias=false))

# model = Chain(Dense(nin, ld, act),
#               Dense(ld,  ld, act),
#               Dense(ld,  1; bias=false))

# model = Chain(Dense(nin, ld, act; init_weight=glorot_uniform64, init_bias=zeros64),
#               Dense(ld, div(ld,2), act; init_weight=glorot_uniform64, init_bias=zeros64),
#               Dense(div(ld,2), div(ld,4), act; init_weight=glorot_uniform64, init_bias=zeros64),
#               Dense(div(ld,4),  1; init_weight=glorot_uniform64, init_bias=zeros64, bias=false))

# look at this in more detail; sets bias to zero for example
ps, st = Lux.setup(Random.default_rng(), model)

# define Hamiltonian via evaluation of network
function hnn(model, x, params::Tuple, state)
    y = Lux.apply(model, x, params, state)
    return sum(y)
end

function hnn(model, x, params::NamedTuple, state)
    y, st = Lux.apply(model, x, params, state)
    return sum(y)
end

# compute vector fields associated with network
grad_ϕ(model, x, params, state) = Zygote.gradient(ξ -> hnn(model, ξ, params, state), x)[1]

# loss for a single datum
loss_sing(model, x, y, params, state) = sqeuclidean(grad_ϕ(model, x, params, state), y)

# total loss
hnn_loss(model, x, y, params, state) = mapreduce(i -> loss_sing(model, x[i], y[i], params, state), +, eachindex(x,y))

# loss gradient
hnn_loss_gradient(model, x, y, params, state) = Zygote.gradient(p -> hnn_loss(model, x, y, p, state), params)[1]


function train_lux_hnn(model, params, state, data, target, runs, η)
    # create array to store total loss
    total_loss = zeros(runs)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in params])

    # do a couple learning runs
    @showprogress 1 "Training..." for j in 1:runs
        batch_data, batch_target = get_batch(data, target, 100)

        # gradient step
        params_grad = hnn_loss_gradient(model, batch_data, batch_target, params_tuple, state)

        # make gradient steps for all the model parameters W & b
        for i in eachindex(params_tuple, params_grad)
            for (p, dp) in zip(params_tuple[i], params_grad[i])
                p .-= η .* dp
            end
        end

        # total loss i.e. loss computed over all data
        total_loss[j] = hnn_loss(model, data, target, params, state)
        # println("Total loss after run $j: $(total_loss[j])")
    end

    return (model, data, target, params, state, total_loss)
end


println()
println("Test output of model on initial parameters")
println()

batch_data, batch_target = get_batch(datanorm, target)

for i in eachindex(batch_data, batch_target)
    println("Batch no. $i")
    println("input  = ", batch_data[i])
    println("target = ", batch_target[i])
    println("grad ϕ = ", grad_ϕ(model, batch_data[i], ps, st))
    println("ϕ(ξ)   = ", hnn(model, batch_data[i], ps, st))
    println("ϕ(ξ+Δ) = ", hnn(model, batch_data[i] .* (1 + 1E-3), ps, st))
    println()
end


# println()
# println("Model parameters before training:")
# println(ps)
# println()

model, data, target, params, state, total_loss = train_lux_hnn(model, ps, st, datanorm, target, nruns, η)

# println()
# println("Model parameters after training:")
# println(params)
# println()


# learned Hamiltonian & vector field
hnn_est(ξ) = hnn(model, ξ, params, state)
dhnn_est(ξ) = hnn_vf(model, ξ, params, state)

# plot results
plot_network(hnn_est, total_loss; filename=hpath)
