module ReducedBasisMethods

using HDF5
using HDF5: H5DataStore
using LinearAlgebra
using MultiIndexArrays: GridTensor, stencil_indices
using ReducedComplexityModeling
using GeometricEquations: ODEProblem, PODEProblem, NullParameters
using ReducedComplexityModeling: read_parameters, save_parameters, _create_group
import ReducedComplexityModeling: h5save, h5load, project

include("utils.jl")

include("reducedbasis.jl")

include("algorithms/svd.jl")

include("algorithms/evd.jl")

include("projection.jl")

export ReducedBasis, PODSVD, PODEVD, CotangentLiftSVD, CotangentLiftEVD, ComplexSVD

export lift, projection_error, reduction_error

include("algorithms/deim.jl")

export get_DEIM_interpolation_matrix, get_DEIM_indices

include("algorithms/symplectic_deim.jl")

export SymplecticDEIM, hamiltonian_gradient!

include("reduced_equations.jl")

export reduced_ode, reduced_pode

include("diagnostics.jl")

export spectrumplot

export h5save, h5load

include("reduced_tensor.jl")

export ReducedTensor

end
