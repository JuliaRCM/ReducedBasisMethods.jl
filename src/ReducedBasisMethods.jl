module ReducedBasisMethods

using HDF5
using HDF5: H5DataStore
using LinearAlgebra
using MultiIndexArrays: _stencil_indices
using ReducedComplexityModeling
using ReducedComplexityModeling: read_parameters, save_parameters, _create_group
import ReducedComplexityModeling: h5save, h5load, project

# `ReducedTensor` wraps the grid tensor that lives in GeometricBrackets, and extends the
# `_nx`/`_nv` accessors rather than redefining them, so one generic covers both packages.
using GeometricBrackets: PoissonTensor
import GeometricBrackets: _nx, _nv

include("utils.jl")

include("regression.jl")

export get_regression_αβ

include("reducedbasis.jl")

include("algorithms/svd.jl")

include("algorithms/evd.jl")

include("projection.jl")

export ReducedBasis, PODSVD, PODEVD, CotangentLiftSVD, CotangentLiftEVD, ComplexSVD

export lift, projection_error, reduction_error

include("algorithms/deim.jl")

export get_DEIM_interpolation_matrix

include("algorithms/symplectic_deim.jl")

export SymplecticDEIM, hamiltonian_gradient!

export h5save, h5load

include("reduced_tensor.jl")

export ReducedTensor

end
