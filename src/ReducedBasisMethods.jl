module ReducedBasisMethods

using HDF5
using HDF5: H5DataStore
using LinearAlgebra
using LazyArrays
using MultiIndexArrays: _stencil_indices
using ReducedComplexityModeling
using ReducedComplexityModeling: read_parameters, save_parameters
import ReducedComplexityModeling: h5save, h5load

# `ReducedTensor` wraps the grid tensor that lives in GeometricBrackets, and extends the
# `_nx`/`_nv` accessors rather than redefining them, so one generic covers both packages.
using GeometricBrackets: PoissonTensor
import GeometricBrackets: _nx, _nv

include("utils.jl")

include("regression.jl")

export get_regression_αβ

include("reducedbasis.jl")

export ReducedBasis, CotangentLiftEVD, CotangentLiftSVD

include("eigen.jl")

export sorteigen

include("algorithms/evd.jl")

export get_PODBasis_EVD, get_PODBasis_cotangentLiftEVD

include("algorithms/deim.jl")

export get_DEIM_interpolation_matrix

include("h5routines.jl")

export h5save, h5load

include("reduced_tensor.jl")

export ReducedTensor

end
