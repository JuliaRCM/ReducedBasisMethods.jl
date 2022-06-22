

const nx = 10
const nv = 24

const T = Float64

_parent = rand(nx, nv)

f = (i,j) -> _parent[i,j]

MultiIndexLazyArray(T, f, (MultiIndexAxis(1:nx), MultiIndexAxis(1:nv)))
MultiIndexLazyArray(T, f, MultiIndexAxis(1:nx), MultiIndexAxis(1:nv))
MultiIndexLazyArray(T, f, (1:nx,), (1:nv,))
MultiIndexLazyArray(T, f, (nx,), (nv,))

@test MultiIndexLazyArray(T, f, (MultiIndexAxis(1:nx), MultiIndexAxis(1:nv))) ==
      MultiIndexLazyArray(T, f, MultiIndexAxis(1:nx), MultiIndexAxis(1:nv)) ==
      MultiIndexLazyArray(T, f, (1:nx,), (1:nv,)) ==
      MultiIndexLazyArray(T, f, (nx,), (nv,))

mila = MultiIndexLazyArray(T, f, (nx,), (nv,))

@test mila[1,1] == _parent[1,1]
@test mila[1,nv] == _parent[1,nv]
@test mila[nx,1] == _parent[nx,1]
@test mila[nx,nv] == _parent[nx,nv]


# __parent = vec(_parent)

# g = i -> __parent[i]

# MultiIndexLazyArray(T, g, (MultiIndexAxis(1:nx, 1:nv)))
# MultiIndexLazyArray(T, g, MultiIndexAxis(1:nx, 1:nv))
# MultiIndexLazyArray(T, g, (1:nx, 1:nv))
# MultiIndexLazyArray(T, g, (nx, nv))

# @test MultiIndexLazyArray(T, g, (MultiIndexAxis(1:nx, 1:nv))) ==
#       MultiIndexLazyArray(T, g, MultiIndexAxis(1:nx, 1:nv)) ==
#       MultiIndexLazyArray(T, g, (1:nx, 1:nv)) ==
#       MultiIndexLazyArray(T, g, (nx, nv))

# mila = MultiIndexLazyArray(T, g, MultiIndexAxis(1:nx, 1:nv))

# @test mila[(1,1)] == _parent[1,1]
# @test mila[(1,nv)] == _parent[1,nv]
# @test mila[(nx,1)] == _parent[nx,1]
# @test mila[(nx,nv)] == _parent[nx,nv]
