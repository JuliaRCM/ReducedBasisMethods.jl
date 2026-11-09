# Known issues

What is known to be wrong in ReducedBasisMethods and is not fixed.

## This package

### K1 · `_ReducedVectorField.cache` has the field type `IdDict{DataType, Any}`, whose value type `Any` is abstract

- **location:** `src/reduced_equations.jl:36`, filled and read in `_work` (`src/reduced_equations.jl:48-52`)
- **evidence:** The cache holds one pair of work buffers `(x, g)` for each state element type `S`
  other than the basis's `T`, such as the dual numbers of Newton's `JacobianAutodiff`. `_work`
  creates the pair with `get!` on its first call for `S`, and the three call forms of the vector
  field read it (`src/reduced_equations.jl:170`, `178`, `186`). The lookup returns `Any`; the
  assertion `::Tuple{Vector{S}, Vector{S}}` after `get!` makes the return of `_work` concrete again,
  at the cost of an `IdDict` lookup and a type check on each call with `S !== T`.
  `Harness/scripts/concrete-fields.jl ReducedBasisMethods` reports the field as
  `ReducedBasisMethods._ReducedVectorField.cache::IdDict{DataType, Any}`, its one violation. The script
  flags a cache keyed by type by design. GeometricMachineLearning has the same pattern in
  `HRedSys.dual_cache` and `ReducedVFCache.dual_cache`. A fix sets the second element type when
  the callable is built: a type parameter `S` and a field `dual::Tuple{Vector{S}, Vector{S}}`,
  with `S` taken from the solver's Jacobian configuration, or buffers that the caller passes in.
  Either way the element type of the derivative must be known at emission, so the fix is a
  design decision.
- **kind:** defect
- **found:** 2026-11-09
