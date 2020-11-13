function massmatrix_PBSpline(S::PBSpline{T}, q = S.p+1) where {T}
    M = zeros(T, S.nₕ, S.nₕ)
    for i in 1:S.nₕ
        for k in 0:S.p
            M[1,i] += integrate_gausslegendre(x -> (eval_PBSpline.(S, 1, x) .* eval_PBSpline.(S, i, x)),
                                                k*S.h, (k+1)*S.h, q)
        end
    end
    for j in 2:S.nₕ
        for i in 1:S.nₕ
            M[j,i] = M[j-1,(i-1+S.nₕ-1)%S.nₕ + 1]
        end
    end
    return M
end

function stiffnessmatrix_PBSpline(S::PBSpline{T}, q = S.p) where {T}
    K = zeros(T, S.nₕ, S.nₕ)
    for i in 1:S.nₕ
        for k in 0:S.p
            K[1,i] += integrate_gausslegendre(x -> (eval_deriv_PBSpline.(S, 1, x) .* eval_deriv_PBSpline.(S, i, x) ),
                                                k*S.h, (k+1)*S.h, q)
        end
    end
    for j in 2:S.nₕ
        for i in 1:S.nₕ
            K[j,i] = K[j-1,(i-1+S.nₕ-1)%S.nₕ + 1]
        end
    end
    return K
end

"""
Solves the Poisson equation for periodic boundary conditions

input: Spline struct S, stiffness matrix K, right hand side vector c
output: b such that Kb = c

sum(c) < ε needs to hold for solubility
"""
function solve_poisson_PBSpline(S::PBSpline{T}, K::Matrix{T}, c::Vector{T}, K_aug = zero(K), c_aug = zero(c)) where {T}
    @assert abs(sum(c)) < S.nₕ*eps(T)
    K_aug .= K
    c_aug .= c
    K_aug[S.nₕ,:] .= ones(T, S.nₕ)
    c_aug[S.nₕ] = 0
    return K_aug\c_aug
end
