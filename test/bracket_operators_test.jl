@testset "Bracket Tests - ϕ and h formulation" begin

    n₁ = 64
    n₂ = 64

    ci = CartesianIndices(zeros(n₁,n₂));
    li = LinearIndices(ci);

    x = range(0,1,length=n₁)
    v = range(-1,1,length=n₂)
    h₁ = x[2] - x[1]
    h₂ = v[2] - v[1]

    f = vec([ sin(_x * 2π) * exp(- 16 * _v^2) for _x in x, _v in v])
    ϕ = [ sin(_x * 2π) * cos(_x * 2π) for _x in x ]
    h = vec( [ _ϕ + 0.5*_v^2 for _ϕ in ϕ, _v in v] )

    function _closure_apply_P_h!(Pf::AbstractVector, f::AbstractVector)
        _apply_P_h!(Pf, f, h, ci, li, h₁, h₂)
    end

    function _closure_apply_P_hᵀ!(Pf::AbstractVector, f::AbstractVector)
        _apply_P_h!(Pf, f, h, ci, li, h₁, h₂)
        Pf .*= -1
    end

    P_h = LinearMap(_closure_apply_P_h!, _closure_apply_P_hᵀ!, n₁*n₂ ;ismutating=true, issymmetric=false)

    @test abs( dot(f, P_h * h) ) < 1e-15 * n₁ * n₂
    @test abs( dot(h, P_h * f) ) < 1e-15 * n₁ * n₂

    function _closure_apply_P_ϕ!(Pf::AbstractVector, f::AbstractVector)
        _apply_P_ϕ!(Pf, f, v, ϕ, ci, li, h₁, h₂)
    end
    
    function _closure_apply_P_ϕᵀ!(Pf::AbstractVector, f::AbstractVector)
        _apply_P_ϕ!(Pf, f, v, ϕ, ci, li, h₁, h₂)
        Pf .*= -1
    end

    P_ϕ = LinearMap(_closure_apply_P_ϕ!, _closure_apply_P_ϕᵀ!, n₁*n₂ ;ismutating=true, issymmetric=false)

    @test abs( dot(f, P_ϕ * h) ) < 1e-15 * n₁ * n₂
    @test abs( dot(h, P_ϕ * f) ) < 1e-15 * n₁ * n₂

    @test norm( (P_h - P_ϕ) * f ) < 1e-15 * n₁ * n₂
end

@testset "Bracket Tests - analytical solution" begin

    n₁ = 512
    n₂ = 512
    n = n₁*n₂

    h₁ = 1/n₁
    h₂ = 2/n₂

    x = range(0,1-h₁,length=n₁)
    v = range(-1,1-h₂,length=n₂)

    ci = CartesianIndices(zeros(n₁,n₂));
    li = LinearIndices(ci);

    f = vec([ 1 / (2π) * cos(_x * 2π) * ( cos(_v * 2π) - 1) for _x in x, _v in v])
    h = vec([ sin(_x * 2π) * exp(- 16 * _v^2) for _x in x, _v in v])

    fh = vec( [ 32 * _v * sin(_x * 2π )^2 * ( cos(_v * 2π) - 1) * exp(- 16 * _v^2) + 2π * cos(_x * 2π)^2 * sin(_v * 2π) * exp(- 16 * _v^2) for _x in x, _v in v])


    function _closure_apply_P_h!(Pf::AbstractVector, f::AbstractVector)
        _apply_P_h!(Pf, f, h, ci, li, h₁, h₂)
    end
            
    function _closure_apply_P_hᵀ!(Pf::AbstractVector, f::AbstractVector)
        _apply_P_h!(Pf, f, h, ci, li, h₁, h₂)
        Pf .*= -1
    end
            
    P_h = LinearMap(_closure_apply_P_h!, _closure_apply_P_hᵀ!, n₁*n₂ ;ismutating=true, issymmetric=false)

    @test norm(P_h * f - fh) / norm(fh) < 5e-4

end