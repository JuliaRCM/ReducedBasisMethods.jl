using IterativeSolvers, LinearMaps

@testset "Poisson1D" begin

    n₁ = 256
    n₂ = 256
    h₁ = 1/n₁
    h₂ = 1/n₂
    x = range(0,1 - h₁,length=n₁)
    v = range(0,1 - h₂,length=n₂)

    ci = CartesianIndices(zeros(n₁,n₂))
    li = LinearIndices(ci)

    f = vec( [ sin(_x * 2π ) * π / 2 * sin(_v * π) for _x in x, _v in v ] )
    ρ = [sin(_x * 2π ) for _x in x];

    function _closure_apply_∫dv!(y::AbstractVector, x::AbstractVector)
        _apply_∫dv!(y, x, ci, li, h₁, h₂)
    end
    ∫dv = LinearMap(_closure_apply_∫dv!, n₁, n₁*n₂; ismutating=true)
    
    @test norm(∫dv * f - ρ) / norm(ρ) < 5e-5
    
    Δ⁻¹ρ_an = [ -( 2π )^(-2) * sin(_x * 2π ) for _x in x ];

    function _closure_apply_Δₓ!(y::AbstractVector, x::AbstractVector)
        _apply_Δₓ!(y, x, h₁)
    end

    function _closure_apply_Δₓ₄!(y::AbstractVector, x::AbstractVector)
        _apply_Δₓ₄!(y, x, h₁)
    end

    Δₓ = LinearMap(_closure_apply_Δₓ!, _closure_apply_Δₓ!, n₁; ismutating=true, issymmetric=true)
    Δₓ₄ = LinearMap(_closure_apply_Δₓ₄!, _closure_apply_Δₓ₄!, n₁; ismutating=true, issymmetric=true)
    Rₓ = LinearMap(_apply_Rₓ!, _apply_Rₓ!, n₁; ismutating=true, issymmetric=true)
    Lₓ = Δₓ + Rₓ
    Lₓ₄ = Δₓ₄ + Rₓ

    Δ⁻¹ρ = zeros(n₁)
    ρ_perp = ρ - Rₓ * ρ
    cg!(Δ⁻¹ρ, Lₓ, ρ_perp );
    @test norm( Δ⁻¹ρ - Δ⁻¹ρ_an ) / norm(Δ⁻¹ρ_an) < 5e-4

    cg!(Δ⁻¹ρ, Lₓ₄, ρ_perp );
    @test norm( Δ⁻¹ρ - Δ⁻¹ρ_an ) / norm(Δ⁻¹ρ_an) < 5e-4

end