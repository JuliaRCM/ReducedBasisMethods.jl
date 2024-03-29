### 1D Laplace operator with periodic bc
function _apply_Δₓ!(y::AbstractVector, x::AbstractVector, h₁) 
    nx = length(x)
    length(x) == length(y) || throw(DimensionMismatch())

    @inbounds for i in 1:nx
        i₋ = mod1(i-1,nx); i₊ = mod1(i+1,nx)
        y[i] = 1 / (h₁^2) * ( x[i₋] - 2*x[i] + x[i₊] )
    end
end

function _apply_Δₓ₄!(y::AbstractVector, x::AbstractVector, h₁) # 1D Laplace operator (4th order) with periodic bc
    nx = length(x)
    length(x) == length(y) || throw(DimensionMismatch())
    @inbounds for i in 1:nx
        i₋ = mod1(i-1,nx); i₊ = mod1(i+1,nx); i₋₋ = mod1(i-2,nx); i₊₊ = mod1(i+2,nx)
        y[i] = - 1 / (12 * h₁^2) * ( 5*x[i₊₊] - 32*x[i₊] + 54*x[i] - 32*x[i₋] + 5*x[i₋₋] )
    end
end


# L = -Δ  + R    
function _apply_Lₓ₄!(y::AbstractVector, x::AbstractVector, h₁) # 1D Laplace operator (4th order) with periodic bc
    nx = length(x)
    length(x) == length(y) || throw(DimensionMismatch())
    Σx = sum(x) / nx
    @inbounds for i in 1:nx
        i₋ = mod1(i-1,nx); i₊ = mod1(i+1,nx); i₋₋ = mod1(i-2,nx); i₊₊ = mod1(i+2,nx)
        y[i] = 1 / (12 * h₁^2) * ( 5*x[i₊₊] - 32*x[i₊] + 54*x[i] - 32*x[i₋] + 5*x[i₋₋] )
        y[i] += Σx
    end
end

### Constant Nullspace Projection
# if 1 ∈ Δ, then Δϕ = ρ is not well posed but (Δ + R)ϕ = (1 - R)ρ is. 
function _apply_Rₓ!(y::AbstractVector, x::AbstractVector) # Nullspace projection
    nx = length(x)
    length(x) == length(y) || throw(DimensionMismatch())
    Σx = sum(x) / nx
    @inbounds for i in 1:nx
        y[i] = Σx
    end
end

### integrate over velocity space
function _apply_∫dv!(y::AbstractVector, f::AbstractVector, ci, li, h₁, h₂)
    nx, nv = size(ci)
    (length(f) == nx*nv && length(y) == nx ) || throw(DimensionMismatch())
    y .= 0
    @inbounds for ij in li
        i,j = Tuple(ci[ij])
        # (∫ f dv)(xᵢ) = ∑ⱼ f(xᵢ,vⱼ) hᵥ
        y[i] += f[ li[i,j] ] * h₂
    end
end

function _apply_∫dvᵀ!(y::AbstractVector, ψ::AbstractVector, ci, li, h₁, h₂)
    nx, nv = size(ci)
    (length(ψ) == nx && length(y) == nx*nv ) || throw(DimensionMismatch())
    y .= 0
    @inbounds for ij in li
        i,j = Tuple(ci[ij])
        # (∫ f dv)(xᵢ) = ∑ⱼ f(xᵢ,vⱼ) hᵥ
        y[ij] = ψ[i] * h₂
    end
end

### weighted integral over velocity space
function _apply_∫dv_μ!(y::AbstractVector, f::AbstractVector, μ, ci, li, h₁, h₂)
    nx, nv = size(ci)
    (length(f) == nx*nv && length(y) == nx ) || throw(DimensionMismatch())
    y .= 0
    @inbounds for ij in li
        i,j = Tuple(ci[ij])
        # (∫ f dv)(xᵢ) = ∑ⱼ f(xᵢ,vⱼ) hᵥ
        y[i] += f[ li[i,j] ] * h₂ * μ[j]
    end
end

function _apply_∫dvᵀ_μ!(y::AbstractVector, ψ::AbstractVector, μ, ci, li, h₁, h₂)
    nx, nv = size(ci)
    (length(ψ) == nx && length(y) == nx*nv ) || throw(DimensionMismatch())
    y .= 0
    @inbounds for ij in li
        i,j = Tuple(ci[ij])
        # (∫ f dv)(xᵢ) = ∑ⱼ f(xᵢ,vⱼ) hᵥ
        y[ij] = ψ[i] * h₂ * μ[j]
    end
end

### First moment over velocity space
function _apply_∫vdv!(y::AbstractVector, f::AbstractVector, v::AbstractVector, ci, li, h₁, h₂)
    nx, nv = size(ci)
    (length(f) == nx*nv && length(y) == nx && length(v) == nv ) || throw(DimensionMismatch())
    y .= 0
    @inbounds for ij in li
        i,j = Tuple(ci[ij])
        # (∫ f dv)(xᵢ) = ∑ⱼ f(xᵢ,vⱼ) hᵥ
        y[i] += f[ li[i,j] ] * h₂ * v[j]
    end
end

### Second moment over velocity space
function _apply_∫v²dv!(y::AbstractVector, f::AbstractVector, v::AbstractVector, ci, li, h₁, h₂)
    nx, nv = size(ci)
    (length(f) == nx*nv && length(y) == nx && length(v) == nv ) || throw(DimensionMismatch())
    y .= 0
    @inbounds for ij in li
        i,j = Tuple(ci[ij])
        # (∫ f dv)(xᵢ) = ∑ⱼ f(xᵢ,vⱼ) hᵥ
        y[i] += f[ li[i,j] ] * h₂ * v[j]^2
    end
end