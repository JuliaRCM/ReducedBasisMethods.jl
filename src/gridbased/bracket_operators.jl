@inline function inc(j,n)
       j + 1 > n ? j - 1 : j + 1
end

@inline function dec(j,n)
       j - 1 < 1 ? j + 1 : j - 1
end

### 2D Poisson Bracket in operator form: f ↦ [f,h]
# ci and li are cartesian / linear indices for f. h₁ and h₂ are grid size in x and v
function _apply_P_h!(Pf::AbstractVector, f::AbstractVector, h::AbstractVector, ci, li, h₁, h₂) 
    nx, ny = size(ci)
    length(Pf) == length(f) == length(h) || throw(DimensionMismatch())
    @inbounds for ij in li
        i,j = Tuple(ci[ij])
        i₋ = mod1(i-1,nx); j₋ = mod1(j-1,ny); 
        i₊ = mod1(i+1,nx); j₊ = mod1(j+1,ny)
        jpp =  ( (f[ li[i₊,j] ] - f[ li[i₋,j] ]) * ( h[ li[i,j₊] ] - h[ li[i,j₋] ] )
               - (f[ li[i,j₊] ] - f[ li[i,j₋] ]) * ( h[ li[i₊,j] ] - h[ li[i₋,j] ] )
               )
        jpc =  ( f[ li[i₊,j] ] * ( h[ li[i₊,j₊] ] - h[ li[i₊,j₋] ] )
               - f[ li[i₋,j] ] * ( h[ li[i₋,j₊] ] - h[ li[i₋,j₋] ] )
               - f[ li[i,j₊] ] * ( h[ li[i₊,j₊] ] - h[ li[i₋,j₊] ] )
               + f[ li[i,j₋] ] * ( h[ li[i₊,j₋] ] - h[ li[i₋,j₋] ] )
               )
        jcp =  ( f[ li[i₊,j₊] ] * ( h[ li[i,j₊] ] - h[ li[i₊,j] ] )
               - f[ li[i₋,j₋] ] * ( h[ li[i₋,j] ] - h[ li[i,j₋] ] )
               - f[ li[i₋,j₊] ] * ( h[ li[i,j₊] ] - h[ li[i₋,j] ] )
               + f[ li[i₊,j₋] ] * ( h[ li[i₊,j] ] - h[ li[i,j₋] ] )
               ) 
        Pf[ij] = 1/(12 * h₁ * h₂) * ( jpp + jpc + jcp )
    end
end

### 2D Poisson Bracket in operator form: f ↦ [f,v²/2 + ϕ]
# ci and li are cartesian / linear indices for f. h₁ and h₂ are grid size in x and v. ϕ is defined on x.
function _apply_P_ϕ!(Pf::AbstractVector, f::AbstractVector, v::AbstractVector, ϕ::AbstractVector, ci, li, h₁, h₂) 
       nx, nv = size(ci)
       length(Pf) == length(f) == nx*nv || throw(DimensionMismatch())
       length(v) == nv || throw(DimensionMismatch())
       length(ϕ) == nx || throw(DimensionMismatch())
       @inbounds for ij in li
           i,j = Tuple(ci[ij])
           i₋ = mod1(i-1,nx); j₋ = mod1(j-1,nv); 
           i₊ = mod1(i+1,nx); j₊ = mod1(j+1,nv)
           jpp =  ( (f[ li[i₊,j] ] - f[ li[i₋,j] ]) * 0.5 * ( v[j₊]^2 - v[j₋]^2 )
                  - (f[ li[i,j₊] ] - f[ li[i,j₋] ]) * ( ϕ[i₊] - ϕ[i₋] )
                  )
           jpc =  ( f[ li[i₊,j] ] * 0.5 * ( v[j₊]^2 - v[j₋]^2 )
                  - f[ li[i₋,j] ] * 0.5 * ( v[j₊]^2 - v[j₋]^2 )
                  - f[ li[i,j₊] ] * ( ϕ[i₊] - ϕ[i₋] )
                  + f[ li[i,j₋] ] * ( ϕ[i₊] - ϕ[i₋] )
                  )
           jcp =  ( f[ li[i₊,j₊] ] * ( ϕ[i] + 0.5 * v[j₊]^2 - ϕ[i₊] - 0.5 * v[j]^2 )
                  - f[ li[i₋,j₋] ] * ( ϕ[i₋] + 0.5 * v[j]^2 - ϕ[i] - 0.5 * v[j₋]^2 )
                  - f[ li[i₋,j₊] ] * ( ϕ[i] + 0.5 * v[j₊]^2 - ϕ[i₋] - 0.5 * v[j]^2 )
                  + f[ li[i₊,j₋] ] * ( ϕ[i₊] + 0.5 * v[j]^2 - ϕ[i] - 0.5 * v[j₋]^2 )
                  ) 
           Pf[ij] = 1/(12 * h₁ * h₂) * ( jpp + jpc + jcp )
       end
   end
   
### Collision Operator
function _apply_C!(Cf::AbstractVector, f::AbstractVector, v::AbstractVector, u::AbstractVector, ε::AbstractVector, ci, li, h₁, h₂) 
       nx, nv = size(ci)
       length(Cf) == length(f) == nx*nv || throw(DimensionMismatch())
       length(v) == nv || throw(DimensionMismatch())
       length(u) == length(ε) == nx || throw(DimensionMismatch())
       @inbounds for ij in li
              i,j = Tuple(ci[ij])
              i₋ = mod1(i-1,nx); j₋ = mod1(j-1,nv); 
              i₊ = mod1(i+1,nx); j₊ = mod1(j+1,nv)
       
              Cf[ij] = (ε[i] - u[i]^2) * (f[ li[i,j₋] ] - 2f[ij] + f[ li[i,j₊] ]) / h₂^2
              Cf[ij] += ( (v[j₊] - u[i]) * f[ li[i,j₊] ] - (v[j₋] - u[i]) * f[ li[i,j₋] ] ) / (2h₂)
       end
   end


function _apply_Cρ!(Cf::AbstractVector, f::AbstractVector, v::AbstractVector, ρu::AbstractVector, ρε::AbstractVector, ρ::AbstractVector, ci, li, h₁, h₂) 
       nx, nv = size(ci)
       length(Cf) == length(f) == nx*nv || throw(DimensionMismatch())
       length(v) == nv || throw(DimensionMismatch())
       length(ρu) == length(ρε) == length(ρ) == nx || throw(DimensionMismatch())
       @inbounds for ij in li
              i,j = Tuple(ci[ij])
              i₋ = mod1(i-1,nx); j₋ = mod1(j-1,nv); 
              i₊ = mod1(i+1,nx); j₊ = mod1(j+1,nv)
       
              Cf[ij] = (ρε[i] - ρu[i]^2 / ρ[i] ) * (f[ li[i,j₋] ] - 2f[ij] + f[ li[i,j₊] ]) / h₂^2
              Cf[ij] += ( (ρ[i] * v[j₊] - ρu[i]) * f[ li[i,j₊] ] - (ρ[i] * v[j₋] - ρu[i]) * f[ li[i,j₋] ] ) / (2h₂)
       end
end

function _apply_Cρ²!(Cf::AbstractVector, f::AbstractVector, v::AbstractVector, ρu::AbstractVector, ρε::AbstractVector, ρ::AbstractVector, ci, li, h₁, h₂) 
       nx, nv = size(ci)
       length(Cf) == length(f) == nx*nv || throw(DimensionMismatch())
       length(v) == nv || throw(DimensionMismatch())
       length(ρu) == length(ρε) == length(ρ) == nx || throw(DimensionMismatch())
       @inbounds for ij in li
              i,j = Tuple(ci[ij])
              i₋ = mod1(i-1,nx); j₋ = mod1(j-1,nv); 
              i₊ = mod1(i+1,nx); j₊ = mod1(j+1,nv)
       
              Cf[ij] = (ρ[i] * ρε[i] - ρu[i]^2) * (f[ li[i,j₋] ] - 2f[ij] + f[ li[i,j₊] ]) / h₂^2
              Cf[ij] += ρ[i] * ( (ρ[i] * v[j₊] - ρu[i]) * f[ li[i,j₊] ] - (ρ[i] * v[j₋] - ρu[i]) * f[ li[i,j₋] ] ) / (2h₂)
       end
end

function _apply_Δᵥ!(Cf::AbstractVector, f::AbstractVector, ci, li, h₁, h₂) 
       nx, nv = size(ci)
       length(Cf) == length(f) == nx*nv || throw(DimensionMismatch())
       @inbounds for ij in li
              i,j = Tuple(ci[ij])
              i₋ = mod1(i-1,nx); j₋ = mod1(j-1,nv); 
              i₊ = mod1(i+1,nx); j₊ = mod1(j+1,nv)
              Cf[ij] = (f[ li[i,j₋] ] - 2f[ij] + f[ li[i,j₊] ]) / h₂^2
       end
end