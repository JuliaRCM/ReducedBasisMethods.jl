function plot_f(x::Array{Float64}, v::Array{Float64}, f)
    F = zeros(Float64, length(x), length(v))

    for i in eachindex(x)
        for j in eachindex(v)
            F[i,j] = f(x[i], v[j])
        end
    end
    p = contour(x, v, F', fill = true)
    display(p)
end

function plot_particles(x::Array{Float64}, v::Array{Float64}, P::Particles, L::Float64)

    for i in eachindex(P.x)
        while P.x[i] < 0
            P.x[i] += L
        end
        while P.x[i] > L
            P.x[i] -= L
        end
    end

    F = zeros(Float64, length(x), length(v))
    x_max = maximum(x)
    x_min = minimum(x)
    v_max = maximum(v)
    v_min = minimum(v)

    for k in eachindex(P.x)
        if x_min < P.x[k] < x_max && v_min < P.v[k] < v_max
            i = Int(floor((P.x[k]-x_min)/(x_max - x_min)*length(x)))
            j = Int(floor((P.v[k]-v_min)/(v_max - v_min)*length(v)))
            F[i+1,j+1] += P.w[k]
        end
    end
    p = PyPlot.pcolormesh(x, v, F', vmin=0.0, vmax=10.0*P.w[1], edgecolors="none")
    PyPlot.xlabel(L"$x$")
    PyPlot.ylabel(L"$v$")
    PyPlot.grid("false", which="both")
    #display(heatmap(p))
    return p
end
