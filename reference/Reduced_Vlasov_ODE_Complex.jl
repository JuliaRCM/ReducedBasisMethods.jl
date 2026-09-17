using GeometricIntegrators
using HDF5
using LinearAlgebra


set_config(:nls_rtol_break, Inf)
set_config(:nls_stol_break, Inf)
set_config(:nls_nmax, 500)       # maximum number of nonlinear iterations
set_config(:verbosity, 0)



# PARAMETERS

dt     = 0.0001                    # time step
ntotal = 10000000                  # total number of time steps to compute
nsave  = 1000                      # save every nsave step
nt     = div(ntotal,nsave)      # total number of time steps to be saved
n      = 5                      # number of particles in the reduced model
beta   = 6.0


# Input and output files
path      = "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001_REFERENCE.hdf5"
path_svd  = "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_PSD_Complex_SVD_dt001_skip10_T1000.hdf5"
path_out  = "/toks/work/ttoma/ModelReductionQuadratic/OUTPUTS/T1000/Complex/RK4/Vlasov_reduced_model_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt00001_PSD_Complex_k5_RK4.hdf5"
path_out2 = "/toks/work/ttoma/ModelReductionQuadratic/OUTPUTS/T1000/Complex/RK4/Vlasov_reduced_model_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt00001_PSD_Complex_k5_RK4_lifted.hdf5"



# INITIAL CONDITIONS
# Reading the full model saved as a ODE file

MySol = h5open(path,"r")
z0 = read(MySol,"q")
close(MySol)

z0 = z0[:,1]

N = div( length(z0), 2 )


h5 = h5open(path_svd, "r")
Phi = read(h5, "Phi")
Psi = read(h5, "Psi")
close(h5)
Phi = Phi[:,1:n]
Psi = Psi[:,1:n]

A = [ Phi -Psi; Psi Phi ]

JN = [zeros(N,N) Matrix{Float64}(I, N, N); -Matrix{Float64}(I, N, N) zeros(N,N)]
Jn = [zeros(n,n) Matrix{Float64}(I, n, n); -Matrix{Float64}(I, n, n) zeros(n,n)]

Ainv = transpose(Jn)*transpose(A)*JN

z0 = Ainv*z0



# BUTCHER's TABLEAU
#tab = getTableauLobattoIIIAIIIB2()
#tab = getTableauGLRK(1)
#tab = getTableauExplicitMidpoint()
tab = getTableauERK4()
#tab = getTableauHeun()


# THE ELECTRIC FIELD
function E(x::Number, param::Number)
    return param^2 * x
end


#THE DRIFT AND DIFFUSION FUNCTIONS
function v(t, q, v_out, param=beta, U=A)

    z_lift = U*q

    N = div( length(z_lift), 2 )
    n = div( length(q), 2 )

    q_lift = z_lift[1:N]
    p_lift = z_lift[N+1:2*N]

    for i in 1:n

        tmp = 0.0

        for k=1:length(q_lift)
            tmp = tmp + E(q_lift[k],param)*U[k,i+n] + p_lift[k]*U[k+N,i+n]
        end

        v_out[i] = tmp
    end


    for i in 1:n

        tmp = 0.0

        for k=1:length(q_lift)

            tmp = tmp -E(q_lift[k],param)*U[k,i] - p_lift[k]*U[k+N,i]

        end

        v_out[i+n]= tmp
    end
end




# DEFINITION OF THE EQUATION, INTEGRATOR AND SOLUTION
MySDE = ODE(v, z0)

integr = Integrator(MySDE, tab, dt)

MySol = Solution(MySDE, dt, ntotal, nsave=nsave)

h5 = create_hdf5(MySol, path_out)


# RUN THE SIMULATION AND SAVE TO FILE

integrate!(integr, MySol)
write_to_hdf5(MySol, h5)
reset!(MySol)

close(h5)


# CALCULATING THE LIFTED SOLUTION
MySol = SSolutionODE(path_out)

z0 = A*MySol.q.d[:,:]
q0 = z0[1:N,:]
p0 = z0[N+1:2*N,:]

t = nsave*dt*collect(range(0,nt,step=1))

h5 = h5open(path_out2, "w")
write(h5, "q", q0)
write(h5, "p", p0)
write(h5, "t", t)
close(h5)
