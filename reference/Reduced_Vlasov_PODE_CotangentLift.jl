using GeometricIntegrators
using HDF5

# PARAMETERS

dt     = 0.0001                    # time step
ntotal = 10000000                  # total number of time steps to compute
nsave  = 1000                      # save every nsave step
nt     = div(ntotal,nsave)      # total number of time steps to be saved
n      = 20                      # number of particles in the reduced model
beta   = 6.0


# Input and output files
path      = "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001_REFERENCE.hdf5"
path_svd  = "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_PSD_Cotangent_SVD_dt001_skip10_T1000.hdf5"
path_out  = "/toks/work/ttoma/ModelReductionQuadratic/OUTPUTS/T1000/Cotangent/SV/Vlasov_reduced_model_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt00001_PSD_Cotangent_k20_SV.hdf5"
path_out2 = "/toks/work/ttoma/ModelReductionQuadratic/OUTPUTS/T1000/Cotangent/SV/Vlasov_reduced_model_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt00001_PSD_Cotangent_k20_SV_lifted.hdf5"




# INITIAL CONDITIONS
# Reading the full model saved as a ODE file

MySol = h5open(path,"r")
z0 = read(MySol,"q")
close(MySol)

z0 = z0[:,1]


N = div( length(z0), 2 )

q0 = z0[1:N]
p0 = z0[N+1:2*N]

h5 = h5open(path_svd, "r")
Phi = read(h5, "Phi")
close(h5)
Phi = Phi[:,1:n]

q0 = transpose(Phi)*q0
p0 = transpose(Phi)*p0


# BUTCHER's TABLEAU
tab = getTableauLobattoIIIAIIIB2()


# THE ELECTRIC FIELD
function E(x::Number, param::Number)
    return param^2 * x
end



#THE DRIFT AND DIFFUSION FUNCTIONS
# Phi is orthogonal, so the form of the kinetic term stays the same
function v(t, q, p, v_out, param=beta, U=Phi)
    for i in 1:length(p)
        v_out[i] = p[i]
    end
end

function f(t, q, p, f_out, param=beta, U=Phi)

    qlift = U*q

    for i in 1:length(q)
        
        tmp = 0.0

        for k=1:length(qlift)

            tmp = tmp -E(qlift[k],param)*U[k,i]

        end

        f_out[i]= tmp
    end
end



# DEFINITION OF THE EQUATION, INTEGRATOR AND SOLUTION
MySDE = PODE(v, f, q0, p0)

integr = Integrator(MySDE, tab, dt)

MySol = Solution(MySDE, dt, ntotal, nsave=nsave)

h5 = create_hdf5(MySol, path_out)


# RUN THE SIMULATION AND SAVE TO FILE

integrate!(integr, MySol)
write_to_hdf5(MySol, h5)
close(h5)


# CALCULATING THE LIFTED SOLUTION
MySol = SSolutionPODE(path_out)

q0 = Phi*MySol.q.d[:,:]
p0 = Phi*MySol.p.d[:,:]

t = nsave*dt*collect(range(0,nt,step=1))


h5 = h5open(path_out2, "w")
write(h5, "q", q0)
write(h5, "p", p0)
write(h5, "t", t)
close(h5)
