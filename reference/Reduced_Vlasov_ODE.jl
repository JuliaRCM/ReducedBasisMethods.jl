using GeometricIntegrators
using HDF5

set_config(:nls_rtol_break, Inf)
set_config(:nls_stol_break, Inf)
set_config(:nls_atol_break, Inf)
set_config(:nls_nmax, 500)       # maximum number of nonlinear iterations
set_config(:verbosity, 0)



# PARAMETERS

dt     = 0.0001                    # time step
ntotal = 10000000                    # total number of time steps to compute
nsave  = 1000                      # save every nsave step
nt     = div(ntotal,nsave)      # total number of time steps to be saved
n      = 5                      # number of particles in the reduced model
beta   = 6.0


# Input and output files
path      = "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001_REFERENCE.hdf5"
path_svd  = "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_POD_SVD_dt001_skip10_T1000.hdf5"
path_out  = "/toks/work/ttoma/ModelReductionQuadratic/OUTPUTS/T1000/POD/RK4/Vlasov_reduced_model_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt00001_POD_k5_RK4.hdf5"
path_out2 = "/toks/work/ttoma/ModelReductionQuadratic/OUTPUTS/T1000/POD/RK4/Vlasov_reduced_model_beta600_sigmax10_a03_sigmav05_v4_T1000_n1000_dt00001_POD_k5_RK4_lifted.hdf5"



# INITIAL CONDITIONS
# Reading the full model saved as a ODE file
MySol = h5open(path,"r")
q0 = read(MySol,"q")
close(MySol)

q0 = q0[:,1]

N = div( length(q0), 2 )

h5 = h5open(path_svd, "r")
Phi = read(h5, "Phi")
close(h5)
Phi = Phi[:,1:n]


# Projecting the initial condition
z0 = transpose(Phi)*q0


# BUTCHER's TABLEAU
tab = getTableauERK4()
#tab = getTableauHeun()
#tab = getTableauExplicitMidpoint()
#tab = getTableauGLRK(1)


# THE ELECTRIC FIELD
function E(x::Number, param::Number)
    return param^2 * x
end


#THE RHS OF THE EQUATION
function RHS(t,z,zdot,param=beta,U=Phi)

	 x = U*z

         N = div( length(x), 2 )
	 n = length(z)

         xdot1 = x[N+1:2*N]
	 xdot2 = zeros(eltype(x),N)

	 for i=1:N
	     xdot2[i]=-E(x[i],param)
	 end

	 tmp = transpose(U)*[xdot1; xdot2]

	 for i=1:n
	     zdot[i]=tmp[i]
	 end

end



# DEFINITION OF THE EQUATION, INTEGRATOR AND SOLUTION
MyODE = ODE(RHS, z0)        

integr = Integrator(MyODE, tab, dt)

MySol = Solution(MyODE, dt, ntotal, nsave=nsave)

h5 = create_hdf5(MySol, path_out)


# RUN THE SIMULATION AND SAVE TO FILE
# CAREFUL, CALCULATING AND SAVING THE WHOLE THING IN ONE GO

integrate!(integr, MySol)
write_to_hdf5(MySol, h5)

close(h5)


# CALCULATING THE LIFTED SOLUTION
MySol = SSolutionODE(path_out)

x0 = Phi*MySol.q.d[:,:]

q0 = x0[1:N,:]
p0 = x0[N+1:2*N,:]

t = nsave*dt*collect(range(0,nt,step=1))


h5 = h5open(path_out2, "w")
write(h5, "q", q0)
write(h5, "p", p0)
write(h5, "t", t)
close(h5)
