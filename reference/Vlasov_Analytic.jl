using GeometricIntegrators
using HDF5
using Random

# PARAMETERS
beta   = 6.05
sigmaX = 10
sigmaV = 0.5
v0     = 4.0
a      = 0.3
dt     = 0.001                    # time step
nt     = 1000000                  # number of time steps to compute in one chunk
n      = 1000                     # number of particles
seed1  = 1 	     		  # the random number generator seed
seed2  = 10
seed3  = 100


# OUTPUT FILE
path = "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta605_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001_REFERENCE.hdf5"




# GENERATING THE INITIAL CONDITIONS FOR X

Random.seed!(seed1)
q0 = sigmaX*randn(n)


# GENERATING THE INITIAL CONDITIONS FOR V

Random.seed!(seed2)
p0 = randn(n)

# bump on tail
Random.seed!(seed3)

for i in 1:n

     tmp = rand()

     if tmp >= 1/(1+a)
         p0[i] = sigmaV*p0[i] + v0
     end
end






# GENERATING THE SOLUTION ARRAY
# q contains both positions and momenta
q = zeros(Float64, 2*n, nt+1)
t = collect(dt*range(0,nt,step=1))

for i in 1:nt+1

	for j in 1:n
		q[j,i] =  (p0[j]/beta)*sin(beta*t[i]) + q0[j]*cos(beta*t[i])
		q[j+n,i] =  p0[j]*cos(beta*t[i]) - beta*q0[j]*sin(beta*t[i])
	end

end


# SAVING TO FILE

h5 = h5open(path, "w")
write(h5, "q", q)
write(h5, "t", t)

attrs(h5)["nd"] = 1
attrs(h5)["n"] = n
attrs(h5)["ni"] = 1
attrs(h5)["nt"] = nt
attrs(h5)["seed1"] = seed1
attrs(h5)["seed2"] = seed2
attrs(h5)["seed3"] = seed3

close(h5)
