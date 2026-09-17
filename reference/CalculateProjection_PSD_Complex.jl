using GeometricIntegrators
using HDF5
using LinearAlgebra


# Input and output files
paths = ["/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta595_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001.hdf5",
         "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta597_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001.hdf5",
         "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta599_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001.hdf5",
         "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta601_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001.hdf5",
         "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta603_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001.hdf5",
         "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_beta605_sigmax10_a03_sigmav05_v4_T1000_n1000_dt0001.hdf5"]


path_out =  "/toks/work/ttoma/ModelReductionQuadratic/LIBRARY/Vlasov_Analytic_PSD_Complex_SVD_dt001_skip10_T1000.hdf5"


skip    = 10
max_idx = 1000001



# CALCULATING THE SVD

M=[]

for file in paths

    MySol = h5open(file,"r")
    q = read(MySol,"q")

    if isempty(M)
        global M = q[:,1:skip:max_idx]
    else
        global M = [M q[:,1:skip:max_idx] ]
    end

end


N = div( size(M)[1], 2 )

M1 = M[1:N,:] + M[N+1:2*N,:] * im


M1_SVD = svd(M1)

#SAVING TO A FILE
h5 = h5open(path_out, "w")
write(h5, "Singular values", M1_SVD.S)
write(h5, "Phi", real(M1_SVD.U) )
write(h5, "Psi", imag(M1_SVD.U) )
close(h5)
