"""
This script estimates expectation values for the kinetic energy
for four given wave functions. It does so by constructing matrix
representations for the kinetic energy operator and then estimate
the expectation value by means of matrix products.

 Inputs:
   PsiFunk - The expression for the unnormalized wave function.
   L       - The extension of the spatial grid 
   N       - The number of grid points

Inputs, indluding the expression for the wave function, are hard coded 
initially.
"""   
  
# Import the numpy library
import numpy as np

# Unnormalzied wave functions (Comment in and out as fit)
def  PsiFunk(x):
  # Psi_A:
  return 1/(1+(x-3)**2)**(3/2)
  # Psi_B:
  #return 1/(1+(x-3)**2)**(3/2)*np.exp(-4*1j*x)
  # Psi_C:
  #return np.exp(-x**2)
  # Psi_D:
  #return (x+1j)*np.exp(-(x-3j-2)**2/10)

# Numerical grid parameters
L = 20
N = 64

# Set up grid
x = np.linspace(-L/2, L/2, N)
Psi = PsiFunk(x)                 # Vector with function values
h = L/(N-1)

# Normalization
Norm = np.sqrt(np.trapz(np.abs(Psi)**2, x))
Psi = Psi/Norm
# Turn into column vector (N \times 1 matrix)
Psi = np.matrix(Psi)                 
Psi = Psi.reshape(N,1)

#
# Set up matrix for the kinetic energy - 3 point formula
#
# Allocate and declare
Tmat_FD3 = np.zeros((N, N), dtype=complex)
# Endpoints
Tmat_FD3[0, 0:2] = [-2, 1]
Tmat_FD3[N-1, (N-2):N] = [1, -2]
# Interior points
for n in range (1,N-1):
  Tmat_FD3[n, [n-1, n, n+1]] = [1, -2, 1]
# Correct pre-factors
Tmat_FD3 = Tmat_FD3/h**2
Tmat_FD3 = -1/2*Tmat_FD3  

#  
# Set up matrix for the kinetic energy - 5 point formula
#
# Allocate and declare
Tmat_FD5 = np.zeros((N, N), dtype=complex)
# Endpoints (and neighbours)
Tmat_FD5[0, 0:3] = [-30, 16, -1]
Tmat_FD5[1, 0:4] = [16, -30, 16, -1]
Tmat_FD5[N-2, (N-4):N] = [-1, 16, -30, 16]
Tmat_FD5[N-1, (N-3):N] = [-1, 16, -30]
# Interior points
for n in range (2,N-2):
  Tmat_FD5[n, range(n-2,n+3)] = [-1, 16, -30, 16, -1]
# Correct pre-factors
Tmat_FD5 = Tmat_FD5/(12*h**2)
Tmat_FD5 = -1/2*Tmat_FD5  

#
# Determine double derivative by means of the fast Fourier transform.
#
# Set up vector of k-values
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Transform identity matrix
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
Tmat_FFT = np.fft.ifft(Tmat_FFT, axis = 0)
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

#
# Calculate expectation values, <T> = h * Psi^\dagger T Psi
#
# 3 point rule
MeanT_FD3 = h*np.matmul(Psi.H, np.matmul(Tmat_FD3, Psi))
MeanT_FD3 = complex(MeanT_FD3)
# 5 point rule
MeanT_FD5 = h*np.matmul(Psi.H, np.matmul(Tmat_FD5, Psi))
MeanT_FD5 = complex(MeanT_FD5)
# FFT
MeanT_FFT = h*np.matmul(Psi.H, np.matmul(Tmat_FFT, Psi))
MeanT_FFT = complex(MeanT_FFT)

# Print results to screen (only real part):
print('Kinetic energy energy estimates:')
print(f'Three point-forumla: {np.real(MeanT_FD3):.4f}')
print(f'Five point-forumla: {np.real(MeanT_FD5):.4f}')
print(f'Fast Fourier transform: {np.real(MeanT_FFT):.4f}')
