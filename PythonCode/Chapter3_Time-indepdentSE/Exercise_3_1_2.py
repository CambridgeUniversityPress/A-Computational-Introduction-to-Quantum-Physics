"""
 This script determines the admissible energies for a bound quantum
 particle confined in a rectuangular-like confining potential. It does so
 by setting up the Hamiltonian, numerically, and diagonalizing it.

 In addition to determining the bound energies, the script also plot
 the corresponding eigen states.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
 
 Input for the confining potential:
   V0      - The "height", the the barrier (must be negative for a well)
   w       - The width of the barrier
   s       - Smoothness parameter, rectangular well for s -> infinity

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Numerical grid parameters
L = 20
N = 512                # Should be 2^k, k integer, for FFT's sake


# Inputs for the smoothly rectangular potential
V0 = -4            # Height, i.e. negative debth
w = 5              # Width
s = 100            # Smootness

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Determine double derivative by means of the fast Fourier transform.
# Set up vector of k-values
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Allocate and declare
Tmat_FFT = np.zeros((N, N), dtype=complex)
# Transform identity matrix
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
Tmat_FFT = np.fft.ifft(Tmat_FFT, axis = 0)
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot(x))


# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham)
# Normalize eigenstates
PsiMat = PsiMat/np.sqrt(h)

# Write bound energies to screen and plot eigenfunctions
n = 0
plt.figure(1)
plt.clf()
while Evector[n] < 0:
    print(f'Admissible bound energy: {Evector[n]:.4f}')
    LegendEntry = 'WF for n = ' + str(n)
    plt.plot(x, PsiMat[:,n], label = LegendEntry)
    n = n+1
    plt.show()

# Insert legend
plt.legend()
plt.grid()
plt.show()