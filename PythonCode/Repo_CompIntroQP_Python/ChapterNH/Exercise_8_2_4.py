"""
 This script determines the admissible energies for a bound quantum
 particle confined in a Gaussian confining potential. It does so
 by setting up the Hamiltonian, numerically, and diagonalizing it.

 In addition to determining the bound energies, the script also plot
 the corresponding eigen states.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
 
 Input for the confining potential:
   V0      - The "height", the the barrier (must be negative for a well)
   Sigma   - The width of the barrier

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Numerical grid parameters
L = 20
N = 512                # Should be 2^k, k integer, for FFT's sake


# Inputs for the smoothly rectangular potential
V0 = -1            # Height, i.e. negative debth
Sigma = np.sqrt(2)          # Width

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
Tmat = np.zeros((N, N), dtype=complex)
# Transform identity matrix
Tmat = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat = np.matmul(np.diag(-k**2), Tmat)
# Transform back to x-representation. 
Tmat = np.fft.ifft(Tmat, axis = 0)
# Correct pre-factor
Tmat = -1/2*Tmat    

# Potential (as function)
def Vpot(x):
    return V0*np.exp(-x**2/(2*Sigma**2))

# Full Hamiltonian
Ham = Tmat + np.diag(Vpot(x))


# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham)
# Normalize eigenstates
PsiMat = PsiMat/np.sqrt(h)