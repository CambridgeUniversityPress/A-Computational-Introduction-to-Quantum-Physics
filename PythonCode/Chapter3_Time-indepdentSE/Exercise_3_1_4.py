""" 
 This script determines the energies for a quantum particle of unit mass
 confined within a harmonic oscillator potential. It does so by setting 
 up the Hamiltonian, numerically, and diagonalizing it.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
 
 Input for the harmonic potential:
   kStrength - the strength of the potential

 Inputs for plotting
   NplotE  - The number of energies to display
   NplotWF - The number of wave functions to display

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt


# Numerical grid parameters
L = 15
N = 512                

# Strength of harmonic potential
kStrength = 1

# The number of plots
NplotWF = 4
NplotE = 50

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Determine double derivative by means of the fast Fourier transform.
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

# Potential (as function)
def Vpot(x):
    return 0.5*kStrength*x**2

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot(x))


# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham)
# Normalize eigenstates
PsiMat = np.real(PsiMat)/np.sqrt(h)

# Plot some of the bound energies together with the analytical formula
plt.figure(1)
plt.clf()                               # Clear figure
IndexVect = range(0, NplotE)
# Numerical values
plt.plot(IndexVect, Evector[0:NplotE], 'rx', linewidth = 2, 
         label = 'Numerical eigenenergies')
# Analytical values
plt.plot(IndexVect, np.sqrt(kStrength)*(np.asarray(IndexVect)+1/2), 
         'ko', linewidth = 2, markerfacecolor = 'none', 
         label = 'Analytical eigenenergies')
plt.xlabel('Quantum number n', fontsize = 12)
plt.ylabel('Energy', fontsize = 12)
plt.legend()
plt.grid()
plt.show()

# Plot a few wave functions
plt.figure(2)
plt.clf()                               # Clear figure
for n in range(0,NplotWF):
    LegendEntry = r'$\Psi_n$ for n = ' + str(n+1)
    plt.plot(x, PsiMat[:,n ], label = LegendEntry)

# Axes labels and grid
plt.xlabel('x', fontsize = 12)
plt.grid()
plt.legend()
plt.show()   