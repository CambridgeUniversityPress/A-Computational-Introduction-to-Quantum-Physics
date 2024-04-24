""" 
This script determines the energies for a particle in a 
periodic potential. In such situations, the energies becomes functions 
of the parameter kappa. 
In this particular example, the potential consists of a equivistant
sequence of Gaussian confining potentials.

Numerical inputs:
  L       - The extension of the spatial grid 
  N       - The number of grid points
 
Input for the potential:
  V0      - The depth of the potential
  SigmaP  - The width of the potential

Input for plotting
  NplotE  - The number of energies to display

All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt


# Numerical grid parameters
L = 10
N = 256                

# Parameters from the potential
V0 = -1
SigmaP = 0.1

# The number of energies to plot
NplotE = 5

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Construct vectors with k values to use for the 
# kinetic energy operator
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))

# Potential (as function)
def Vpot(x):
    return V0*np.exp(-x**2/(2*SigmaP**2))


# Set up vector with kappa values 
# note: These differ from the k-values
kappaMax = np.pi/L
Nkappa = 250                   # Number of kappa values to include
kappaVector = np.linspace(-kappaMax, kappaMax, Nkappa)

# Initiate and allocate
indX = 0
EigMat=np.zeros([Nkappa, NplotE])
# Loop over kappa
for kappa in kappaVector:
  # Set up Hamiltonian
  # Kinetic energy - with kappa
  # Transform identity matrix
  Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
  # Multiply by (ik + i kappa)^2
  Tmat_FFT = np.matmul(np.diag((1j*k+1j*kappa)**2), Tmat_FFT)
  # Transform back to x-representation. 
  Tmat_FFT = np.fft.ifft(Tmat_FFT, axis = 0)
  # Correct pre-factor
  Tmat_FFT = -1/2*Tmat_FFT    
  # Total Hamiltonian - with potential
  Ham = Tmat_FFT + np.diag(Vpot(x))          
  # Diagonalize
  Evector, PsiMat = np.linalg.eigh(Ham)
  # Get the lowest energies
  EigMat[indX,:] = Evector[0:NplotE]
  indX = indX+1

# Plot energies
plt.figure(1)
plt.clf()                               # Clear figure
for n in range(0, NplotE):
    plt.plot(kappaVector, EigMat[:, n], linewidth=2)
    
# Labels and grid
plt.ylabel('Energy', fontsize = 12)
plt.xlabel(r'$\kappa$', fontsize = 12)
plt.grid()
plt.show()