"""
 This script determines the eigen energies for a quantum particle
 with a rectuangular-like confining potential. It does so by setting up 
 both the non-relativistic Schrödigner Hamiltonian and the relativistic 
 Dirac Hamiltonian numerically and diagonalizing them.

 It uses the Fast Fourier transform for estimating the kinetic energy.
 A comparison between relativistic and non-relativistic bound states is
 plotted - in addition to the full relativistic spectrum. The latter 
 consist of both pseudo continuum states and bound states.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
 
 Input for the confining potential:
   V0      - The "height" of the potetial (must be negative for a well)
   w       - The width of the potential
   s       - Smoothness parameter, rectangular well for s -> infinity

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Constants
# Unit mass
m = 1
# The speed of light
c = 137

# Numerical grid parameters
L = 20
N = 512


# Inputs for the smoothly rectangular potential
V0 = -100           # Height, i.e. negative debth
w = 5               # Width
s = 20              # Smootness

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
TransIdentity = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^1 and with (ik)^2, respelctively
Pmat = np.matmul(np.diag(1j*k), TransIdentity)
Tmat = np.matmul(np.diag(-k**2), TransIdentity)
# Transform back to x-representation. 
Pmat = np.fft.ifft(Pmat, axis = 0)
Tmat = np.fft.ifft(Tmat, axis = 0)
# Correct pre-factor
Pmat = -1j*Pmat
Tmat = -1/2*Tmat    

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Non-relativistic Hamiltonian
HamSchrod = Tmat + np.diag(Vpot(x))

# Diagaonalize non-relativistic Hamiltonian (Hermitian matrix)
EvectorSchrod, aux = np.linalg.eigh(HamSchrod)
EvectorSchrod = EvectorSchrod

# Dirac Hamiltonian
HamDirac = np.zeros((2*N, 2*N), dtype = 'complex')
# Upper left block
HamDirac[0:N, 0:N] = np.diag(Vpot(x)) + m*c**2*np.identity(N)
# Lower right block
HamDirac[N:(2*N), N:(2*N)] = np.diag(Vpot(x)) - m*c**2*np.identity(N)
# Upper right block
HamDirac[0:N, N:(2*N)] = c*Pmat 
# Lower left block
HamDirac[N:(2*N), 0:N] = c*Pmat

# Diagaonalize relativistic Hamiltonian (Hermitian matrix)
EvectorDirac, aux = np.linalg.eigh(HamDirac)
EvectorDirac = EvectorDirac

# Find the eigenenergies for bound states
BoundIndexSchrod = np.where(EvectorSchrod < 0)
BoundIndexDirac = np.where((EvectorDirac > -m*c**2) & \
                              (EvectorDirac < m*c**2))

# Plot bound energies
plt.figure(1)
plt.clf()
plt.plot(EvectorSchrod[BoundIndexSchrod], '.', color = 'blue', 
         label = 'Non-relativistic')
plt.plot(EvectorDirac[BoundIndexDirac] - m*c**2, 'o', color = 'red', 
         fillstyle = 'none', label = 'Relativistic')
plt.grid()
plt.xlabel('Index', fontsize = 12)
plt.ylabel('Energy', fontsize = 12)
plt.legend()
plt.show()