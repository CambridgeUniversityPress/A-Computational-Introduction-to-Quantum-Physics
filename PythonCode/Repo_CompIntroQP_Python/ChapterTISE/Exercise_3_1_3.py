""" 
 This script determines the admissible energies for a hydrogen atom.
 It does so by setting up the Hamiltonian, numerically, and 
 diagonalizing it. It does so by using a three point finite difference
 formula. The boundary conditions at r = 0 and r = L, where L is the size
 of the numerical domain, is included manifestly in our approximation.

 In addition to determining the bound energies, the script also plots the
 admissible energies and compares the spectrum with the Bohr formula.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
 
 The number of wave functions to plot: Nplot
 
 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt


# Numerical grid parameters
L = 50;
N = 1000;                

# The number of plots
Nplots = 8;

# Set up grid. Here r = 0 and r = L are excluded. It does, however, use
# that psi(0) = psi(L) = 0.
h = L/(N+1);
r = np.linspace(h, L-h, N);      
# Convert to column vector
r = np.transpose(r)

# Set up potential
Vpot = -1/r

# Set up matrix for the kinetic energy - 3 point formula
# Allocate and declare
Tmat_FD3 = np.zeros((N, N), dtype=complex)
# Endpoints
Tmat_FD3[0, :2] = [-2, 1]
Tmat_FD3[N-1, (N-2):] = [1, -2]
# Interior points
for n in range (1,N-1):
  Tmat_FD3[n, [n-1, n, n+1]] = [1, -2, 1]
# Correct pre-factors
Tmat_FD3 = Tmat_FD3/h**2
Tmat_FD3 = -1/2*Tmat_FD3  



# Full Hamiltonian
Ham = Tmat_FD3 + np.diag(Vpot);         

# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham)
# Normalize eigenstates
PsiMat = PsiMat/np.sqrt(h)

# Plot eigenfunctions and write bound energies to screen
plt.figure(1)
plt.clf()
n = 0
while Evector[n] < 0:
    print(f'Admissible bound energy: {Evector[n]:.4f}')
    LegendEntry = r'$\Psi_n$ for n = ' + str(n+1)
    plt.plot(r, PsiMat[:,n], label = LegendEntry)
    n = n+1

# Insert legend and grid
plt.grid()
plt.xlabel('r [a. u.]', fontsize = 12)
plt.legend()
plt.show()

# Plot comparison to the Bohr formula
Nbound = n
plt.figure(2)
plt.clf()
plt.plot(range(1,Nbound+1), np.real(Evector[:Nbound]), 'x', 
         color = 'red', label = 'Numerical')    
plt.plot(range(1,Nbound+1), -0.5/np.linspace(1, Nbound, Nbound)**2, 
         'o', color = 'black', mfc = 'none', label = 'Bohr formula')
plt.grid()
plt.xlabel('Quantum number n', fontsize = 12)
plt.ylabel('Energy [a.u.]', fontsize = 12)
plt.legend()
plt.show()