"""
 This script sets out to fix the duration of the spin-spin interaction 
 between to spin 1/2-particles so that it, in effect, implements the SWAP 
 gate. The parameters are rigged so that the Hamiltonian assumes a 
 particularly simple form. The only inputs are the upper limit for the 
 interaction time T - and dT, the resolution in the vector with 
 time-durations.
 
 It plots the cost function 
 C(T) = 1 - |1/4 * Tr(U_target^\dagger U(T))|^2
 as a function of duration T.
 When C=0, our gate coincides with the target gate, and the fidelity is 100%.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Input
Tmax = 3
dT = 0.01

# Vector with durations
Tvector = np.arange(0, Tmax+dT, dT)

# Hamiltonian
H = np.matrix([[1, 0, 0, 0],
               [0, -1, 2, 0,],
               [0, 2, -1, 0],
               [0, 0, 0, 1]])
# Gate of one step
UdT = linalg.expm(-1j*H*dT)

# Target gate
Uswap = np.matrix([[1, 0, 0, 0], 
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])

# Initiate variables
T = 0
U = np.identity(4)         
index = 0
InFidelity = np.zeros(int(np.floor(Tmax/dT))+1)

# Loop over durations T
for T in Tvector:
  Product = np.matmul(np.conj(Uswap.T), U)
  InFidelity[index] = 1 - np.abs(1/4*np.trace(Product))**2
  # Update gate and index
  U = np.matmul(UdT, U)
  index = index+1

# Plot cost function
plt.figure(1)
plt.clf()
plt.plot(Tvector, InFidelity, '-', linewidth = 2, color = 'black')
# Indicate exact zero-points
plt.vlines(np.pi/4, 0, 1, linewidth = 1.5, color = 'red', 
           linestyles = 'dashed')
plt.vlines(3*np.pi/4, 0, 1, linewidth = 1.5, color = 'red', 
           linestyles = 'dashed')
# Labels, fontsize and grid
plt.xlabel('Interaction duration T', fontsize = 12)
plt.ylabel('C(T)', fontsize = 12)
plt.grid()
plt.show()