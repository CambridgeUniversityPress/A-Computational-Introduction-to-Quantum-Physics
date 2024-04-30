"""
This script estimates expectation values for the position
x for four given wave functions. It also plots them.

Inputs:
   PsiFunk - The expression for the unnormalized wave function.
   L       - The extension of the spatial grid 
   N       - The number of grid points

Inputs are hard coded initialy, and the function in question is 
selected by commenting in and out the adequate lines.
"""

# Import libraries (numpy and matplotlib)
import numpy as np
from matplotlib import pyplot as plt

# Unnormalzied wave functions (Comment in and out as fit)
def  PsiFunk(x):
  # Psi_A:
  # return 1/(1+(x-3)**2)**(3/2)
  # Psi_B:
  return 1/(1+(x-3)**2)**(3/2)*np.exp(-4*1j*x)
  # Psi_C:
  #return np.exp(-x**2)
  # Psi_D:
  #return (x+1j)*np.exp(-(x-3j-2)**2/10)

# Numerical grid parameters
L = 20
N = 100

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = x[1]-x[0]
Psi = PsiFunk(x)                 # Vector with function values

# Normalization
Norm = np.sqrt(np.trapz(np.abs(Psi)**2, dx = h))
Psi = Psi/Norm

# Make plot
plt.figure(1)
plt.clf()
plt.plot(x, np.abs(Psi)**2, '-', color='black', label = r'$|\Psi|^2$')
plt.xlabel('x')
# Check if Psi is complex and, if so, plot real and imaginary contributions
if np.max(np.abs(np.imag(Psi)))>1e-7:     
  plt.plot(x, np.real(Psi)**2, '--', color = 'blue', 
           label = r'$(Re \; \Psi)^2$')
  plt.plot(x, np.imag(Psi)**2, '-.', color='red', label = r'$(Im \; \Psi)^2$')
  plt.legend()

plt.grid()
plt.show()

# Calculate mean position and write result to screen
MeanX = np.trapz(x*np.abs(Psi)**2, x)
print(f'Mean position: {MeanX:.4f}' )
