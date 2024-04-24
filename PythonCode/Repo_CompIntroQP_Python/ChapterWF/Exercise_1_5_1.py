"""
This script estimates the momentum expectation value
for four different wave functions.

Inputs:
  PsiFunk - The expression for the unnormalized wave function.
  L       - The extension of the spatial grid 
  N       - The number of grid points

Inputs are hard coded initialy, and the function in question is 
selected by commenting in and out the adequate lines.
"""

# Import numpy library
import numpy as np


# Unnormalzied wave functions (Comment in and out as fit)
def  PsiFunk(x):
  # Psi_A:
  #return 1/(1+(x-3)**2)**(3/2)
  # Psi_B:
  return 1/(1+(x-3)**2)**(3/2)*np.exp(-4*1j*x)
  # Psi_C:
  #return np.exp(-x**2)
  # Psi_D:
  #return (x+1j)*np.exp(-(x-3j-2)**2/10)

# Numerical grid parameters
L = 20
N = 50

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = x[1]-x[0]
Psi = PsiFunk(x)                 # Vector with function values

# Normalization
Norm = np.sqrt(np.trapz(np.abs(Psi)**2, x))
Psi = Psi/Norm

# Set up vector with Psi'(x)
PsiDeriv = np.zeros(N, dtype=complex)             # Allocate and declare
# End points (assume Psi = 0 outside the interval)
PsiDeriv[0] =  Psi[1]/(2*h)
PsiDeriv[N-1] = -Psi[N-1]/(2*h)
# Estimate the derivative with the midpoint rule
for n in range(1,N-1):
  PsiDeriv[n] = (Psi[n+1]-Psi[n-1])/(2*h)
  
# Calculate expectation value
MeanP = -1j*np.trapz(np.conj(Psi)*PsiDeriv, x)   # Mean momentum
print(f'Mean momentum: {np.real(MeanP):.4f}')
print(f'Imaginary part: {np.imag(MeanP):.4f}')