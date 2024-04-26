""" 
This script estimates the energy expectation value for 
the ground state of Hydrogen.
It does so using SI units.

Inputs:
   L       - The extension of the radial grid 
   N       - The number of grid points

Inputs are hard coded initialy, and the function in question is 
selected by commenting in and out the adequate lines.

The input function is used to provide the input parameters from screen.
"""

# Impoet the numpy library
import numpy as np

# Numerical grid parameters
L = float(input('Please provide the extension of your grid (in atomic units): '))
N = int(input('Please provide the number of grid points: '))


# Normalzied wave function
def  PsiFunk(r):
  return 2*r*np.exp(-r)


# Set up grid (avoiding r=0)
h = L/N                   # Increment 
r = np.linspace(h, L, N)
Psi = PsiFunk(r)              # Vector with function values

# Calculate expectation value for Coulumb potential
MeanV = -sum(np.abs(Psi)**2/r)*h              

# Set up vector with Psi''(r)
PsiDoubleDeriv = np.zeros(N, dtype=complex)             # Allocate and declare
# End points (assume Psi = 0 outside the interval)
PsiDoubleDeriv[0] =  (-2*Psi[0] + Psi[1])/(h**2)
PsiDoubleDeriv[N-1] = (Psi[N-2] - 2*Psi[N-1])/(h**2)
# Estimate the derivative with the midpoint rule
for n in range(1,N-1):
  PsiDoubleDeriv[n] = (Psi[n-1] - 2*Psi[n] + Psi[n+1])/(h**2)


# Calculate expectation value for kinetic energy
MeanT = -1/2*sum(np.conj(Psi)*PsiDoubleDeriv)*h   

# Print resulst to screen
print(f'Mean kinetic energy: {np.real(MeanT):.4e} a.u.')
print(f'Mean potential energy: {np.real(MeanV):.4e} a.u.')
print(f'Mean energy: {np.real(MeanT + MeanV):.4e} a.u.')