""" 
This script estimates the energy expectation value for 
the ground state of Hydrogen.
It does so using SI units.

Inputs:
   L       - The extension of the radial grid 
   N       - The number of grid points

The input function is used to provide the input parameters from screen.
"""

# Import the numpy library
import numpy as np

# Numerical grid parameters
L = float(input('Please provide the extension of your grid (in metres): '))
N = int(input('Please provide the number of grid points: '))

# Relevant constants in SI units
a0 = 5.292e-11                  # The Bohr radius
hbar = 1.055e-34                # The reduced Planck constant 
Eps0 = 8.854e-12                # Permittivity of free space
ElemCharge = 1.602e-19          # Elementary charge
mass = 9.11e-31                 # Electron mass

# Normalzied wave function
def  PsiFunk(r):
  return 2/a0**(3/2)*r*np.exp(-r/a0)


# Set up grid (avoiding r=0)
h = L/N                   # Increment 
r = np.linspace(h, L, N)
Psi = PsiFunk(r)              # Vector with function values

# Calculate expectation value for Coulumb potential
MeanV = -ElemCharge**2/(4*np.pi*Eps0)*np.trapz(np.abs(Psi)**2/r, r)              

# Set up vector with Psi''(r)
PsiDoubleDeriv = np.zeros(N, dtype=complex)             # Allocate and declare
# End points (assume Psi = 0 outside the interval)
PsiDoubleDeriv[0] =  (-2*Psi[0] + Psi[1])/(h**2)
PsiDoubleDeriv[N-1] = (Psi[N-2] - 2*Psi[N-1])/(h**2)
# Estimate the derivative with the midpoint rule
for n in range(1,N-1):
  PsiDoubleDeriv[n] = (Psi[n-1] - 2*Psi[n] + Psi[n+1])/(h**2)


# Calculate expectation value for kinetic energy
MeanT = -hbar**2/(2*mass)*np.trapz(np.conj(Psi)*PsiDoubleDeriv, r)   

# Print results to screen
print(f'Mean kinetic energy: {np.real(MeanT):.4e} J')
print(f'Mean potential energy: {np.real(MeanV):.4e} J')
print(f'Mean energy (SI): {np.real(MeanT + MeanV):.4e} J')
print(f'Mean energy (eV): {np.real(MeanT + MeanV)/ElemCharge:.4e} eV')