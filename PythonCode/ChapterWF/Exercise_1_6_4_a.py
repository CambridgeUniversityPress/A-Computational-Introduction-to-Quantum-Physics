""" 
This script estimates expectation values for r
and p for the ground state of Hydrogen.
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
a0 = 5.292e-11                # The Bohr radius
hbar = 1.055e-34              # The reduced Planck constant 

# Normalzied wave function
def  PsiFunk(r):
  return 2/a0**(3/2)*r*np.exp(-r/a0)


# Set up grid
r = np.linspace(0, L, N)
Psi = PsiFunk(r)              # Vector with function values
h = L/(N-1)                   # Increment 

# Calculate expectation value for r
MeanX = sum(r*np.abs(Psi)**2)*h                # Mean position
print(f'Mean position: {MeanX:.4e} m')

# Set up vector with Psi'(r)
PsiDeriv = np.zeros(N, dtype=complex)             # Allocate and declare
# End points (assume Psi = 0 outside the interval)
PsiDeriv[0] =  Psi[1]/(2*h)
PsiDeriv[N-1] = -Psi[N-1]/(2*h)
# Estimate the derivative with the midpoint rule
for n in range(1,N-1):
  PsiDeriv[n] = (Psi[n+1]-Psi[n-1])/(2*h)


# Calculate expectation value for p
MeanP = -1j*hbar*sum(np.conj(Psi)*PsiDeriv)*h   
print(f'Mean momentum: {np.real(MeanP):.4e} kg m/s')
print(f'Imaginary part: {np.imag(MeanP):.4e} kg m/s')