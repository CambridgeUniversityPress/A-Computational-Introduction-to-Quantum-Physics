"""
This script calculates the width in position
and momentum for four given wave functions. 

Inputs:
   PsiFunk - The expression for the unnormalized wave function.
   L       - The extension of the spatial grid 
   N       - The number of grid points
      
Inputs are hard coded initialy, and the function in question is 
selected by commenting in and out the adequate lines.
"""

# Import the numpy library
import numpy as np


# Unnormalzied wave functions (Comment in and out as fit)
def  PsiFunk(x):
  # Psi_A:
  return 1/(1+(x-3)**2)**(3/2)
  # Psi_B:
  #return 1/(1+(x-3)**2)**(3/2)*np.exp(-4*1j*x)
  # Psi_C:
  #return np.exp(-x**2)
  # Psi_D:
  #return (x+1j)*np.exp(-(x-3j-2)**2/10)

# Numerical grid parameters
L = 20
N = 200

# Set up grid
x = np.linspace(-L/2, L/2, N)
Psi = PsiFunk(x)                  # Vector with function values
h = L/(N-1)                       # Increment

# Normalization
Norm = np.sqrt(np.trapz(np.abs(Psi)**2, x))
Psi = Psi/Norm

# Set up vector with Psi'(x)
PsiDeriv = np.zeros(N, dtype=complex)           # Allocate and declare
# End points (assume Psi = 0 outside the interval)
PsiDeriv[0] =  Psi[1]/(2*h)
PsiDeriv[N-1] = -Psi[N-1]/(2*h)
# Estimate the derivative with the midpoint rule
for n in range(1,N-1):
  PsiDeriv[n] = (Psi[n+1]-Psi[n-1])/(2*h)

# Set up vector with Psi''(x)
PsiDoubleDeriv = np.zeros(N, dtype=complex);    # Allocate and declare
# End points (assume Psi = 0 outside the interval)
PsiDoubleDeriv[0] =  (-2*Psi[1]+Psi[2])/h**2
PsiDoubleDeriv[N-1] = (Psi[N-2]-2*Psi[N-1])/h**2
# Estimate the derivative with three point-rule
for n in range(1,N-1):
  PsiDoubleDeriv[n] = (Psi[n-1]-2*Psi[n]+Psi[n+1])/h**2

# Calculate x expectation value
MeanX = np.trapz(x*np.abs(Psi)**2, x)
# Calculate x^2 expectation value
MeanXsq = np.trapz(x**2*np.abs(Psi)**2, x) 
# Calculate p expectation value
MeanP = -1j*np.trapz(np.conj(Psi)*PsiDeriv, x)
# Calculate p^2 expectation value
MeanPsq = -np.trapz(np.conj(Psi)*PsiDoubleDeriv, x)

# Determine widths
SigmaX = np.sqrt(MeanXsq - MeanX**2)
SigmaP = np.sqrt(MeanPsq - MeanP**2)
print(f'Position width: {SigmaX:.4f}')
print(f'Momentum width (real part): {np.real(SigmaP):.4f}')

# Check uncertainty principle
Product = SigmaX*SigmaP
print(f'Product of the uncertainties (real part): {np.real(Product):.4f}')
      