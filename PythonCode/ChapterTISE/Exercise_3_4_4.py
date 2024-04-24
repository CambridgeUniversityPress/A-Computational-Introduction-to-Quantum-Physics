"""
 This script estimates the ground state energy of a specific potential 
 by means of the variational principle. As trial function we use a Gaussian 
 wave packet with varable width mean position.

 The Hamiltonian is estimated numerically using an FFT representation of 
 the kinetic energy.

 The ground state energy is also calculated "exactly" by diagoalizing 
 the full, numerical Hamiltonian.

 Inputs for the gradient decent approach:
 Sigma0    - starting point for the width
 Mu0       - starting point for the mean position
 Gamma     - the so-called learning rate
 GradientMin  - the lower limit for the length of the gradient
 DerivStep    - the numerical step size for estimating the gradient
 
 Grid parameters
 L         - size of domain 
 N         - number of grid points (should be 2^n)

 Physical inputs:
 V0        - the height of the smoothl rectangular potential (negative)
 w         - the width of the potential
 s         - the "smoothness" of the potential 
 Vpot      - the potential - which has an additional square term

 All parameters and functions are hard coded initially
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Parameters for the gradient descent method
Sigma0 = 2;
Mu0 = 4;
Gamma = 0.1;
GradientMin = 1e-3;
DerivStep = 1e-1;

# Numerical grid parameters
L = 20
N = 512                

# Inputs for the smoothly rectangular part of the potential
V0 = -5
w = 6
s = 4

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Test function
def PsiFunk(x, Sigma, Mu):
    return (2*np.pi*Sigma**2)**(-0.25) * np.exp(-(x-Mu)**2/(4*Sigma**2))

# Kinetic energy operator (FFT)
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Transform identity matrix
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
Tmat_FFT = np.fft.ifft(Tmat_FFT, axis = 0)
# Correct pre-factor (unit mass and \hbar = 1)
Tmat_FFT = -1/2*Tmat_FFT    

# Potential (as function)
def Vsmooth(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

def Vpot(x):
    return Vsmooth(x-2) + x**2/50

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot(x))
# Ensure real Hamiltonian
Ham = np.real(Ham)

# Function that calculates the energy expectation value
def EnergyExpectation(sigma, mu):
   Psi = PsiFunk(x, sigma, mu)
   # Column vector
   Psi = Psi.reshape(N,1)
   Aux = np.matmul(Ham, Psi)
   return h*np.matmul(np.transpose(np.conj(Psi)), Aux)

# 
# Gradient descent method
#
# Initiate
index = 0
sigma = Sigma0
mu = Mu0
LengthGradient = 1e6;           # Fix high value to get the loop going
# Iterate while the gradient is large enough
while LengthGradient > GradientMin:
  # Calculate new energy estimate
  EnergyEstimate = EnergyExpectation(sigma, mu)
 
  # Estimate partial derivatives by means of midpoint rule
  dEdSigma = (EnergyExpectation(sigma+DerivStep, mu) - 
      EnergyExpectation(sigma-DerivStep, mu))/(2*DerivStep)
  dEdMu = (EnergyExpectation(sigma, mu+DerivStep) - 
      EnergyExpectation(sigma, mu-DerivStep))/(2*DerivStep)
  # Calculate length of the gradient
  LengthGradient = np.sqrt(dEdSigma**2 + dEdMu**2)
  
  # Update sigma and mu
  sigma = sigma - Gamma*dEdSigma
  mu = mu - Gamma*dEdMu
  
  # Write estimate to screen for every 10th iteration
  if np.mod(index, 10) == 0:
      print(f'Iteration: {index}, energy: {float(EnergyEstimate):.4f}')
  # Update index
  index=index+1;                          

# Assign estimate
EnergyMin = float(EnergyEstimate)
    
# In order to compare with "exact" result:
# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham)
# Extract ground state
GroundStateEnergy = Evector[0]
GroundState = PsiMat[:,0]
# Normalize
GroundState = GroundState/np.sqrt(h)

# Write restults to screen
print(f'Minimal energy: {EnergyMin:.4f}')
print(f'Actual ground state energy: {GroundStateEnergy:.4f}')
Error = EnergyMin-GroundStateEnergy
print(f'Error: {Error:.4f}')
RelError = Error/np.abs(GroundStateEnergy)
print(f'Relative error: {RelError*100:2.2f} %')


# Plot wave functions 
# - both the estimate that minimizes E and the actual one
plt.figure(1)
plt.clf()
# Variational
PsiVar = PsiFunk(x, sigma, mu)
plt.plot(x, np.abs(PsiVar.T)**2, 'k-', label = 'Variatonal')
# Actual
plt.plot(x, np.abs(GroundState)**2, 'r--', label = 'Actual')
# Potential
ScaleFactor = np.max(np.abs(PsiVar)**2)
ScaleFactor = 0.2*ScaleFactor/np.abs(V0)
plt.plot(x, ScaleFactor*Vpot(x), 'b:', label = 'Potential')
# Cosmetics
plt.xlabel('x', fontsize = 12)
plt.ylabel(r'$|\Psi(x)|^2$', fontsize = 12)
plt.legend()
plt.grid()
plt.show()
