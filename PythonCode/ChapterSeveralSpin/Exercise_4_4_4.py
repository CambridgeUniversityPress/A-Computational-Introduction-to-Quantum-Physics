"""
 This script estimates the ground state energy of a one-dimensional
 two-particle system by the method self-consistent field. We assume 
 that the ground can be approxmated by a simple product of two identical 
 wave functions for each particle. From this wave function an effective 
 potential is calculated. The Hamiltonian of the resulting effective 
 one-particle Hamiltonian is diagonalsed and the ground state of this 
 is taken as our updated one-particle wave function. This is iterated 
 until the effective one-particle energy is converged.

 As initial guess we have taken the one-particle ground state.
 
 This is a simple example of a Hartree-Fock calculation.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Epres   - The energy precision, used to set convergence criteria

 Input for the confining potential:
   V0      - The "height", the the barrier (must be negative for a well)
   w       - The width of the barrier
   s       - Smoothness parameter, rectangular well for s -> infinity

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Numerical grid parameters
L = 20
N = 512                

# Paraeter to set energy precision
Epres = 1e-5

# Inputs for the smoothly rectangular potential
V0 = -1            # Height, i.e. negative debth
w = 4              # Width
s = 5              # Smootness

# Input for the interaction
W0 = 1

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)
# Meshgrid for interaction
X1, X2 = np.meshgrid(x, x);

# Determine double derivative by means of the fast Fourier transform.
# Set up vector of k-values
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
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Interaction (as function)
def Wint(x1, x2):
    return W0/np.sqrt((x1-x2)**2+1)


# One-particle Hamiltonian
HamOnePart = Tmat_FFT + np.diag(Vpot(x))

# Matrix for intearction
Wmat = Wint(X1, X2)

# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(HamOnePart)
# Normalize and initialize zeroth order wave function
Psi = PsiMat[:,0]/np.sqrt(h)
Psi = Psi.reshape(N,1)
EnergyOnePart = Evector[0]
# Initialte old energy
EnergyOnePartOld = 1e6;

# Allocate effective potential
Veff = np.zeros((N,1))

# Iterate until self-consistency
iterations = 0
while np.abs(EnergyOnePart-EnergyOnePartOld) > Epres:
  # Set up effective interaction potential
  PsiSq = np.abs(Psi)**2
  Veff = h*np.matmul(Wmat, PsiSq)
  # Note that Veff has to be reshaped for the diag function in numpy
  Heff = HamOnePart + np.diag(Veff.reshape(N,))
  # Diagonalize effective Hamiltonian
  Evector, PsiMat = np.linalg.eigh(Heff)
  Psi = PsiMat[:,0]/np.sqrt(h)
  Psi = Psi.reshape(N,1)
  # Update energy
  EnergyOnePartOld = EnergyOnePart
  EnergyOnePart = np.real(Evector[0])
  iterations = iterations + 1
  print(f'Iteration: {iterations}, one-particle energy: {EnergyOnePart:.6f}')
  
# Determine energy estimate
EonePart = h*np.matmul(np.conj(Psi.T), np.matmul(HamOnePart,Psi))
PsiTwoPart = np.matmul(Psi, Psi.T)
EtwoPart = h**2*np.sum(np.multiply(Wmat, np.abs(PsiTwoPart)**2))
Etotal = float(2*np.real(EonePart) + np.real(EtwoPart))
# Write result to screen:
print(f'Two-particle energy estimate: {Etotal:,f}')
print(f'Iterations: {iterations}')