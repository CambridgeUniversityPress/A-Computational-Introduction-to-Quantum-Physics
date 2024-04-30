"""
 This script estimates the ground state energy for a two-particle system 
 consisting of two identical fermions. The particles interact via a smooth 
 Coulomb-like interaction, and they are confined by a "smooth" rectangular 
 potential.
 The ground state energy is estimated by means of the variational principle. 
 As a trial function we use a product of two idential Gaussian wave packets 
 with a variable width.

 The Hamiltonian is estimated numerically using an
 FFT representation of the kinetic energy.

 The estimated energy is plotted as a function of the
 width.

 Numerical inputs:
 SigmaMin  - minimal value for the widht
 SigmaMax  - maximal value for the width
 SigmaStep - step size used for the width
 L         - size of domain 
 N         - number of grid points (should be 2^n)

 Physical inputs:
 V0        - the height of the potential (should be negative)
 w         - the width of the potential
 s         - the "smoothness" of the potential 
 W0        - the strength of the interaction

 All parameters and functions are hard coded initially
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Grid for the sigma parameter
SigmaMin = 0.1
SigmaMax = 3
SigmaStep = 0.005

# Numerical grid parameters
L = 10
N = 512                

# Inputs for the smoothly rectangular potential and interaction
V0 = -1            # Height, i.e. negative debth
w = 4              # Width
s = 5              # Smootness
W0 = 1             # Interaction strength

# Set up grids
# x-grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)
# sigma-grid
Nsigma = int(np.floor((SigmaMax-SigmaMin)/SigmaStep))+1
SigmaVector = np.linspace(SigmaMin, SigmaMax, Nsigma)
# Two-particle grid
X1, X2 = np.meshgrid(x, x)

#
# Simple functions: Trial function, potential and interaction
#
# Trial function
def PsiFunk(x, Sigma):
    return (2*np.pi*Sigma**2)**(-0.25)*np.exp(-x**2/(4*Sigma**2))

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Interaction
def Wint(x1, x2):
    return W0/np.sqrt((x1-x2)**2+1)

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


# Full one-particle Hamiltonian
HamOnePart = Tmat_FFT + np.diag(Vpot(x))
# Ensure real Hamiltonian
HamOnePart = np.real(HamOnePart)

# Matrix with interaction
WintMat = Wint(X1, X2)

# Function that calculates the energy expectation value
def EnergyExpectation(sigma):
   # One-particle wave function (as column vector)
   PsiOnePart = PsiFunk(x, sigma)
   PsiOnePart = PsiOnePart.reshape(N,1)
   
   # One particle energy
   EnergyOnePart = h*np.matmul(PsiOnePart.T, 
                np.matmul(HamOnePart, PsiOnePart))
   
   # Interaction energy
   PsiTwoPart = np.matmul(PsiOnePart, 
                np.transpose(PsiOnePart))
   EnergyTwoPart = h**2*np.sum(np.multiply(WintMat, 
                np.abs(PsiTwoPart)**2))
   
   # Total Energy
   EnergyTotal = 2*EnergyOnePart + EnergyTwoPart
   return float(np.real(EnergyTotal))

# Initiate and allocate for sigma-loop
Sigma = SigmaMin
SigmaInd = 0
EnergyMin = 1e6
EnergyVector = np.zeros(Nsigma)
# Loop
for Sigma in SigmaVector:
    # Energy expectation value
    Energy = EnergyExpectation(Sigma)
    # Check for new minimum
    if Energy < EnergyMin:
        EnergyMin = Energy
        SigmaForMinimum = Sigma
    # Copy energy to vector
    EnergyVector[SigmaInd] = Energy
    # Update width and index
    Sigma = Sigma + SigmaStep
    SigmaInd = SigmaInd + 1        
    
# Write restults to screen
print(f'Minimal energy: {EnergyMin:.4f}')
print(f'Optimal sigma: {SigmaForMinimum:.4f}')

# Plot energy expectation value
plt.figure(1)
plt.clf()
plt.plot(SigmaVector, EnergyVector, 'k-', label = '<E($\sigma$)>')
plt.xlabel(r'$\sigma$', fontsize = 12)
plt.ylabel('< E >', fontsize = 12)
plt.grid()
plt.show()