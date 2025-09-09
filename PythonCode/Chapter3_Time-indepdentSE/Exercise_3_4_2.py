"""
 This script estimates the ground state energy of a "smooth" rectangular 
 potential by means of the variational principle. As a trial function we 
 use a Gaussian wave packet centered at the origin with a variable width.

 The Hamiltonian is estimated numerically using an FFT representation of 
 the kinetic energy.

 The ground state energy is also calculated "exactly" by diagoalizing the 
 full, numerical Hamiltonian.

 The estimated energy is plotted as a function of the width - along with 
 the "exact" ground state energy

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
 Vpot      - the actual potential (function variable)

 All parameters and functions are hard coded initially
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Grid for the sigma parameter
SigmaMin = 0.1
SigmaMax = 5
SigmaStep = 0.05

# Numerical grid parameters
L = 30
N = 2048                


# Inputs for the smoothly rectangular potential
V0 = -3            # Height, i.e. negative debth
w = 8              # Width
s = 5              # Smootness

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Set up grids
# x-grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)
# sigma-grid
Nsigma = int(np.floor((SigmaMax-SigmaMin)/SigmaStep))+1
SigmaVector = np.linspace(SigmaMin, SigmaMax, Nsigma)

# Trial function
def PsiFunk(x, Sigma):
    return (2*np.pi*Sigma**2)**(-0.25)*np.exp(-x**2/(4*Sigma**2))

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

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot(x))
# Ensure real Hamiltonian
Ham = np.real(Ham)

# Initiate and allocate
Sigma = SigmaMin
SigmaInd = 0
EnergyMin = 1e6
EnergyVector = np.zeros(Nsigma)
for Sigma in SigmaVector:
    # Trial function
    Psi = PsiFunk(x, Sigma)
    # Hermitian adjoint
    PsiDagger = np.transpose(Psi)
    PsiDagger = np.conj(PsiDagger)
    # Energy expectation value
    Energy = h*np.matmul(PsiDagger, np.matmul(Ham, Psi))
    # Check for new minimum
    if Energy < EnergyMin:
        EnergyMin = Energy
        SigmaForMinimum = Sigma
    # Write energy to vector
    EnergyVector[SigmaInd] = Energy
    # Update
    Sigma = Sigma + SigmaStep
    SigmaInd = SigmaInd + 1        
    
# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham)
# Extract ground state
GroundStateEnergy = Evector[0]
GroundState = PsiMat[:,0]
# Normalize
GroundState = PsiMat[:,0]
GroundState = GroundState/np.sqrt(h)

# Write restulst to screen
print(f'Minimal energy: {EnergyMin:.4f}')
print(f'Actual ground state energy: {GroundStateEnergy:.4f}')
Error = EnergyMin-GroundStateEnergy
print(f'Error: {Error:.4f}')
RelError = Error/np.abs(GroundStateEnergy)
print(f'Relative error: {RelError*100:2.2f} %')


# Plot energy expectation value
plt.figure(1)
plt.clf()
plt.plot(SigmaVector, EnergyVector, 'k-', label = '<E($\sigma$)>')
plt.axhline(y = GroundStateEnergy, color = 'r', 
            linestyle = '--', label = 'Actual energy')
plt.xlabel('$\sigma$')
plt.legend()
plt.grid()
plt.show()

# Plot wave functions 
# - both the one that minimizes E and the real one
plt.figure(2)
plt.clf()
# Variational
PsiVar = PsiFunk(x, SigmaForMinimum)
plt.plot(x, np.abs(PsiVar)**2, 'k-', label = 'Variatonal')
# Actual
plt.plot(x, np.abs(GroundState)**2, 'r--', label = 'Actual')
# Potential
ScaleFactor = np.max(np.abs(PsiVar)**2)
ScaleFactor = 0.2*ScaleFactor/np.abs(V0)
plt.plot(x, ScaleFactor*Vpot(x), 'b', label = 'Potential')
# Cosmetics
plt.xlabel('x', fontsize = 12)
plt.legend()
plt.grid()
plt.show()
