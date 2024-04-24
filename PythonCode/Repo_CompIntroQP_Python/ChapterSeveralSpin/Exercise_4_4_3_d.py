"""
 This script estimates the ground state energy for a two-particle system in 
 one dimensions. The trial function is a symmetrized product of Gaussian 
 single-particle functions with mean positions of opposite signs.
 Specifically: 
 Psi ~ 
 \psi(x_1; +mu) \psi(x_2; -mu) + \psi(x_1; -mu) \psi(x_2; +mu)

 The Hamiltonian is estimated numerically using an FFT representation of 
 the kinetic energy. It features an interaction term which is a "softened" 
 Coulomb interaction.

 The ground state energy is estimated using the variational principle taking 
 the width sigma and the mean position mu of the Gaussians as variational 
 parameters.

 The actual minimization of the energy expectation value is done using the 
 gradient descent method.
 
 Inputs for the gradient decent approach:
 Sigma0    - starting point for the width
 Mu0       - starting point for the mean position
 Gamma     - the so-called learning rate
 GradientMin  - the lower limit for the length of the gradient
 DerivStep    - the numerical step size for numerical integration

 Grid parameters
 L         - size of domain 
 N         - number of grid points (should be 2^n)

 Physical inputs:
 V0        - the height of the smoothl rectangular potential (negative)
 w         - the width of the potential
 s         - the "smoothness" of the potential 
 W0        - the strength of the two-particle interaction

 All parameters and functions are hard coded initially
"""

# Libraries
import numpy as np

# Parameters for gradeient descent
Sigma0 = 2
Mu0 = 2
Gamma = 0.05
GradientMin = 1e-3
DerivStep = 1e-3

# Parameters for the smoothly rectangular part of the potential:
V0 = -1
w = 4
s = 5
# Interaction strength
W0 = 1

# Grid parameters
L = 20
N = 1024

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Functions
#
# Potential
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1)
# Single particle, Gaussian function
def PsiTest(x, sigma, mu):
    return (2*np.pi*sigma**2)**(-0.25)* \
        np.exp(-(x - mu)**2/(4*sigma**2))
# Interatcion
def Wint(x1, x2):
    return W0/np.sqrt((x1-x2)**2 + 1)

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

# One-particle Hamiltonian (including one-particle potential)
H_OnePart = Tmat_FFT + np.diag(Vpot(x))

# Interaction as matrix
xx, yy = np.meshgrid(x, x)
IntMat = Wint(xx, yy)

# Function that calculates the energy expectation value
def EnergyExpectation(sigma, mu):
   PsiPlus  = PsiTest(x, sigma, +mu)
   PsiMinus = PsiTest(x, sigma, -mu)
   
   # Column vectors
   PsiPlus  = PsiPlus.reshape(N,1)
   PsiMinus = PsiMinus.reshape(N,1)
   
   # Overlap
   Overlap = h*np.matmul(np.transpose(np.conj(PsiPlus)), PsiMinus)
   Overlap = float(np.real(Overlap))
   
   # 1st one-particle energy
   OnePart1 = h*np.matmul(np.transpose(np.conj(PsiPlus)), 
                          np.matmul(H_OnePart, PsiPlus))
   OnePart1 = float(np.real(OnePart1))
   
   # 2nd one-particle energy
   OnePart2 = h*np.matmul(np.transpose(np.conj(PsiMinus)), 
                          np.matmul(H_OnePart, PsiPlus))
   OnePart2 = OnePart2*Overlap
   OnePart2 = float(np.real(OnePart2))
   
   # Two-particle terms
   PsiPlusMinus = np.matmul(PsiPlus, PsiMinus.T)
   PsiMinusPlus = np.matmul(PsiMinus, PsiPlus.T)
   # 1st two-particle energy
   TwoPart1 = h**2*sum(sum(np.conj(PsiPlusMinus)*IntMat*PsiPlusMinus))
   TwoPart1 = float(np.real(TwoPart1))
   # 2nd two-particle energy
   TwoPart2 = h**2*sum(sum(np.conj(PsiPlusMinus)*IntMat*PsiMinusPlus))
   TwoPart2 = float(np.real(TwoPart2))
   
   # Total energy - symmetric version
   return (2*OnePart1 + 2*OnePart2 + TwoPart1 + TwoPart2)/(1 + Overlap**2)
   # Total energy - anti-symmetric version
   #return (2*OnePart1 - 2*OnePart2 + TwoPart1 - TwoPart2)/(1 - Overlap**2)

# 
# Gradient descent
#
# Initiate parameters for gradient descent
index = 1;
sigma = Sigma0
mu = Mu0
LengthGradient = 1e6       # Fix high value to get the loop going

# Iterate while the gradient is large enough
while LengthGradient > GradientMin:
  # Calculate new energy estimate
  EnergyEstimate = EnergyExpectation(sigma, mu)
  
  # Estimate partial derivatives by means of midpoint rule
  dEdSigma = (EnergyExpectation(sigma+DerivStep, mu) - 
               EnergyExpectation(sigma-DerivStep, mu))/ \
                   (2*DerivStep)
  dEdMu   =  (EnergyExpectation(sigma, mu+DerivStep) - 
               EnergyExpectation(sigma, mu-DerivStep))/ \
                   (2*DerivStep)
  
  # Calculate length of the gradient
  LengthGradient = np.sqrt(dEdSigma**2 + dEdMu**2)
  
  # Update sigmaX and sigmaY
  sigma = sigma - Gamma*dEdSigma
  mu = mu - Gamma*dEdMu
  
  # Write estimate to screen for every 10th iteration
  if np.mod(index, 10) == 0:
      print(f'Iteration: {index}, energy: {float(EnergyEstimate):.6f}')
  # Update index
  index=index+1                          

# Resulting energy estimate
EnergyEstimate = float(EnergyEstimate)

# Write results to screen:
print(f'Minmal energy: {EnergyEstimate:.5f}')
print(f'Optimal value for sigma: {sigma:.4f}')
print(f'Optimal value for mu: {mu:.4f}')
print(f'Total number of iterations: {index}')