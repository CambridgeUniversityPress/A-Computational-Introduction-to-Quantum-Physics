"""
 This script estimates the ground state energy of a specific two-dimensional 
 potential by means of the variational principle. As trial function we use 
 a Gaussian wave packet with variable width mean position.

 The Hamiltonian is estimated numerically using an FFT representation of 
 the kinetic energy.

 A trial function of product form is assumed. Two functional forms are 
 set: A Gaussian and a cosine shaped one.

 The gradient descent implementation calls a function which calculates
 the energy expectation value.

 Inputs for the gradient decent approach:
 SigmaX0    - starting point for the width in the x-direction
 SigmaY0    - starting point for the width in the y-direction
 Gamma      - the so-called learning rate 
 GradientMin  - the lower limit for the length of the gradient
 DerivStep    - the numerical step size estimating the gradient

 Grid parameters
 L         - size of domain % N         
 N         - the number of grid points (should be 2^n)

 Physical inputs:
 V0        - the height of the smoothl rectangular potential (negative)
 wX        - the width of the potential in the x-direction
 wY        - the width of the potential in the y-direction
 s         - the "smoothness" of the potential 

 All parameters and functions are hard coded initially
"""

# Library
import numpy as np


# Parameters for gradeient descent
SigmaX0 = 2
SigmaY0 = 2
Gamma = 0.1
GradientMin = 1e-5
DerivStep = 1e-1

# Parameters for the smoothly rectangular part of the potential:
V0 = -1
wX = 4
wY = 2
s = 5

# Grid parameters
L = 20
N = 1024

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Test function
def PsiTest(x, sigma):
    # Gaussian version
    return (2*np.pi*sigma**2)**(-0.25) * np.exp(-(x)**2/(4*sigma**2))
#    # Cosine version 
#    # (sigma will here play the role of an inverse width)
    return np.sqrt(2*sigma/np.pi)*np.cos(sigma*x) * (np.abs(x) < np.pi/(2*sigma)) 

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
def Vpot(x, y):
    return V0/(np.exp(s*(np.abs(x)-wX/2))+1)/ \
        (np.exp(s*(np.abs(y)-wY/2))+1)
# Potential (as matrix)
xx, yy = np.meshgrid(x, x)
Vmat = Vpot(xx, yy)

# Function that calculates the energy expectation value
def EnergyExpectation(sigmaX, sigmaY):
   PsiX = PsiTest(x, sigmaX)
   PsiY = PsiTest(x, sigmaY)
   
   # Column vectors
   PsiX = PsiX.reshape(N,1)
   PsiY = PsiY.reshape(N,1)
   
   # Kinetic energies
   Tx = h*np.matmul(PsiX.T, np.matmul(Tmat_FFT, PsiX))
   Ty = h*np.matmul(PsiY.T, np.matmul(Tmat_FFT, PsiY))
   
   # Potential energy
   Psi2D = np.matmul(PsiX, np.transpose(PsiY))
   Vexpect = h**2*np.sum(np.multiply(Vmat, np.abs(Psi2D)**2))
   
   return float(np.real(Tx + Ty + Vexpect))

# 
# Gradient descent
#
# Initiate parameters for gradient descent
index = 1;
sigmaX = SigmaX0
sigmaY = SigmaY0
LengthGradient = 1e6       # Fix high value to get the loop going
# Iterate while the gradient is large enough
while LengthGradient > GradientMin:
  # Calculate new energy estimate
  EnergyEstimate = EnergyExpectation(sigmaX, sigmaY)
  
  # Estimate partial derivatives by means of midpoint rule
  dEdSigmaX = (EnergyExpectation(sigmaX+DerivStep, sigmaY) - 
               EnergyExpectation(sigmaX-DerivStep, sigmaY))/ \
                   (2*DerivStep)
  dEdSigmaY = (EnergyExpectation(sigmaX, sigmaY+DerivStep) - 
               EnergyExpectation(sigmaX, sigmaY-DerivStep))/ \
                   (2*DerivStep)
  # Calculate length of the gradient
  LengthGradient = np.sqrt(dEdSigmaX**2 + dEdSigmaY**2)
  
  # Update sigmaX and sigmaY
  sigmaX = sigmaX - Gamma*dEdSigmaX
  sigmaY = sigmaY - Gamma*dEdSigmaY
  
  # Write estimate to screen for every 10th iteration
  if np.mod(index, 10) == 0:
      print(f'Iteration: {index}, energy: {float(EnergyEstimate):.4f}')
  # Update index
  index=index+1                          

# Resulting energy estimate
EnergyEstimate = float(EnergyEstimate)

# Write results to screen:
print(f'Minmal energy: {EnergyEstimate:.4f}')
print(f'Optimal value for the x-width: {sigmaX:.4f}')
print(f'Optimal value for the y-width: {sigmaY:.4f}')
print(f'Total number of iterations: {index}')