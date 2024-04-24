 """
 This script is intended to check the validity of the WKB approximation when 
 it comes to estimating tunneling rates. It relates to a specific model for 
 tunneling from a metal surface towards a needle used in the STM setup. 

 The inputs are given in units which are converted to atomic units for the 
 calculations.
 
 Inputs:
 d     - The avarage distance from needle to surface
 V0    - Work function of the metal
 E     - Enery of the conductance electron 
 U     - Voltage between surface and needle
 fMin  - Minimal value the surface function
 fMax  - Maximal value for the surface function
 df    - Step size used in vector with data
 
 These input parameters are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


# Input parameters
dInAngstrom = 4
V0_in_eV = 4.5
Ein_eV = 0.5
UinV = 1
fMinInAngstrom = -1
fMaxInAngstrom = 1
dfInAngstrom = 0.01

# Atomic units
a0 = 5.292e-11           # Length in metres
E0 = 4.360e-18           # Energy in Joule
e = 1.602e-19            # Elementary charge in Coulomb

# Convert to atomic units
d = dInAngstrom*1e-10/a0
fMin = fMinInAngstrom*1e-10/a0
fMax = fMaxInAngstrom*1e-10/a0
df = dfInAngstrom*1e-10/a0
V0 = V0_in_eV*e/E0
E = Ein_eV*e/E0
eU = UinV*e/E0

# Wave number
k = np.sqrt(2*E)


# Allocate/initiate vectors
fVector = np.arange(fMin, fMax+df, df)
L = len(fVector)
TransVectorNum = np.zeros(L)
TransVectorWKB = np.zeros(L)

# The potential
def Vpot(x, a):
    return V0-a*x

# Function which calculates T for a given energy
def TransProb(f):
    
    # Set up the equation the right hand side of the ODE,
    # y'(t) = M y(t), where y is [psi, phi] and M is
    # the coefficient matrix
    D = d - f
    a = eU/D
    def RHS(x, y):
        CoeffMat = np.matrix([[0, 1], [2*Vpot(x, a)-k**2, 0]])
        return np.matmul(CoeffMat, y.reshape(2,1))
      
    # Initial state (must be  complex)
    y0 = [complex(1,0), 1j*k]
    y0 = np.asarray(y0)

    # ODE solver
    # Numerical solution of the ODE
    sol = integrate.solve_ivp(RHS, (D, 0), y0, vectorized = True)

    # Get psi(0) and phi(0)
    psi0 = sol.y[0, -1]
    phi0 = sol.y[1, -1]

    # Numerical transmission probability
    T = 4*k**2/np.abs(k*psi0-1j*phi0)**2
    return T


# Loop over f values
index = 0
for f in fVector:
  a = eU/(d-f)  
  TransVectorNum[index] = TransProb(f)
  TransVectorWKB[index] = np.exp(-4*np.sqrt(2)/(3*eU)*(d-f)*
                ((V0-E)**(3/2)-(V0-eU-E)**(3/2)))
  index = index + 1

# Plot result
plt.figure(1)
plt.clf()
plt.semilogy(fVector, TransVectorNum, '-', linewidth = 2, 
         color = 'black')
plt.semilogy(fVector, TransVectorWKB, '--', linewidth = 2, 
         color = 'red')
plt.xlabel('Energy', fontsize = 15)
plt.ylabel('Transmission Probability', fontsize = 15)
plt.grid()
plt.show()