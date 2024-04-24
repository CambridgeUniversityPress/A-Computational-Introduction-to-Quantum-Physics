"""
 This script determines the tunneling rate for a particle of unit mass 
 hitting a potential supported in a finite interval.
 It solves the time-independent Schrödinger equation in the region where 
 the potential is supported by formulating it as a first order coupled ODE in 
 \psi(x) and \phi(x), where \phi(x) = \psi'(x). It does so for a set of 
 several energy values.

 Inputs:
 Vpo   - The potential this input is a function variable- one
 that features certain additional input parameters
 D     - The width of the interval in which the potential 
 is nonzero
 Emin  - The minimal energy of the particle, must be positive
 Emax  - The maximal energy
 dE    - The increment of the vector containing input energies
 
 These input parameters are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Input parameters
D = 3
Emin = 0.1
Emax = 4
dE = 0.05

# The potential
V0 = 2
a = 0.4
def Vpot(x):
    return V0-a*x


# Function which calculates T for a given energy
def TransProb(E):
    # Wave number
    k = np.sqrt(2*E)

    # Set up the equation the right hand side of the ODE,
    # y'(t) = M y(t), where y is [psi, phi] and M is
    # the coefficient matrix
    def RHS(x, y):
        CoeffMat = np.matrix([[0, 1], [2*Vpot(x)-k**2, 0]])
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


# Allocate/initiate vectors
EnergyVector = np.arange(Emin, Emax+dE, dE)
L = len(EnergyVector)
TransVector = np.zeros(L)

index = 0
for Energy in EnergyVector:
  TransVector[index] = TransProb(Energy)
  index = index + 1

# Plot result
plt.figure(1)
plt.clf()
# With linear axes
plt.plot(EnergyVector, TransVector, '-', linewidth = 2, 
         color = 'black')
# With logarithmic y-axis
#plt.semilogy(EnergyVector, TransVector, '-', linewidth = 2, 
#         color = 'black')
plt.xlabel('Energy', fontsize = 12)
plt.ylabel('Transmission Probability', fontsize = 12)
plt.grid()
plt.show()