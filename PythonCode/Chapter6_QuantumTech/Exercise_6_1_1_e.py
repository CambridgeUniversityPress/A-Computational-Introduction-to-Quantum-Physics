"""
 This script determines the tunneling rate for a particle of unit mass 
 hitting a barrier supported in a finite interval.
 It solves the time-independent Schrödinger equation in the region where 
 the potential is supported by formulating it as a first order coupled ODE 
 in \psi(x) and \phi(x), where \phi(x) = \psi'(x). 

 In this particular case, the potential is constant. This allows us to test 
 our implementation by comparing it to an analytical, exact solution.

 Inputs:
 V0 - The height of the potential
 D  - The width of the interval in which the potential 
 is nonzero
 E  - The energy of the particle, must be positive
 
 These input parameters are hard coded initially.
"""

# Libraries
import numpy as np
from scipy import integrate

# Inputs
V0 = 3
D = 2
E = 1.75

# Wave number
k = np.sqrt(2*E)

# Set up the equation the right hand side of the ODE,
# y'(t) = M y(t), where y is [psi, phi] and M is
# the coefficient matrix
CoeffMat = np.matrix([[0, 1], [2*V0-k**2, 0]])
def RHS(x, y):
  return np.matmul(CoeffMat, y.reshape(2,1))
  
# Initial state (must be  complex)
y0 = [complex(1,0), 1j*k]
y0 = np.asarray(y0)

# ODE solver
# Numerical solution of the ODE
sol = integrate.solve_ivp(RHS, (D, 0), y0, vectorized = True)

# Get psi(0) and phi(0), the last entry in their respective vectors
psi0 = sol.y[0, -1]
phi0 = sol.y[1, -1]

# Numerical transmission probability
Tnumerical = 4*k**2/np.abs(k*psi0-1j*phi0)**2

# Analytical expression for T
if E < V0:
    alpha = np.sqrt(2*(V0-E))
    Tanalytical = 1/(1 + V0**2/(4*E*(V0-E))*np.sinh(alpha*D)**2)
else:
    alpha = np.sqrt(2*(E-V0))
    Tanalytical = 1/(1 + V0**2/(4*E*(E-V0))*np.sin(alpha*D)**2)

# Write results to screen
print(f'Numerical tunneling rate:  {100*Tnumerical:.3} %.')
print(f'Analytical tunneling rate: {100*Tanalytical:.3} %.')