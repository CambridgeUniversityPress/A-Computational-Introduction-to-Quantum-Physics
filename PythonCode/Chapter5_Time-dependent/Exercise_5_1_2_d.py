"""
This script simply tests our numerical solution of the Schrödinger equation 
against an exact analytical soltuion. The test case corresponds to a constant 
Hamiltonian. The initial state is a spin up-state.

The implementation solves the Schrödinger equation (TDSE) by using the SciPy 
function solve_ivp. It plots the  probability of a spin up-measurement as a 
function of time.

The inputs are
E         - the difference betweeen the diagonal elements 
of the Hamiltonian
Omega     - twice the coupling element
Tfinal    - the duration

All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Parameters for the magnetic field
E = 0.5
Omega = 1

# Duration of the simulation
Tfinal = 40

# Set up the equation the right hand side of the ODE,
# y'(t) = -i H(t) y(t), where y is the spinor and H is
# the Hamiltonian
# Static part of the Hamiltonian
H0 = np.matrix([[-E/2, 0], [0, E/2]])
# The coupling
Interaction = np.matrix([[0, 1], [1, 0]])
def RHS(t, y):
  Ham = H0 + Omega/2*Interaction
  Yderiv = np.matmul(Ham, y.reshape(2,1))
  return -1j*np.asarray(Yderiv)
  
# Initial state (must be  complex)
y0 = [complex(1,0), complex(0, 0)]
y0 = np.asarray(y0)

# ODE solver
# Fix times for the outputs from ODE solver
tVect = np.linspace(0, Tfinal, 500)
# Numerical solution of the ODE
sol = integrate.solve_ivp(RHS, (0, Tfinal), y0, t_eval = tVect, 
                          vectorized = True)

# Analytical solution
Analytical = 1/(E**2 + Omega**2)* \
    (E**2 + Omega**2*np.cos(np.sqrt(E**2+Omega**2)/2*tVect)**2)

# Plot results
plt.figure(1)
plt.clf()
plt.plot(sol.t, np.abs(sol.y[0,:])**2, '-', color='black', 
         label = 'Numerical solution')
plt.plot(tVect, Analytical, '--', color='red', 
         label = 'Analytical solution')
plt.xlabel('Time')
plt.ylabel('Spin up-probability')
plt.legend(loc = 'upper right')
plt.grid()
plt.ylim(0, 1.1)
plt.show()