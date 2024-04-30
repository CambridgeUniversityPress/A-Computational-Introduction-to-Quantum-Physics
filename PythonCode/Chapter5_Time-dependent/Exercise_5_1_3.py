"""
This script simulates a spin 1/2-particle which is exposed to a magnetic 
field. It does so in two ways: 1) by direct, numerical solution of the 
Schroedinger equation (TDSE) and 2) approximatively by the analytical 
solution within the rotating wave approximation (RWA).

The magnetic field has both a static part and a dynamic part. The dynamic, 
which points along the x axis, part oscillates in time as a sine function. 
The static field points in the z-direction, thus lifting the degenerecy 
between spin up and spin down.

The initial state is a spin up-state.

The implementation solves the Schrödinger equation by using the SciPy 
function solve_ivp. It plots the probability of a spin up-measurement as 
a function of time.

The inputs are
E     - the energy separation induced by the static field
Omega - the strength of the oscillating field
w     - the angular frequency of the oscillating field
OpticalCycles - the duration of the simulation, given
by the number of periods of the oscillating field.

All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Parameters for the magnetic field
E = 1
Omega = 0.25
w = 1.1

# Duration of the simulation
OpticalCycles = 15
Tfinal = OpticalCycles*2*np.pi/w

# Set up the equation the right hand side of the ODE,
# y'(t) = -i H(t) y(t), where y is the spinor and H is
# the Hamiltonian
# Static part of the Hamiltonian
H0 = np.matrix([[-E/2, 0], [0, E/2]])
# The coupling
Interaction = np.matrix([[0, 1], [1, 0]])
def RHS(t, y):
  Ham = H0 + Omega*np.sin(w*t)*Interaction
  Yderiv = np.matmul(Ham, y.reshape(2,1))
  return -1j*np.asarray(Yderiv)
  
# Initial state (must be  complex)
y0 = [complex(1,0), complex(0, 0)]
y0 = np.asarray(y0)

# ODE solver
# Fix times for output
tVect = np.linspace(0, Tfinal, 500)
# Numerical solution of the ODE
sol = integrate.solve_ivp(RHS, (0, Tfinal), y0, t_eval = tVect, 
                          vectorized = True)

# Analytical, approximate solution (RWA)
delta = w - E
SpinUpProbRWA = 1/(delta**2 + Omega**2) * (delta**2 + Omega**2 *
                    np.cos(np.sqrt(delta**2 + Omega**2)*tVect/2)**2 )

# Plot results
plt.figure(1)
plt.clf()
# TDSE solution
plt.plot(sol.t, np.abs(sol.y[0,:])**2, '-', color='black', 
         label = 'TDSE')
# RWA solution
plt.plot(tVect, SpinUpProbRWA, '--', color='red', 
         label = 'RWA')
plt.grid()
plt.ylim(0, 1.1)
plt.legend(loc = 'upper right')
plt.xlabel('Time')
plt.ylabel('Spin up-probability')
plt.show()