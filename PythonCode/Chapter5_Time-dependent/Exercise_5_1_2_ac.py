"""This script simulates a spin 1/2-particle which is exposed to a magnetic 
field. This field has both  a static part and a dynamic part. The dynamic part 
oscillates in time as a sine function. The static field points in the 
z-direction, thus lifting the degenerecy between spin up and spin down. The 
oscillating field is taken to point along the x-axis, corresponding to a real 
coupling in the Hamiltonian.

The initial state is a spin up-state.

The implementation solves the Schrödinger equation (TDSE) by using the 
SciPy function solve_ivp. It plots the  probability of a spin up-measurement 
as a function of time.

The inputs are
E     - the energy separation induced by the static field
Omega - the strength of the oscillating field
w     - the angular frequency of the oscillating field
OpticalCycles - the duration of the simulation, given by the number of 
periods of the oscillating field.

All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Parameters for the magnetic field
E = 1
Omega = 0.1
w = 1.1

# Duration of the simulation
OpticalCycles = 15
Tfinal = OpticalCycles*2*np.pi/w

# Set up the equation the right hand side of the ODE,
# y'(t) = -i H(t) y(t), where y is the spinor and H is
# the Hamiltonian
# Static part of the Hamiltonian
H0 = np.array([[-E/2, 0], [0, E/2]])
# The coupling
Interaction = np.array([[0, 1], [1, 0]])
def RHS(t, y):
  Ham = H0 + Omega*np.sin(w*t)*Interaction
  Yderiv = np.matmul(Ham, y)
  return -1j*Yderiv
  
# Initial state (must be  complex)
y0 = np.array([1, 0], dtype = complex)

# ODE solver
# In order to fix the time points for the output
tVect = np.linspace(0, Tfinal, 500)
# Numerical solution of the ODE
sol = integrate.solve_ivp(RHS, (0, Tfinal), y0, t_eval = tVect, 
                          vectorized = True)

# Plot result
plt.figure(1)
plt.clf()
plt.plot(sol.t, np.abs(sol.y[0,:])**2, '-', color='black')
plt.grid(visible = True)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Spin up-probability', fontsize = 12)
plt.ylim(0, 1.1)
plt.show()
