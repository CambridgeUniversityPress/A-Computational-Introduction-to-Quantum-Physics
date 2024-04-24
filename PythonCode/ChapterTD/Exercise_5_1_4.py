"""
This script simulates a system consisting of two spin-1/2 particles 
exposed to the same magnetic field. This field has both a static part 
and a dynamic part. The dynamic part oscillates in time as a sine function.
The static field points in the z-direction, thus lifting the degenerecy 
between spin up and spin down. The oscillating field is taken to point 
along the x-axis.

The two particles interact thorough the spin-spin interaction.

The implementation solves the Schrödinger equation (TDSE) by using the 
solve_ivp from SciPy. It plots the probability of both spins pointing 
upwards and the probability of opposite alignment along the z-axis as 
functions of time.

Several choices of initial state are listed; the desired one is selected by 
commenting out the other ones.

The inputs are
E     - the energy separation induced by the static field
Omega - the strength of the oscillating field
w     - the angular frequency of the oscillating field
OpticalCycles - the duration of the simulation, given
by the number of periods of the oscillating field.
u     - the strength of the spin-spin interaction.
 
All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Parameters for the magnetic field
E = 1
Omega = 0.5
w = 1.1

# Strength of the spin-spin interaction
u = 0.025

# Duration of the simulation
OpticalCycles = 10
Tfinal = OpticalCycles*2*np.pi/w

# Set up the equation the right hand side of the ODE,
# y'(t) = -i H(t) y(t), where y is the spinor and H is
# the Hamiltonian
# Static part of the Hamiltonian
H0 = np.matrix([[-E+u, 0, 0 , 0], 
                [0, -u, 2*u, 0],
                [0, 2*u, -u, 0],
                [0, 0, 0, E+u]])
# The coupling
Interaction = np.matrix([[0, 1, 1, 0], 
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]])
def RHS(t, y):
  Ham = H0 + Omega*np.sin(w*t)*Interaction
  Yderiv = np.matmul(Ham, y.reshape(4,1))
  return -1j*np.asarray(Yderiv)
  
# Initial state (must be  complex)
y0 = [complex(1,0), complex(0, 0), complex(0, 0), complex(0, 0)]
#y0 = [complex(0,0), complex(1, 0), complex(1, 0), complex(0, 0)]/np.sqrt(2)
#y0 = [complex(0,0), complex(1, 0), complex(-1, 0), complex(0, 0)]/np.sqrt(2)
#y0 = [complex(0,0), complex(1, 0), complex(0, 0), complex(0, 0)]
#
y0 = np.asarray(y0)

# ODE solver
# Fix times
tVect = np.linspace(0, Tfinal, 500)
# Numerical solution of the ODE
sol = integrate.solve_ivp(RHS, (0, Tfinal), y0, t_eval = tVect, 
                          vectorized = True)

# Plot result
plt.figure(1)
plt.clf()
# Plot up-up probability
plt.plot(tVect, np.abs(sol.y[0,:])**2, '-', color= 'black', label = 'Up-up')
# Plot probability for oppsite z-alignment
plt.plot(tVect, np.abs(sol.y[1,:])**2 + np.abs(sol.y[2,:])**2, 
         '-', color='red', label = 'Opposite')
# Plot up-down probability
plt.plot(tVect, np.abs(sol.y[1,:])**2, 
         '-', color='blue', label = 'Up-down')
plt.grid()
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Probability', fontsize = 12)
plt.legend()
plt.show()

# Check for symmetry breaking
plt.figure(2)
plt.clf()
plt.plot(tVect, np.abs(sol.y[1,:]-sol.y[2,:]), color = 'black')
plt.grid()
plt.xlabel('Time', fontsize = 12)
plt.ylabel('|b(t)-c(t)|', fontsize = 12)
plt.show()