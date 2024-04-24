"""
 This script simulates the evolution of an particle which hits a double 
 barrier. Each of the two identical parts of the barrier has a smooth 
 rectangular-like shape.
 
 The transmission probability is determined by semi-analytical means 
 where plane wave solutions are imposed in the regions where the double 
 barrier potential is no longer supported. The script loops over several
 energy values for the solution and and, in the end, plot the 
 transmission probability as a function of mean energy.
 
 It imposes a complex absorbing potential and uses the accumulated
 absorption at each end in order to estimate reflection and trasmissiion
 probabilities.

 The absorbing potential is a quadratic monomial.
 
 Input for the barrier:
   V0      - The height of the barrier (can be negative)
   w       - The width of the barrier
   s       - Smoothness parameter
   d       - Distance between the barriers (centre to centre)
   D       - The length of the interval, starting from x = 0, where the
   potential is supported. This quantity may very well be related to d 
 and  w.

 Inputs for the energy grid
   Emin    - Minmal energy 
   Emax    - Maximal energy
   dE      - Energy increment
 
 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Inputs for the couble barrier potential
V0 = 4              # Heigth
s = 25              # Smoothness
width = 0.5         # Width
d = 3               # Half the distance between barriers
D = 2*d + 4*width

# Energies to calculate
dE = 0.01
Emin = 0.01
Emax = V0

# Set up potential (single barrier)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-width/2))+1)

# Seut up double barrier
def VpotDouble(x):
    return Vpot(x-d) + Vpot(x+d)

# Shift the potential to the right
def VpotDoubleShifted(x):
    return VpotDouble(x-D/2)

# Define function for determining the transmission probabbility
# Function which calculates T for a given energy
def TransProb(E):
    # Wave number
    k = np.sqrt(2*E)

    # Set up the equation the right hand side of the ODE,
    # y'(t) = M y(t), where y is [psi, phi] and M is
    # the coefficient matrix
    def RHS(x, y):
        CoeffMat = np.matrix([[0, 1], [2*VpotDoubleShifted(x)-k**2, 0]])
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

  
# Vector with meane energies for the initial wave packet
Evector = np.arange(Emin, Emax, dE)    
# Allocate vector with transmission probabilities
Tvector = np.zeros(len(Evector))
  
# Loop over energies
index = 0
for E in Evector:
    # Determine tranmsission probability and store in vector
    Tvalue = TransProb(E)
    Tvector[index] = Tvalue
    
    # Print transmission and reflection probability result to screen
    print(f'E = {E:.3f}, T = {float(Tvalue):.4f} ')

    # Update index
    index = index + 1

# Plot transmission probabilit as a function of energy
plt.figure(1)
plt.clf()
plt.plot(Evector, Tvector, '-', color = 'black', linewidth = 2)
plt.grid()
plt.xlabel('Energy', fontsize = 12)
plt.ylabel('Probabilty', fontsize = 12)
plt.show()