"""
 This implementation investigates how amplitude damping has a  detrimental 
 effect on the implementation of the NOT gate. Within a model with degenerate 
 states with constant coupling W, the NOT gate may be implemented by tuning 
 the duration such that T = pi \hbar/W. With a finite decay rate Gamma, it 
 is no longer that simple.

 The script solves the GKLS equation for this model and determines the 
 probability of measuring 1 with 0 as the initial state. It does so for
 various values of the decay rate Gamma

 The inputs are
 W     - the coupling (in the Hamiltonian), taken to be real
 GammaMax     - the maximal decay rate (amplitude damping)
 GammaStep    - the step size for Gamma

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Coupling strength
W = 1

# Gamma parameters
GammaMax = 1.5
GammaStep = 1e-3

# Final time
Tfinal = np.pi/W

# Gamma vector
GammaVector = np.arange(0, GammaMax, GammaStep)

# Matrix with couplings
Coupl = W/2*np.matrix([[0, -1, 1, 0],
            [-1, 0, 0, 1], 
            [1, 0, 0, -1],
            [0, 1, -1, 0]])
  
# Initial state (a pure |0> state)
x0 = [complex(1, 0), 0, 0, 0]
x0 = np.asarray(x0)
x0 = x0.T                       # Transpose

# Allocate fidelity vector and index
FidelityVector = np.zeros(len(GammaVector))
index = 0

# Loop over gamma
for Gamma in GammaVector:
  # Add decay (in the Lindbladian)
  M = Coupl + 1j*Gamma/2*np.matrix([[0, 0, 0, 2],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, -2]])

  # Right and side of the dyamical equation 
  def RHS(t, y): 
      return -1j*np.matmul(M, y)

  # Solve ODE
  sol = integrate.solve_ivp(RHS, (0, Tfinal), x0, vectorized = True, 
                            rtol = 1e-4)
  
  # Extract the probability of measuring 1
  Rho11 = sol.y[3, -1]
  FidelityVector[index] = np.real(Rho11)
  index = index + 1

# Plot result
plt.figure(1)
plt.clf()
plt.plot(GammaVector, FidelityVector.T, '-', color = 'black', linewidth = 2)
plt.xlabel('Decay rate, $\Gamma$', fontsize = 12)
plt.ylabel('Flip probability', fontsize = 12)
plt.grid()
plt.axis([0, GammaMax, 0, 1])
plt.show()