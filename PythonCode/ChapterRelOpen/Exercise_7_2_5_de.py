"""
 This script solves the GKLS equation for a qubit with amplitude damping. 
 The unitary part of the evolution is given by a time-independent Hamiltonian. 
 The rate at which the 1-state decays spontaneously into the 0-state is also 
 constant.
 
 The equation is solved by writing the density matrix as a column vector, in 
 which case the time derivative may be written as a matrix times the vector 
 itself.

 The inputs are
 E     - the diagonal elements of the Hamiltonian
 W     - the coupling (in the Hamiltonian), taken to be real
 Gamma     - the decay rate (amplitude damping)
 Ttotal    - the duration of the interaction
 Tsteps    - the number of numerical time steps (for plotting)

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import sympy

# Parameters for the Hamiltonian
E = 0.2
W = 1

# Decay rate
Gamma = 0.05

# Duration of the simulation
Tfinal = 75
# Number of time steps
Tsteps = 500

# Vector with time 
Tvector = np.linspace(0, Tfinal, Tsteps)

# Set up the equation - with the Hamiltonian
# Allocate
M = np.zeros((4, 4))
# Diagonal part (in the Hamiltonian)
M[1, 1] = -E;
M[2, 2] = E;
# Add the coupling (in the Hamiltonian)
M = M + W/2*np.matrix([[0, -1, 1, 0],
            [-1, 0, 0, 1], 
            [1, 0, 0, -1],
            [0, 1, -1, 0]])
# Add decay (in the Lindbladian)
M = M + 1j*Gamma/2*np.matrix([[0, 0, 0, 2],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, -2]])

# Right and side of the dyamical equation 
def RHS(t, y):
    return -1j*np.matmul(M, y)

# Initial state (a pure |0> state)
x0 = [0, 0, 0, complex(1, 0)]
x0 = np.asarray(x0)
x0 = x0.T                       # Transpose

# Solve ODE
sol = integrate.solve_ivp(RHS, (0, Tfinal), x0, vectorized = True, 
                          t_eval = Tvector, rtol = 1e-4)

# Analytical solution for Gamma = 0
AnalyticalNoDecay = 1/(E**2 + W**2) * \
    (E**2 + W**2*np.cos(0.5*np.sqrt(E**2 + W**2)*Tvector)**2)
    

# Determine the steady state, using a function from the sympy library to
# determine the reduced row echelon form of the augmented coefficient matrix
Aug = np.zeros((5,5), dtype = 'complex')
Aug[0:4, 0:4] = M
# Trace equal to one
Aug[4, 0] = 1
Aug[4, 3] = 1
Aug[4, 4] = 1
AugRREF = sympy.Matrix(Aug).rref()[0]
# Extract last column (save the last entry, which should be zero)
RhoSteady = AugRREF[0:4, 4]

# Plot the probability of remaining in the |1> state -
# along with the analytical solution for Gamma = 0 and the steady state
plt.figure(1)
plt.clf()
# Numerical solution, with decay
plt.plot(Tvector, sol.y[3, :], '-', color = 'black', linewidth = 2,
         label = 'With decay')
# Analytical solution without decay
plt.plot(Tvector, AnalyticalNoDecay, '-.', color = 'blue', linewidth = 1.5, 
         label = 'Without decay')
# Steady state solution
plt.axhline(y = RhoSteady[3], color = 'red', label = 'Steady state')
plt.grid()
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Probability', fontsize = 15)
plt.legend()
plt.show()