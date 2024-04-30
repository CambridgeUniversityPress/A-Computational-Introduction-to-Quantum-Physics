"""
 This script simulates a system consisting of two spin 1/2-particles exposed 
 to the same magnetic field. From the two-particle state, the reduced density 
 matrix for the first particle is determined, and various quantieis such as 
 its spin projection and purity, are determined from this reduced density 
 matrix.
 
 The initial state is one in which both particles are eigenstates of the spin 
 projection operator s_z. The first particle has positive spin projection 
 (spin up) while the other one has negative spin projection (spin down).

 The external magnetic field has both a static part and a dynamic part. The 
 dynamic part oscillates in time as a sine function. The static field points 
 in the z-direction, thus lifting the degenerecy between spin up and spin 
 down. The oscillating field is taken to point along the x-axis.

 The two particles interact thorough the spin-spin interaction.
 
 From this solution, the time-dependent reduced density matrix for the first 
 particle is calculated - at each end every time.

 The inputs are
 E     - the energy separation induced by the static field
 Omega - the strength of the oscillating field
 w     - the angular frequency of the oscillating field
 OpticalCycles - the duration of the simulation, given
 by the number of periods of the oscillating field.
 u     - the strength of the spin-spin interaction.
 
 Nsteps - the number of numerical time-steps (for plotting)
 
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
OpticalCycles = 15
Tfinal = OpticalCycles*2*np.pi/w

# Number of time-steps
Nsteps = 500

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
  return -1j*np.matmul(Ham, y)
  
# Initial state (complex column vector)
y0 = [0, complex(1, 0), 0, 0]
y0 = np.asarray(y0)
y0 = y0.T

# ODE solver
# Fix times
tVect = np.linspace(0, Tfinal, Nsteps)
# Numerical solution of the ODE (usese the option for relative tolerance)
sol = integrate.solve_ivp(RHS, (0, Tfinal), y0, t_eval = tVect, 
                          vectorized = True, rtol = 1e-4)

# Spin projection operators (Pauli matrices)
Sz = 1/2*np.array([[1, 0], [0, -1]])
Sx = 1/2*np.array([[0, 1], [1, 0]])
Sy = 1/2*np.array([[0, -1j], [1j, 0]])

# Allocate
SpinUpA = np.zeros(Nsteps)
MeanSpinA_x = np.zeros(Nsteps)
MeanSpinA_y = np.zeros(Nsteps)
MeanSpinA_z = np.zeros(Nsteps)
OffDiagonalA = np.zeros(Nsteps)
PurityA = np.zeros(Nsteps)

# Loop over the time vector T and calculate
# quantities from the reduced density matrix
for index in range(0, Nsteps):
  # Amplitudes
  a = sol.y[0, index]
  b = sol.y[1, index]
  c = sol.y[2, index]
  d = sol.y[3, index]
  # rhoA
  rhoA = np.array([[np.abs(a)**2 + np.abs(b)**2, a*np.conj(c) + b*np.conj(d)],
      [np.conj(a)*c + np.conj(b)*d, np.abs(c)**2 + np.abs(d)**2]])
  # Calculate the various quantities from rhoA
  SpinUpA[index] = np.real(rhoA[0, 0])
  MeanSpinA_x[index] = np.real(np.trace(np.matmul(Sx, rhoA)))
  MeanSpinA_y[index] = np.real(np.trace(np.matmul(Sy, rhoA)))
  MeanSpinA_z[index] = np.real(np.trace(np.matmul(Sz, rhoA)))
  OffDiagonalA[index] = np.real(np.abs(rhoA[0, 1]))
  PurityA[index] = np.real(np.trace(np.matmul(rhoA, rhoA)))

# Plot results

# Spin-up probability, size of off-diagonal elements and purity
plt.figure(1)
plt.clf()
# Spin up
plt.plot(tVect, SpinUpA, 'b-.', linewidth = 2, label = r'$P_\uparrow(t)$')
# Off-diagonal element
plt.plot(tVect, OffDiagonalA, 'k--', linewidth = 2, 
         label = r' $|\rho_{0, 1}|$')
# Purity
plt.plot(tVect, PurityA, 'r-', linewidth = 2, label = r'$\gamma(t)$')
plt.grid()
plt.xlabel('Time', fontsize = 12)
plt.legend(fontsize = 12)
plt.show()

# Plot the spin expectation values - along each axis
plt.figure(2)
plt.clf()
# X-projection
plt.plot(tVect, MeanSpinA_x, 'k-', linewidth = 2, label = r'$<s_x>$')
# X-projection
plt.plot(tVect, MeanSpinA_y, 'r--', linewidth = 2, label = r'$<s_y>$')
# X-projection
plt.plot(tVect, MeanSpinA_z, 'b-.', linewidth = 2, label = r'$<s_z>$')
plt.grid()
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Spin projection [$\hbar$]', fontsize = 12)
plt.legend(fontsize = 12)
plt.show()