"""
 This script estimates the ground state energy for a two-particle system 
 consisting of two identical fermions. The particles interacts via a smooth 
 Coulomb-like interaction, and they are confined by a "smooth" rectangular 
 potential 
 
 We determine the ground state energy by propagation in imaginary time.
 The implementation also has the option of finding the lowest energy for 
 an exchange anti-symmetric state.
 
 The one-particle Hamiltonian is estimated numerically using an FFT-
 representation of the kinetic energy.

 
 Numerical inputs:
 L         - size of domain 
 N         - number of grid points (should be 2^n)
 dt        - step size in "time"-propagation
 Tmax      - duration of simulation
 
 Physical inputs:
 V0        - the height of the potential (should be negative)
 w         - the width of the potential
 s         - the "smoothness" of the potential 
 W0        - the strength of the interaction
 Xsymm     - the exchange symmetry, whould be +1 or -1

 All parameters and functions are hard coded initially
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt


# Numerical grid parameters
L = 10
N = 128                

# Time parameteres
dt = 1e-3
Tmax = 5

# Inputs for the smoothly rectangular potential and interaction
V0 = -1            # Height, i.e. negative debth
w = 4              # Width
s = 5              # Smootness
W0 = 1             # Interaction strength

# Exchange symmetry
Xsymm = +1

# Set up grids
# x-grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)
# Two-particle grid
X1, X2 = np.meshgrid(x, x)


# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Interaction
def Wint(x1, x2):
    return W0/np.sqrt((x1-x2)**2+1)

# Kinetic energy operator (FFT)
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Transform identity matrix
Tmat = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat = np.matmul(np.diag(-k**2), Tmat)
# Transform back to x-representation. 
Tmat = np.fft.ifft(Tmat, axis = 0)
# Correct pre-factor (unit mass and \hbar = 1)
Tmat = -1/2*Tmat    


# Full, one-particle Hamiltonian
HamOnePart = Tmat + np.diag(Vpot(x))
# Ensure real Hamiltonian
HamOnePart = np.real(HamOnePart)

# Matrix with interaction
WintMat = Wint(X1, X2)

# Initiate wave function 
Psi = np.random.rand(N, N)
# Enforce the correct symmetry
Psi = 0.5*(Psi + Xsymm*Psi.T)
NormSq = h**2*sum(sum(np.abs(Psi)**2))
Psi = Psi/np.sqrt(NormSq) 

# Initiate and allocate
t = 0
Tvector = np.arange(0, Tmax, dt)
EnergyVector = np.zeros(len(Tvector))
index = 0

for t in Tvector:
  # Hamiltonian acting on the wave function
  HamPsi = np.matmul(HamOnePart, Psi) + \
      np.matmul(Psi, HamOnePart) + np.multiply(WintMat, Psi)
  
  # Take a step in time
  Psi = Psi - HamPsi*dt
  # Enfocrce correct symmetry
  Psi = 0.5*(Psi + Xsymm*Psi.T)
  
  # Renormalize and estimate energy
  NormSq = h**2*sum(sum(np.abs(Psi)**2))
  Psi = Psi/np.sqrt(NormSq)
  Energy = 1/dt*(1-np.sqrt(NormSq))
  EnergyVector[index] = Energy
  
  # Update index
  index = index + 1
  
# Write energy to screen
print(f'Energy estimate: {Energy:.4f}') 

# Plot wave function
plt.figure(1)
plt.clf()
plt.pcolor(x, x, Psi)
plt.xlabel(r'$x_1$', fontsize = 12)
plt.ylabel(r'$x_2$', fontsize = 12)
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

# Plot energy convergence
plt.figure(2)
plt.clf()
plt.plot(Tvector, EnergyVector, color = 'black')
plt.grid()
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Enegy', fontsize = 12)
plt.ylim(-2, 2)
plt.show()