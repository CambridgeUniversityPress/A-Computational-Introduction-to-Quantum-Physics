"""
 This script simulates the evolution of an intially Gaussian wave 
 passing a well. In addition to scattering, the simulation features
 the possibility of capturing the incident particle in the ground 
 state. This comes about via a jumb operator of Lindblad form.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation
   dt      - The step size in time

 Inputs for the initial Gaussian:
   x0      - The mean position of the initial wave packet
   p0      - The mean momentum of the initial wave packet
   sigmaP  - The momentum width of the initial, Gaussian wave packet
   tau     - The time at which the Gaussian is narrowest (spatially)
 
 Input for the barrier:
   V0      - The height of the barrier (can be negative)
   w       - The width of the barrier
   s       - Smoothness parameter

 Input for the capture model:
   Gamma0  - the over all strength of the decay rate

 All inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Numerical grid parameters
L = 400
N = 1024                # Should be 2^k, k integer, for FFT's sake
h = L/(N-1)

# Numerical time parameters
Tfinal = 50
dt = 0.05

# Input parameters for the Gaussian
x0 = -20
p0 = 1
sigmaP = 0.2
tau = -x0/p0

# Input parameters for the barrier
V0 = -1
w = 2
s = 5

# Capture rate
Gamma0 = 0.1

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Set up well (V0 should be negative)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1)

# Approximate kinetic energy oerator by means of the fast Fourier transform.
# Set up vector of k-values
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
# Correct pre-factor
Tmat = -1/2*Tmat    

# Full Hamiltonian
Ham = Tmat + np.diag(Vpot(x))

# Diagonalize Hamiltonian
E, B = np.linalg.eigh(Ham)
# Normalize eigenstates
B = B/np.sqrt(h)       
# Ground state
GroundState = B[:, 0]
GroundState = GroundState.reshape(N, 1)
# Write ground state energy to screen
print(f'Ground state energy: {E[0]:.4f}')
# The number of bound states
Nbound = len(np.argwhere(E<0))     
print(f'The potential supports {Nbound} bound state(s).')

# Construct anti-Hermitian contribution to effective Hamiltonian
# Vector with couplings, <\phi_l | x | \phi_0>
aux = np.matmul(np.diag(x), B[:, 0])
ProjectVect = h*np.matmul(np.conj(B.T), aux)
ProjectVect = ProjectVect.reshape(N,1)              # Column vector
# No decay from ground state
ProjectVect[0] = 0                              
# Matrix with Gamma coefficients
GammaKL = Gamma0*np.matmul(ProjectVect, np.conj(ProjectVect.T))       
# Anti-Hermitian "interaction matrix"
aux = np.matmul(GammaKL, np.conj(B.T))
HamAH = h/2*np.matmul(B, aux)

# Total, Non-Hermitian Hamiltonian:
HamTot = Ham - 1j*HamAH
# Non-unitary propagator
U = linalg.expm(-1j*HamTot*dt)                   

# Set up intial Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)

# Initiate plots
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black', linewidth = 2)
# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
ScalingFactor = 0.5*Psi0Max/np.abs(V0)
line2, = ax.plot(x, ScalingFactor*Vpot(x), '-', color='red')
plt.xlabel('x')
# Fix window
ax.set(xlim = (-50, 50), ylim=(-1.2*ScalingFactor, 1.5*Psi0Max))                

# Initiate and allocate
Psi = Psi0.reshape(N, 1)
NstepTime = int(np.floor(Tfinal/dt))
Tvec = np.arange(0, Tfinal, dt)
rho00Vec = np.zeros(len(Tvec))
rho00 = 0

# Loop which updates wave functions and plots in time
for timeIndX in range(0, NstepTime):
  # Assign population to vector
  rho00Vec[timeIndX] = rho00
  
  # Update wave function
  Psi = np.matmul(U, Psi)

  # Update norm and ground state population
  Norm = np.trapz(np.abs(Psi.T)**2, dx = h)
  Norm = float(Norm)
  # Update gound state population
  aux = np.matmul(HamAH, Psi)
  rho00 = rho00 + dt*2*h*np.matmul(np.conj(Psi.T), aux)
  rho00 = np.real(rho00)
  
  # Update plot
  line1.set_ydata(np.power(np.abs(Psi), 2) + \
                  (1-Norm)*np.power(np.abs(GroundState), 2))
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)

# Plot time evolution of the ground state population
plt.figure(2)
plt.clf()
plt.plot(Tvec, np.real(rho00Vec), '-', color = 'black', linewidth = 2)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Capture probability', fontsize = 12)
plt.grid()
plt.show()

# Write final capture probability to screen
print(f'Capture probability: {100*float(rho00):.2f} %')
