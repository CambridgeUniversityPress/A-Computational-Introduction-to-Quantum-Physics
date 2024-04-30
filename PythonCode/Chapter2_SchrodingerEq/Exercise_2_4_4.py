"""
 This script simulates the evolution of an intially Gaussian wave packet 
 placed between two barriers. The barriers have a "smoothly rectangular" 
 shape. In addition to simulating the evolution of the wave packet, the 
 script estimates the probability of remaining between the barriers.

 As we go aloing, the evolution of the wave packet is shown - with a 
 logarithmic y-axis.
 
 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation
   dt      - The step size in time

 Inputs for the initial Gaussian:
   sigmaP  - The momentum width of the initial, Gaussian wave packet
 
 Input for the barriers:
   V0      - The height of the barriers (can be negative)
   w       - The width of the barriers
   s       - Smoothness parameter
   d       - Distance between barriers (centre to centre)

 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Numerical grid parameters
L = 200
N = 2048         # Should be 2**k, with k being an integer

# Numerical time parameters
Tfinal = 150
dt = 0.25

# Inputs for the smoothly rectangular potential
V0 = 1              # Heigth
s = 25              # Smoothness
width = 0.5         # Width
d = 10              # Half the distance between barriers

# Input for the Gaussian 
sigmaP = 0.2


# Fixed parameters for the Gaussian
x0 = 0
p0 = 0
tau = 0
# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Determine double derivative by means of the fast Fourier transform.
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

# Potential
# Single
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-width/2))+1.0)
# Double
def VpotDouble(x):
    return Vpot(x-d) + Vpot(x+d)

# Full Hamiltonian
Ham = Tmat + np.diag(VpotDouble(x))
# Propagator
U = linalg.expm(-1j*Ham*dt)

# Set up Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
# Initial Gaussian
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)

# Initiate plots
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
ax.set(ylim=(1e-4, 1))                # Fix window
line1, = ax.semilogy(x, np.abs(Psi0)**2, '-', color='black')
# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
line2, = ax.semilogy(x, 0.5*VpotDouble(x)*Psi0Max/V0, '-', color='red')
plt.xlabel('x')
plt.ylabel(r'$|\Psi(x)|^2$')

# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi = Psi0
t = 0
index = 0
# Allocate vector with probability to remain
Ntimes = int(np.floor(Tfinal/dt))
PbetweenVector = np.zeros(Ntimes)
# Time vector
TimeVector = np.arange(0, Tfinal, dt)

# Iterate while time t is less than final time Tfinal
while t < Tfinal:
  # Population between barriers
  Pbetween = np.trapz((np.abs(x) < d)  * np.asarray(np.abs(Psi.T))**2, 
                      dx = h)
  PbetweenVector[index] = float(Pbetween)
  
  # Update wave function, time og index
  Psi = np.matmul(U, Psi)
  t = t + dt
  index = index + 1
  
  # Update plot
  line1.set_ydata(np.power(np.abs(Psi), 2))
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
# Plot probability to remain  
plt.figure(2)
plt.clf()
plt.plot(TimeVector, PbetweenVector, '-', color = 'black', linewidth = 2)
plt.grid()
plt.xlabel('Time', fontsize = 15)
plt.ylabel(r'$P_{between}$', fontsize = 15)
plt.show()