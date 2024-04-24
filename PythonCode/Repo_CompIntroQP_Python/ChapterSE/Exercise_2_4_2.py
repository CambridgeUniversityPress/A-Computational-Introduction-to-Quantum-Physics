"""
 This script simulates the evolution of an particle which hits a barrier. 
 It does so by simulating it both as a classical process and as quantum
 physical prociess - solving Newtons 2nd law and the Schrödinger equation,
 respectively.
 The classical initial momentum and position are taken as initial mean
 values in the quantum physical case.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation
   dt      - The step size in time

 Inputs for the initial Gaussian:
   x0      - The (mean) initial position
   p0      - The (mean) initial momentum
   sigmaP  - The momentum width of the initial, Gaussian wave packet
   tau     - The time at which the Gaussian is narrowest (spatially)
 
 Input for the barrier:
   V0      - The height of the barrier (can be negative)
   w       - The width of the barrier
   s       - Smoothness parameter

 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Inputs for the smoothly rectangular potential
V0 = -2         # Heigth
s = 5           # Smoothness
width = 2       # Width

# Inputs for the Gaussian 
x0 = -20.0
p0 = 1.0
sigmaP = 0.2
tau = 0.0

# Numerical grid parameters
L = 200
N = 512         # Should be 2**k, with k being an integer

# Numerical time parameters
Tfinal = 60
dt = 0.25

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
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
Tmat_FFT = np.fft.ifft(Tmat_FFT, axis = 0)
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-width/2))+1.0)

# Derivative of potential (midpoint rule)
def VpotDeriv(x):
    return (Vpot(x+h)-Vpot(x-h))/(2*h)

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot(x))
# Propagator
U = linalg.expm(-1j*Ham*dt)

# Set up Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
# Initial Gaussian
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)
# Initial classical values (mean values of the initial wave function)
xCl = x0
pCl = p0


# Initiate plots
plt.ion()
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot()
ax.set(ylim=(-0.05, .2))                # Fix window
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black', 
                 label = 'Analytical')
# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
line2, = ax.plot(x, 0.7*Vpot(x)*Psi0Max/np.abs(V0), '-', color='red', 
                 label = 'FD3')
line3, = ax.plot(xCl, 0, '*', color = 'blue')

# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi = Psi0
t = 0

# Iterate while time t is less than final time Tfinal
while t < Tfinal:
  # Copy classical values
  xCl_old = xCl
  pCl_old = pCl

  # Update wave function
  Psi = np.matmul(U, Psi)

  # Update classical values
  xCl = xCl_old + pCl_old*dt - 1/2*VpotDeriv(xCl_old)*dt**2
  pCl = pCl_old - VpotDeriv(xCl_old+pCl_old*dt/2)*dt

  # Update plots
  line1.set_ydata(np.power(np.abs(Psi), 2))
  line3.set_xdata([xCl])
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
    
  #Update time
  t = t+dt             
  
# Estimate reflection and transmission probabilities
# For reflection probability we multiply Psi with 0 for x<0,
# for reflection probability we multiply Psi with 0 for x>0
#
# Change format of arrays
Psi = np.asarray(Psi.T)
R = np.trapz((x < 0)*np.abs(Psi)**2, dx = h)
T = np.trapz((x > 0)*np.abs(Psi)**2, dx = h)
# Print result to screen
print(f'Transmission probability: {float(100*T):.2f} %')
print(f'Reflection probability: {float(100*R):.2f} %')

# Conclusion for the classical particle
if xCl < 0:
    print('The classical particle was reflected.')
else:
    print('The classical particle was transmitted.')
