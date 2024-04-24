"""
 This script simulates the evolution of two Gaussian wave packets who meet. 
 It does so by solving the Schrödinger equation by approximating the kinetic 
 energy operator using the Fast Fourier transform. The two Gaussians have 
 their own sets of initial mean position, mean momentum, and momentum widths.
 The mean momenta should be such that the two waves travel towards each other, 
 and the mean position and widths should be such that they do not overlap 
 initially.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation
   dt      - The step size in time

 Physical inputs, wave 1:
   x01      - The mean position of the initial wave packet
   p02      - The mean momentum of the initial wave packet
   sigmaP1  - The momentum width of the initial, Gaussian wave packet
   tau1     - The time at which the Gaussian is narrowest (spatially)

 -Correspondingly for wave 2.    

 All inputs are hard coded initially.
"""   

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Numerical grid parameters
L = 200
N = 512             # Should be 2**k, with k being an integer

# Numerical time parameters
Tfinal = 50
dt = 0.1

# Inputs for 1st Gaussian 
x01 = -20
p01 = 1
sigmaP1 = 0.2
tau1 = 0

# Inputs for 2nd Gaussian 
x02 = 20
p02 = -1.5
sigmaP2 = 0.4
tau2 = 0

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
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex))
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
# Transpose necessary as we want to transform columnwise
Tmat_FFT = np.fft.ifft(np.transpose(Tmat_FFT))
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

# Construct propagator, uses expm from the SciPy library
U_FFT = linalg.expm(-1j*Tmat_FFT*dt)

# Set up Gaussian - analytically
# Gaussian 1
InitialNorm1 = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP1/(1-2j*sigmaP1**2*tau1))
Psi1 = InitialNorm1*np.exp(-sigmaP1**2*(x-x01)**2/(1-2j*sigmaP1**2*tau1)+
                           1j*p01*x)
# Gaussian 2
InitialNorm2 = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP2/(1-2j*sigmaP2**2*tau2))
Psi2 = InitialNorm2*np.exp(-sigmaP2**2*(x-x02)**2/(1-2j*sigmaP2**2*tau2)+
                           1j*p02*x)
# Normalization - note: this requires negligible overlap
Psi0 = 1.0/np.sqrt(2)*(Psi1 + Psi2)

# Initiate plots
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
ax.set(ylim=(0, .2))
# Absolute value
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black', 
                 label = r'$|\Psi|^2$')
# Real part
line2, = ax.plot(x, np.real(Psi0)**2, '--', color='blue', linewidth = 1, 
                 label = r'$(Re \; \Psi)^2$')
plt.legend()

# Initiate wave functon and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi_FFT = Psi0
t = 0

# Iterate while time t is less than final time Tfinal
while t < Tfinal:
  # Update numerical wave function
  Psi_FFT = np.matmul(U_FFT, Psi_FFT)

  # Update data for plots
  line1.set_ydata(np.power(np.abs(Psi_FFT), 2))
  line2.set_ydata(np.power(np.real(Psi_FFT), 2))
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
  #Update time
  t = t+dt             
