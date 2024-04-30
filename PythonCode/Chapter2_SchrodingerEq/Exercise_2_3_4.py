"""
 This script calculates position and momentum expectation values and widths 
 for a Gaussian wave packet moving freely in one dimension. It plots these 
 four quantities after the simulation is over - in addition to the product of 
 the position and momentum widths. This is done in order to check that the 
 Heisenberg uncertainty relation holds for this wave packet.

 The numerical derivatives, which are involved both in solving the Schrödinger 
 equation and in calculating momentum expectation values and widhts, are 
 approximated by means of the Fast Fourier Transform.

 The time-dependent expectation value for x and the position width are also 
 calculated analytically and compared with the numerical solution.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation
   dt      - The step size in time

 Physical inputs:
   x0      - The mean position of the initial wave packet
   p0      - The mean momentum of the initial wave packet
   sigmaP  - The momentum width of the initial, Gaussian wave packet
   tau     - The time at which the Gaussian is narrowest (spatially)
 
 All inputs are hard coded initially.
"""   

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg


# Numerical grid parameters
L = 200.0
N = 512         # Should be 2**k, with k being an integer

# Numerical time parameters
Tfinal = 20;
dt = 0.25;

# Inputs for the Gaussian 
x0 = -50.0
p0 = 3.0
sigmaP = 0.2
tau = 5.0


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Diagonal matrices with x and x^2
# (Far from optimal implementation, but consistent)
Xmat = np.diag(x)
X2mat = np.diag(x**2)

# Determine double derivative by means of the fast Fourier transform.
# Set up vector of k-values
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Transform identity matrix
Aux = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by ik and (ik)^2, respectively
D1_FFT = np.matmul(np.diag(1j*k), Aux)
D2_FFT = np.matmul(np.diag(-k**2), Aux)
# Transform back to x-representation. 
D1_FFT = np.fft.ifft(D1_FFT, axis = 0)
D2_FFT = np.fft.ifft(D2_FFT, axis = 0)

# Construct propagator, uses expm from the SciPy library
U_FFT = linalg.expm(-1j*(-1/2)*D2_FFT*dt)

# Set up Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
# Initial Gaussian
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)

# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi = Psi0
t = 0

# Allocate vectors for making plots
Npoints = int(np.ceil(Tfinal/dt))
Tvector = np.zeros(Npoints)
MeanXvector = np.zeros(Npoints)
MeanPvector = np.zeros(Npoints)
WidthXvector = np.zeros(Npoints)
WidthPvector = np.zeros(Npoints)

# Iterate while time t is less than final time Tfinal
counter = 0;
#while t < Tfinal:
while counter < Npoints:
  # Calculate expectations values by means of matrix multiplication
  # <x> and <x^2> 
  Xmean = h*np.real(np.matmul(Psi.H, np.matmul(Xmat, Psi)))
  X2mean = h*np.real(np.matmul(Psi.H, np.matmul(X2mat, Psi)))
  # <p> and <p^2>
  Pmean = h*np.real(np.matmul(Psi.H, np.matmul(-1j*D1_FFT, Psi)))
  P2mean = h*np.real(np.matmul(Psi.H, np.matmul(-D2_FFT, Psi)))
  
  # Store mean values and widths in vectors
  Tvector[counter] = t
  MeanXvector[counter] = Xmean
  WidthXvector[counter] = np.sqrt(X2mean-Xmean**2) 
  MeanPvector[counter] = Pmean
  WidthPvector[counter] = np.sqrt(P2mean-Pmean**2) 
  
  #Update time
  t = t+dt;             
  counter = counter + 1
  # Update numerical wave functions
  Psi = np.matmul(U_FFT, Psi)

# Make plots
# Mean position
plt.figure(1)
plt.clf()
plt.plot(Tvector, MeanXvector, '-', color='black', label = 'Numerical')
# Analytical expression
AnalyticalX0 = x0 + p0*Tvector
plt.plot(Tvector, AnalyticalX0, '--', color='red', label = 'Analytical')
plt.xlabel('Time')
plt.ylabel(r'$< x >$')
plt.legend()
plt.grid()
plt.show()

# Position width
plt.figure(2)
plt.clf()
plt.plot(Tvector, WidthXvector, '-', color='black', label = 'Numerical')
# Analytical expression
AnalyticalSigmaX = np.sqrt(1+4*sigmaP**4*(Tvector-tau)**2)/(2*sigmaP)
plt.plot(Tvector, AnalyticalSigmaX, '--', color='red', label = 'Analytical')
plt.xlabel('Time')
plt.ylabel(r'$\sigma_x$')
plt.legend()
plt.grid()
plt.show()

# Mean momentum
plt.figure(3)
plt.clf()
plt.plot(Tvector, MeanPvector, '-', color='black')
plt.xlabel('Time')
plt.ylabel(r'$< p >$')
plt.grid()
plt.show()

# Momentum width
plt.figure(4)
plt.clf()
plt.plot(Tvector, WidthPvector, '-', color='black')
plt.xlabel('Time')
plt.ylabel(r'$\sigma_p$')
plt.grid()
plt.show()

# Product of widths
plt.figure(5)
plt.clf()
plt.plot(Tvector, WidthXvector*WidthPvector, '-', color='black')
plt.axhline(y = 0.5, linestyle = '--', color = 'red')
plt.axvline(x = tau, linestyle = ':', color = 'blue')
plt.xlabel('Time')
plt.ylabel(r'$\sigma_x \cdot \sigma_p$')
plt.grid()
plt.show()