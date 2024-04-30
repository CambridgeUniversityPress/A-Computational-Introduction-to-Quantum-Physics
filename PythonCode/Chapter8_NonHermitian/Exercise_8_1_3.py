"""
 This script simulates the evolution of an particle which hits a barrier.
 The barrier has a smoth rectangular-like shape.
 
 It imposes a complex absorbing potential and uses the accumulated
 absorption at each and in order to estimate reflection and trasmissiion
 probabilities.

 The absorbing potential is a quadratic monomial.

 The simulation lasts until the probability of having a particle on the grid 
 falls below 0.5%. (This limit can be changed.)
 
 Inputs for the initial Gaussian:
   x0      - The (mean) initial position
   p0      - The (mean) initial momentum
   sigmaP  - The momentum width of the initial, Gaussian wave packet
   tau     - The time at which the Gaussian is narrowest (spatially)
 
 Input for the barrier:
   V0      - The height of the barrier (can be negative)
   w       - The width of the barrier
   s       - Smoothness parameter

 Inputs for the absorbing potential
   eta     - The strength of the absorber
   Onset   - the |x| value beyond which absorption starts
 
 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   dt      - The step size in time

 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Numerical grid parameters
L = 200
N = 1024         

# Numerical time parameter
dt = 0.25

# Inputs for the smoothly rectangular potential
V0 = 3              # Heigth
s = 5               # Smoothness
width = 2           # Width

# Inputs for the Gaussian 
x0 = -20
p0 = 1
sigmaP = 0.2
tau = 0

# Inputs for the absorber
eta = 1e-2
Onset = 80


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Set up potential
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-width/2))+1)

# Set up the absorbing potential
def Vabs(x):
    return  eta * (np.abs(x) > Onset) * (np.abs(x) - Onset)**2

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

# Full Hermitian Hamiltonian
Ham = Tmat + np.diag(Vpot(x))
# Add absorber
Ham = Ham - 1j*np.diag(Vabs(x))
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
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black')
# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
line2, = ax.plot(x, 0.7*Vpot(x)*Psi0Max/V0, '-', color='blue')
# Plott absorber
MaxAbs = np.max(Vabs(x))
line2, = ax.plot(x, 0.5*Vabs(x)*Psi0Max/MaxAbs, '--', color='red')
plt.grid()
plt.xlabel('x')

# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi = Psi0
t = 0

# Initiate reflection and transmission probabilities
Rprob = 0
Tprob = 0

# Loop which updates wave functions and plots in time
# The limit of 99.5 # is set somewhat arbitrarily; it could be 
# higher and it could be slightly lower
while Rprob + Tprob < 0.995:
  # Update wave function
  Psi = np.matmul(U, Psi)
  
  # Update R and T
  # Reshape Psi for integral
  PsiForInt = np.asarray(Psi.T)
  Rprob = Rprob + \
  2*dt*np.trapz((x < -Onset)*Vabs(x)*np.abs(PsiForInt)**2, dx = h)
  Tprob = Tprob + \
  2*dt*np.trapz((x >  Onset)*Vabs(x)*np.abs(PsiForInt)**2, dx = h)
  
  # Update plot
  line1.set_ydata(np.power(np.abs(Psi), 2))
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
  # Update time
  t = t+dt
  
# Print transmission and reflection probability result to screen
print(f'Reflection probability: {float(100*Rprob):.2f} %')
print(f'Transmission probability: {float(100*Tprob):.2f} %')
