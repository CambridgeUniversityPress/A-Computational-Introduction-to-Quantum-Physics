"""
 This script determines the ground state for a specific 
 potential by means of propagation in inmaginary time.
 The potential barrier has a "smoothly rectangular" shape.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation - in imaginary time
   dt      - The step size in imaginary time

 Input for the potential:
   V0      - The "height" of the barrier (must be negative)
   w       - The width of the barrier
   s       - Smoothness parameter
   Vpot    - The confining potential (function variable)

 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Numerical grid parameters
L = 20.0
N = 256         # Should be 2**k, with k being an integer

# Numerical time parameters
Tfinal = 10;
dt = 0.05;

# Inputs for the smoothly rectangular potential
V0 = -3.0            # Heigth
width = 8.0         # Width
s = 5.0             # Smoothness

# The potential
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-width/2))+1)

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Determine double derivative by means of the 
# fast Fourier transform.
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

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot(x))
# Propagator
U = linalg.expm(-Ham*dt)

# Initiate random wave functons and time
Psi0 = np.random.rand(N)
NormInit = np.trapz(np.abs(Psi0)**2, dx = h)
Psi0 = Psi0/np.sqrt(NormInit)                 
Psi0 = Psi0.reshape(N,1)

# Initiate plots
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black')
plt.xlabel('x', fontsize = 12)
plt.ylabel(r'$|\Psi|^2$', fontsize = 12)

# Initiate wave function
Psi = Psi0
# Vectors for plotting
Ntime = int(Tfinal/dt)
tVector = np.linspace(0, Tfinal, Ntime)
EnergyVector = 0*tVector

# Iterate while time t is less than final time Tfinal
index = 0
for t in tVector:
  # Update wave function
  Psi = np.matmul(U, Psi)
  # Normalize
  Norm2 = float(np.trapz(np.abs(Psi.T)**2, x))
  Psi = Psi/np.sqrt(Norm2)
  # Energy estimate
  EnergyEstimate = -0.5/dt*np.log(Norm2)
  EnergyVector[index] = EnergyEstimate
  # Update data for plots
  line1.set_ydata(np.power(np.abs(Psi), 2))
  # recompute the ax.dataLim
  ax.relim()    
  # update ax.viewLim using the new dataLim
  ax.autoscale_view()  # Update plots
  # Adjust axes
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  # Update index
  index = index + 1
  
  
# Determine actual enegy eigen state  
Evector, PsiMat = np.linalg.eigh(Ham)
# Extract ground state
GroundStateEnergy = Evector[0]

  
# Plot "evolution" of energy estimate
plt.figure(2)
plt.clf()
plt.plot(tVector, EnergyVector, label = 'Estimated energy')  
plt.axhline(y = GroundStateEnergy, color = 'r', 
            linestyle = '--', label = 'Actual energy')
plt.grid()
plt.legend()
plt.xlabel('Imaginary time', fontsize = 12)
plt.ylabel('Energy estimate', fontsize = 12)
plt.show()

# Diagonalize Hamiltonian
Evector, PsiMat = np.linalg.eigh(Ham)
# Extract ground state
GroundStateEnergy = Evector[0]
GroundState = PsiMat[:,0]
# Normalize
GroundState = PsiMat[:,0]
GroundState = GroundState/np.sqrt(h)

# Plot comparison with the ground state from direct diagonalization
plt.figure(3)
plt.clf()
plt.plot(x, np.abs(Psi)**2, 'k-', label = 'Imaginary time')
# Actual
plt.plot(x, np.abs(GroundState)**2, 'r--', label = 'Diagonalization')
# Potential
ScaleFactor = np.max(np.abs(Psi)**2)
ScaleFactor = 0.2*ScaleFactor/np.abs(V0)
plt.plot(x, ScaleFactor*Vpot(x), 'b', label = 'Potential')
# Cosmetics
plt.xlabel('x', fontsize = 12)
plt.legend()
plt.grid()
plt.show()

# Write estimate to screen
print(f'Estimated ground state energy: {EnergyEstimate:.4f}')
print(f'Actual ground state energy on the grid: {GroundStateEnergy:.4f}')