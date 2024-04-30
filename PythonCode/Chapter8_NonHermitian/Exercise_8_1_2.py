"""
 This script solve the Schrödinger equation for a 1D model of an atom exposed 
 to a linearly polarized laser pulse in the dipole approximation. It does so 
 by directly propagating the wave packet on a numerical grid by means of a 
 Magnus propagator approximated as a spilt operator.

 The purely time-dependent laser pulse is modelled as a sin^2-type envelope 
 times a sine-carrier. 

 The numerical domaine is truncated by imposing an absorber close to the 
 boundaries. The ammount of absorption is monitored in time and used to 
 estimate the total ionization probability in the end. A squre monomial is 
 used for the absoprtion potential.

 All parameters are given in atomic units.

 Inputs for the confining potential
 w     -   the width of the potential
 V0    -   the "height" of the potential (should be negative)
 s     -   "smoothness" parameter

 Inputs for the laser pulse 
 Ncycl  -   the number of optical cycles
 Tafter -   time propagation after pulse 
 omega  -   the central frequency of the laser
 E0     -   the strength of the pulse

 Inputs for the absorbing potential
 eta   - the absoprtion strength
 Onset - for |x| beyond this value, the absorbin potential is supported

 Numerical input parameters
 dt    - numerical time step
 N     - number of grid points, should be 2^n
 L     - the size of the numerical domain; it extends from -L/2 to L/2
 yMax    - the maximal y-value in the display of the wave packet
 Tafter - additional time of simulation - after the pulse is over

 All input parameters are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Inputs for the confining potential
w = 5
V0 = -1
s = 5

# Inputs for the laser pulse
Ncycl  = 10
omega = 1.0
E0 = 0.5

# Inputs for the absorbing potential
eta = 1e-2
Onset = 30

# Numerical input parameters
dt = 0.1
N = 256
L = 100
yMax = 5e-3

# Numerical time parameters
Tafter = 100
dt = 0.5


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Laser field
Tpulse=Ncycl*2*np.pi/omega;            
def Epulse(t) :
    return (t < Tpulse)*E0*np.sin(np.pi*t/Tpulse)**2*np.sin(omega*t)

# Absorbing potential
def Vabs(x):
    return (np.abs(x) > Onset) * eta * (np.abs(x)-Onset)**2

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

# Add potential and determine time-independent Hamiltonian H0
# Time-independent, Hermitian Hamiltonian
Ham0 = Tmat + np.diag(Vpot(x))
# Time-independent, non-Hermitian Hamiltonian
Ham0Eff = Ham0 - 1j*np.diag(Vabs(x))
# Half propagator for H0
Uhalf = linalg.expm(-1j*Ham0Eff*dt/2)

# Diagaonalize time-independent Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham0)
# Normalize eigenstates
PsiMat = PsiMat/np.sqrt(h)
# Write ground state and number of bound states to screen:
print(f'Ground state energy: {Evector[0]:.4f}.')
nbound = sum(np.array(Evector < 0))
print(f'The potential supports {nbound} bound states.')

# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi = PsiMat[:, 0]
Psi = Psi.reshape(N, 1)
t = 0

# Initiate plots
fig, (ax1, ax2) = plt.subplots(2, 1, num=1)

# First plot: Laser pulse (with progagation progress)
#Tvector = np.linspace(0, Tpulse+Tafter, 500)
Tvector = np.arange(0, Tpulse+Tafter, dt)
ax1.plot(Tvector, Epulse(Tvector), color = 'black')
line1, = ax1.plot([t], [Epulse(t)], '*', color = 'red')

# Second plot: Wave packet and absorber
ax2.set(ylim=(0, yMax))                # Fix window
line2, = ax2.plot(x, np.abs(Psi)**2, '-', color='black')
ScalingFactor = 0.7*yMax/np.max(Vabs(x))
line3, = ax2.plot(x, ScalingFactor*Vabs(x), '--', color = 'red')


# Initate norm vector and index
index = 0
NormVector = np.zeros(len(Tvector))

# Iterate while time t is less than final time Tfinal
while t < Tpulse:
  # Half step with time-independent propagator
  Psi = np.matmul(Uhalf, Psi)
  # Dynamical part of the propagator
  DynamicalProp = np.diag(np.exp(-1j*x*Epulse(t+dt/2)*dt))
  Psi = np.matmul(DynamicalProp, Psi)
  # Half step with time-independent propagator
  Psi = np.matmul(Uhalf, Psi)
  
  # Calculate norm
  Norm = np.trapz(np.abs(Psi.T)**2, dx = h)
  NormVector[index] = Norm
  
  # Update data for plots
  line1.set_xdata([t])
  line1.set_ydata([Epulse(t)])
  line2.set_ydata(np.abs(Psi)**2)
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
  # Update time and index
  t = t+dt              
  index = index + 1  


# Propagate after pulse
Ufull = np.matmul(Uhalf, Uhalf)
while t < Tpulse + Tafter:
  # Full step with time-independent propagator
  Psi = np.matmul(Ufull, Psi)
  
  # Calculate norm
  Norm = np.trapz(np.abs(Psi.T)**2, dx = h)
  NormVector[index] = Norm

  # Update data for plots
  line1.set_xdata([t])
  line2.set_ydata(np.power(np.abs(Psi), 2))
  
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
  # Update time and index
  t = t+dt              
  index = index + 1
  
# Estimate ionization probability
Pion = 100*(1-Norm)
Pion = float(Pion)
print(f'Estimated ionization probability: {Pion:2.2f} %')

# Plot norm as a function of time
plt.figure(2)
plt.clf()
plt.plot(Tvector, NormVector, '-', color = 'black', linewidth = 2)
plt.grid()
plt.xlabel('Time [a.u.]', fontsize = 12)
plt.ylabel('Probability', fontsize = 12)
plt.show()
