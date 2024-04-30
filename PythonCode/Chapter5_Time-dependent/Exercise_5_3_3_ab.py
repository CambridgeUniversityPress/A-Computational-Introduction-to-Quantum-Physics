"""
This script solve the Schrödinger equation for a 1D model of an atom exposed 
to a linearly polarized laser pulse in the dipole approximation. It does so 
by directly propagating the wave packet on a numerical grid by means of a 
Magnus propagator.

The purely time-dependent laser pulse is modelled as a sin^2-type envelope 
times a sine-carrier.

The interaction lasts a while beyond the interaction with the pulse - so 
that we can disinguish between the bound and the unbound part.

Finally, the momentum distribution of the unbpund part is determined.

Inputs for the confining potential
w     -   the width of the potential
V0    -   the "height" of the potential (should be negative)
s     -   "smoothness" parameter

Inputs for the laser pulse
Ncycl  -   the number of optical cycles
Tafter -   time propagation after pulse 
omega  -   the central frequency of the laser
E0     -   the strength of the pulse

Numerical input parameters
dt    - numerical time step
N     - number of grid points, should be 2^n
L     - the size of the numerical domain it extends from -L/2 to L/2
Xbeyond - the limit beyod which we assume the particle is liberated
yMax  - the maximal y-value in the display of the wave packet
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

# Numerical input parameters
dt = .1
N = 1024
L = 400
yMax = 5e-3
Xbeyond = 7

# Numerical time parameters
Tafter = 25
dt = 0.5


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Laser field
Tpulse=Ncycl*2*np.pi/omega
def Epulse(t) :
    return (t > 0)*(t < Tpulse)*E0*np.sin(np.pi*t/Tpulse)**2*np.sin(omega*t)

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

# Add potential and determine time-independent Hamiltonian H0
# Time-independent Hamiltonian
Ham0 = Tmat_FFT + np.diag(Vpot(x))
# Half propagator for H0
Uhalf = linalg.expm(-1j*Ham0*dt/2)

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
Tvector = np.arange(0, Tpulse+Tafter, dt)
# First plot: Laser pulse (with progagation progress)
ax1.plot(Tvector, Epulse(Tvector), color = 'black')
ax1.set_xlabel('Time [a.u.]')
ax1.set_ylabel('E(t) [a.u.]')
line1, = ax1.plot([t], [Epulse(t)], '*', color = 'red')
# Second plot: Wave packet
ax2.set(ylim=(0, yMax))                # Fix window
ax2.set_xlabel('x [a.u.]')
ax2.set_ylabel(r'$|\Psi(x)|^2$ [a.u.]')
line2, = ax2.plot(x, np.abs(Psi)**2, '-', color='black')

# Iterate while time t is less than final time Tfinal
while t < Tpulse:
  # Update time
  t = t+dt              
  # Half step with time-independent propagator
  Psi = np.matmul(Uhalf, Psi)
  # Dynamical part of the propagator
  DynamicalProp = np.diag(np.exp(-1j*x*Epulse(t+dt/2)*dt))
  Psi = np.matmul(DynamicalProp, Psi)
  # Half step with time-independent propagator
  Psi = np.matmul(Uhalf, Psi)
  
  # Update data for plots
  line1.set_xdata([t])
  line1.set_ydata([Epulse(t)])
  line2.set_ydata(np.power(np.abs(Psi), 2))
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)

# Propagate after pulse
Ufull = np.matmul(Uhalf, Uhalf)
while t < Tpulse + Tafter:
  # Update time
  t = t+dt              
  # Full step with time-independent propagator
  Psi = np.matmul(Ufull, Psi)
  # Update data for plots
  line1.set_xdata([t])
  line2.set_ydata(np.power(np.abs(Psi), 2))
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
# Estimate ionization probability
# Change format of arrays
Psi = np.asarray(Psi.T)
Pbeyond = np.trapz((np.abs(x) > Xbeyond)*np.abs(Psi)**2, dx = h)
PionPercent = 100*float(Pbeyond)
# Print result to screen
print(f'Ionization probability: {PionPercent:.2f} ')

# Filter out bound part
PsiUnbound = (np.abs(x) > Xbeyond)*Psi
# Fourier transform
PhiUnbound = np.fft.fft(PsiUnbound)
# Ensure normalization
PhiUnbound = PhiUnbound*h/np.sqrt(2*np.pi)
IndSort = np.argsort(k)
kSort = k[IndSort]
PhiUnboundSort = PhiUnbound[0, IndSort]
plt.figure(2)
plt.clf()
plt.plot(kSort, np.abs(PhiUnboundSort)**2, '-', color = 'black', 
        linewidth=2)
plt.xlabel('Momentum [a.u.]', fontsize = 12)
plt.ylabel('Probability density', fontsize = 12)
plt.grid()

# Indicate peaks according to photon number
for n in range(1,4):
  kPhoton = np.sqrt(2*(n*omega + Evector[0]))
  plt.axvline(x = kPhoton, color = 'red', linestyle = '--')
  plt.axvline(x =-kPhoton, color = 'red', linestyle = '--')
plt.xlim(-1.5*kPhoton, 1.5*kPhoton)
plt.show()  