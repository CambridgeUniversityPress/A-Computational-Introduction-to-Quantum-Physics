"""
This script simulates the time-evolution of the wave packet for a 
particle trapped in a harmonic potential. The initial state is fixed by
more or less randomly select the coefficients in a linear combination 
of theoin the eigen states of the Hamiltonian.

The particle is assumed to have unit mass.

Numerical input parameters: 
 Ttotal    - the duration of the simulation
 dt        - numerical time step, serves to tune the speed of the simulation
 N         - number of grid points, should be 2^n
 L         - the size of the numerical domain; it extends from -L/2 to L/2
 
Physical input parameters:
Avector   - the set of coefficients defining the initial state 
Kpot      - strength of the harmonic potential
 
All input parameters are hard coded initially.
""" 

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Numerical time parameters:
Ttotal = 30
dt = 0.05

# Grid parameters
L = 30
N = 512              # For FFT's sake, we should have N=2^n

# Physical parameters:
Kpot = 1
# Assign values to the first few
Avector = [4, 1, .1, 2, 1, .7, .3, .5, 2, 1]
# Number of states in our basis
Nstates = len(Avector)
# Ensure normalization
Avector = Avector/np.sqrt(np.sum(np.abs(Avector)**2))

# Set up the grid.
x = np.linspace(-L/2, L/2, N)
h = L/(N+1)

# Set up Hamiltonian
# Kinetic energy:
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
# Potential energy:
V = np.diag(Kpot/2*x**2)

# Total Hamiltonian
Ham = Tmat_FFT + V


# Diagaonalize Hamiltonian (Hermitian matrix)
EigVector, EigStates = np.linalg.eigh(Ham)
# Normalize eigenstates
EigStates = EigStates/np.sqrt(h)
# Truncate
EigVector = EigVector[0:Nstates]
EigStates = EigStates[:, 0:Nstates]

# Check and correct sign og eigen states
for n in range(0, Nstates): 
  if np.abs(np.min((x>0)*EigStates[:,n])) > \
  np.max((x>0)*EigStates[:,n]):
    EigStates[:,n] = -EigStates[:,n]


#
# Construct initial condition
#
# Wave function
Psi = np.matmul(EigStates, np.transpose(Avector))


# Initiate plot
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
MaxY = np.max(np.abs(Psi)**2)
ax.set(ylim=(0, 1.5*MaxY))                # Fix window
line1, = ax.plot(x, np.abs(Psi)**2, '-', color='black')
plt.xlabel('Position, x', fontsize = 12)
plt.ylabel(r'$|\Psi(x; t)|^2$', fontsize = 12)

# Loop over time
t = 0
while t < Ttotal:  
  # Update time
  t=t+dt
  # Update wave packet
  AvectorTime = np.exp(-1j*EigVector*t)*Avector
  Psi = np.matmul(EigStates, np.transpose(AvectorTime))
  # Update data for plots
  line1.set_ydata(np.power(np.abs(Psi), 2))
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)