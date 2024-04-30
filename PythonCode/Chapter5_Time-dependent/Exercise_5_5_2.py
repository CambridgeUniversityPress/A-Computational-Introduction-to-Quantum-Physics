"""
 This script calculates the probability to remain in the the 
 initial state for a particle in a harmonic osciallator which
 is shaken. It does so by first expressing the kinetic energy term
 via FFT and then approximating the second order Magnus propagator as 
 a split operator.
 
 The harmonic oscillator potential is subject to a time-dependent 
 translation which shifts it in both directions before restoring it 
 in its original position. In addition to time, it takes omega as an 
 input; the duraton of the time-dependent shift is related to omega by 
 T = 2*pi/omega.

 Inputs
 L         - size of domain 
 N         - number of grid points (should be 2^n)
 Nt        - the number of time steps for each run
 k         - the strength of the potential
 omegaMin  - minimal omega value
 omegaMax  - maximal omega value
 domega    - step size in omega
 InitialState - the index of the initial state;  InitialState = 0 is 
 the ground state

 All inputs are hard coded initially
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg


# Grid parameters
L = 10
N = 512              

# Strength of the harmonic oscillator
Kpot = 1

# Time parameters
Nstep = 500
omegaMin = 0.05
omegaMax = 1.5
domega = 0.025

# Index of initial state (ground state corresponds to zero)
InitialState = 0

# Shape of the potential
def Vpot(x):
    return 0.5*Kpot*x**2

# The displacement of the potential
def Ftrans(t, omega): 
    return 8/3**(3/2)*np.sin(omega*t/2)**2*np.sin(omega*t)

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Kinetic energy operator (FFT)
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
# Correct pre-factor (unit mass and \hbar = 1)
Tmat = -1/2*Tmat    

# Add potential
# Full Hamiltonian
Ham = Tmat + np.diag(Vpot(x))

# Diagaonalize time-independent Hamiltonian (Hermitian matrix)
Evector, Bmat = np.linalg.eigh(Ham)
# Normalize eigenstates
Bmat = Bmat/np.sqrt(h)

# InitialState
Psi0 = Bmat[:, InitialState]

# Loop over the omega values
omegaVector = np.arange(omegaMin, omegaMax+domega/10, domega)
Pvector = 0*omegaVector          # Allocate    
index = 0
for omega in omegaVector:
  # Duration and time-step
  tTotal = 2*np.pi/omega
  dt = tTotal/Nstep
  # Propagator for half-step with kinetic energy
  UkinHalf = linalg.expm(-1j*Tmat*dt/2)
  t = 0
  # Initial state
  Psi = Psi0
 
  # Do the dynamics
  while t < tTotal:
     # Half step with kinetic energy
     Psi = np.matmul(UkinHalf, Psi)  
     # Full step with poetntial energy
     Translation = Ftrans(t+dt/2, omega)
     UpotFull = np.diag(np.exp(-1j*Vpot(x-Translation)*dt))
     Psi = np.matmul(UpotFull, Psi)
     # Another half step with kinetic energy
     Psi = np.matmul(UkinHalf, Psi)
     # Update time
     t = t + dt
  
  # Probability for remaining in the initial state
  Pinit = np.abs(np.trapz(np.conj(Psi)*Psi0, x))**2
  # Write result to screen
  print(f'omega: {omega:.4f}, Pinit: {Pinit:.4f}')
  Pvector[index] = Pinit
  index = index + 1

# Plot Pinit as a function of omega
plt.figure(1)
plt.clf()
plt.plot(omegaVector, Pvector, '-', linewidth = 2, color = 'black')
plt.xlabel(r'$\omega$', fontsize = 12)
plt.ylabel(r'$P_{init}$', fontsize = 12)
plt.grid()
plt.show()