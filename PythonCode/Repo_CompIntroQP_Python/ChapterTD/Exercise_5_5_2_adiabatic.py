"""
 This script calculates the probability to remain in the the 
 initial state for a particle in a harmonic osciallater which
 is shaken. It does so by solving the Schrödinger equation formulated
 within the adiabatic basis. The time-stepping is performed
 using a propagator of Crank-Nicolson form.
 
 The harmonic oscillated potential is subject to a 
 time-dependent translation which shifts it in both directions
 before restoring it in its original position. The dynamics is
 resolved repeatedly with varying duration.

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
 InitialState - the index of the initial state; 
 InitialState = 0 is the ground state

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
omegaMax = 0.5
domega = 0.025

# Index of initial state (ground state corresponds to zero)
InitialState = 0

# Shape of the potential
def Vpot(x):
    return 0.5*Kpot*x**2

# The displacement of the potential
def Ftrans(t, omega): 
    return 8/3**(3/2)*np.sin(omega*t/2)**2*np.sin(omega*t)

# The derivative of the displacement of the potential
def FtransDeriv(t, omega):
    return (t<2*np.pi/omega)*4/3**(3/2)*omega* \
    (2*np.sin(omega*t/2)**2*np.cos(omega*t) + np.sin(omega*t)**2)


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Kinetic energy operator (FFT)
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Transform identity matrix
Itrans = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2 and k for T and P, respectively
Tmat = np.matmul(np.diag(-k**2), Itrans)
Pmat = np.matmul(np.diag(k), Itrans)
# Transform back to x-representation. 
Tmat = np.fft.ifft(Tmat, axis = 0)
Pmat = np.fft.ifft(Pmat, axis = 0)
# Correct pre-factor (unit mass and \hbar = 1)
Tmat = -1/2*Tmat    

# Add potential
# Full Hamiltonian
Ham = Tmat + np.diag(Vpot(x))

# Diagaonalize time-independent Hamiltonian (Hermitian matrix)
Evector, Bmat = np.linalg.eigh(Ham)
# Normalize eigenstates
Bmat = Bmat/np.sqrt(h)

# Digonal matrix with energies
Dmat = np.diag(Evector)
# Coupling matrix
aux = np.matmul(Pmat, Bmat)
CoupMat = h*np.matmul(np.conj(Bmat.T), aux)

# InitialState
Avec0 = np.zeros((N, 1))
Avec0[InitialState] = 1
Imat = np.identity(N) 

# Loop over the omega values
omegaVector = np.arange(omegaMin, omegaMax+domega/10, domega)
Pvector = 0*omegaVector          # Allocate    
index = 0
for omega in omegaVector:
  # Duration and time-step
  tTotal = 2*np.pi/omega
  dt = tTotal/Nstep
  t = 0
  Fderiv = 0
  Ham = Dmat
  Avec = Avec0
  # Do the dynamics
  while t < tTotal:
    # Update Hamiltonian and the derivative of the displament
    FderivOld = Fderiv;
    HamOld = Ham;
    Fderiv = FtransDeriv(t+dt, omega);
    Ham = Dmat - Fderiv*CoupMat;
    
    # The Crank-Nicolson scheme
    Avec = np.matmul((Imat - 1j*HamOld*dt/2), Avec)
    InvStep = np.linalg.inv(Imat + 1j*Ham*dt/2) 
    Avec = np.matmul(InvStep, Avec)
    
    # Update time
    t = t + dt
  # Probability for remaining in the initial state
  Pinit = np.abs(Avec[InitialState])**2
  Pinit = float(Pinit)
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