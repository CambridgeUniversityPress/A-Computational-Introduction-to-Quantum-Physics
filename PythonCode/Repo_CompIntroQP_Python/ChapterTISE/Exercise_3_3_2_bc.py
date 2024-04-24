"""
This script simulates the time-evolution of the wave packet for a 
particle trapped in a harmonic potential in the special case that it is
in a Glauber state, or a Coherent state as it is also called.

The script also simulate the evolution of a classical particle with the 
initial condition corresponding.

The particle is assumed to have unit mass.

Numerical input parameters: 
Ttotal - the duration of the simulation
dt     - numerical time step
N      - number of grid points, should be 2^n
L      - the size of the numerical domain; it extends from -L/2 to L/2
Ntrunc - the number of states to used in our (truncated) basis
 
Physical input parameters:
alpha  - the complex parameter specifying the Glauber state
Kstrength   - strength of the harmonic potential
 
All input parameters are hard coded initially.
""" 

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import factorial

# Numerical time parameters:
Ttotal = 30
dt = 0.05

# Grid parameters
L = 30
N = 512              # For FFT's sake, we should have N=2^n

# Truncation of the spectra basis
Ntrunc = 100

# Strength of the Harmonic potential:
Kstrength = 1
# The complex Alpha parameter, which determines 
#the Glauber state
alpha = 1-1j

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
V = np.diag(Kstrength/2*x**2)

# Total Hamiltonian
Ham = Tmat_FFT + V


# Diagaonalize Hamiltonian (Hermitian matrix)
Evector, PsiMat = np.linalg.eigh(Ham)
# Normalize eigenstates
PsiMat = PsiMat/np.sqrt(h)
  

# Enforce real eigenvectors - with postive exctreme for positive x
for n in range(0,Ntrunc):
  # Select eigen vector for x>0
  State = PsiMat[int(N/2):N, n]
  # Find index of the maximal absolute value
  MaxInd = np.argmax(np.abs(State))
  Val = State[MaxInd]
  Phase = np.angle(Val)
  # Multiply through with the inverse phase factor
  PsiMat[:, n] = PsiMat[:, n]*np.exp(-1j*Phase)


# Construct initial condition for the wave function
Avector = np.zeros(Ntrunc)                      # Allocate
Nvector = range(0, Ntrunc)
# Coefficients
Avector = np.exp(-np.abs(alpha)**2/2)* \
    np.power(alpha, Nvector)/np.sqrt(factorial(Nvector))
Psi = np.matmul(PsiMat[:,0:Ntrunc], np.transpose(Avector))
# Classical postion and momentum
xCl = np.real(alpha)*np.sqrt(2)
pCl = np.imag(alpha)*np.sqrt(2)

# Initiate plot
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
MaxY = np.max(np.abs(Psi)**2)
ax.set(ylim=(-0.03, 1.5*MaxY))                # Fix window
line1, = ax.plot(x, np.abs(Psi)**2, '-', color='black')
plt.xlabel('Position, x', fontsize = 12)
plt.ylabel(r'$|\Psi(x; t)|^2$', fontsize = 12)

# Classical position
line2, = ax.plot(xCl, 0, 'rx', linewidth = 2)

# Loop over time
t = 0
while t < Ttotal:  
  # Update time
  t=t+dt
  
  # Update wave packet
  AvectorTime = np.exp(-1j*Evector[0:Ntrunc]*t)*Avector
  Psi = np.matmul(PsiMat[:,0:Ntrunc], np.transpose(AvectorTime))
  
  # Update classical position and momentum
  xClOld = xCl
  pClOld = pCl  
  Force = -Kstrength*xCl;
  xCl = xClOld + pClOld*dt - 1/2*Kstrength*xClOld*dt**2
  pCl = pClOld - Kstrength*(xClOld+pClOld*dt/2)*dt
  
  # Update data for plots
  line1.set_ydata(np.power(np.abs(Psi), 2))
  line2.set_xdata([xCl])
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)