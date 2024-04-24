"""
 This script simulates the evolution of a Gaussian wave packet moving
 freely in one dimension. It does so in four ways:
 Firstly, by plotting the analytically known wave function, secondly by
 estimating the numerical solution of the Schrödinger equation using three
 different approximation for the kinetic energy operator:
 1) A three point finite difference formula,
 2) A five point finite difference formula and
 3) the Fast Fourier transform.

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
L = 100
N = 256         # Should be 2**k, with k being an integer

# Numerical time parameters
Tfinal = 10
dt = 0.1

# Inputs for the Gaussian 
x0 = -20
p0 = 3
sigmaP = 0.2
tau = 5


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Set up matrix for the kinetic energy - 3 point formula
# Allocate and declare
Tmat_FD3 = np.zeros((N, N), dtype=complex)
# Endpoints
Tmat_FD3[0, 0:2] = [-2, 1]
Tmat_FD3[N-1, (N-2):N] = [1, -2]
# Interior points
for n in range (1,N-1):
  Tmat_FD3[n, [n-1, n, n+1]] = [1, -2, 1]
# Correct pre-factors
Tmat_FD3 = Tmat_FD3/h**2
Tmat_FD3 = -1/2*Tmat_FD3  

# Set up matrix for the kinetic energy - 5 point formula
# Allocate and declare
Tmat_FD5 = np.zeros((N, N), dtype=complex)
# Endpoints
Tmat_FD5[0, 0:3] = [-30, 16, -1]
Tmat_FD5[1, 0:4] = [16, -30, 16, -1]
Tmat_FD5[N-2, (N-4):N] = [-1, 16, -30, 16]
Tmat_FD5[N-1, (N-3):N] = [-1, 16, -30]
# Interior points
for n in range (2,N-2):
  Tmat_FD5[n, range(n-2,n+3)] = [-1, 16, -30, 16, -1]
# Correct pre-factors
Tmat_FD5 = Tmat_FD5/(12*h**2)
Tmat_FD5 = -1/2*Tmat_FD5  

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

# Construct propagators, uses expm from the SciPy library
U_FD3 = linalg.expm(-1j*Tmat_FD3*dt)
U_FD5 = linalg.expm(-1j*Tmat_FD5*dt)
U_FFT = linalg.expm(-1j*Tmat_FFT*dt)

# Set up Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
# Initial Gaussian
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)
# Time-dependent Gaussian (absolute value squared)
def PsiAbsDynamic(x, t): 
    return np.sqrt(2/np.pi)*sigmaP/np.sqrt(1+4*sigmaP**4*(t-tau)**2)\
    *np.exp(-2*sigmaP**2*(x-x0-p0*t)**2/(1+4*sigmaP**4*(t-tau)**2))

# Initiate plots
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
ax.set(ylim=(0, .2))
# Analytical wave function
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black', 
                 label = 'Analytical')
# Three point finite difference approximation
line2, = ax.plot(x, np.abs(Psi0)**2, ':', color='red', label = 'FD3')
# Five point finite difference approximation
line3, = ax.plot(x, np.abs(Psi0)**2, '-.', color='blue', label= 'FD5')
# FFT approximation
line4, = ax.plot(x, np.abs(Psi0)**2, '--', color='green', label= 'FFT')
ax.legend()

# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi_FD3 = Psi0
Psi_FD5 = Psi0
Psi_FFT = Psi0
t = 0

# Iterate while time t is less than final time Tfinal
while t < Tfinal:
  # Update analytical wave function
  PsiAbsAnalytic = PsiAbsDynamic(x,t)
  
  # Update numerical wave functions
  Psi_FD3 = np.matmul(U_FD3, Psi_FD3)
  Psi_FD5 = np.matmul(U_FD5, Psi_FD5)
  Psi_FFT = np.matmul(U_FFT, Psi_FFT)
  
  # Update data for plots
  line1.set_ydata(PsiAbsAnalytic)
  line2.set_ydata(np.power(np.abs(Psi_FD3), 2))
  line3.set_ydata(np.power(np.abs(Psi_FD5), 2))
  line4.set_ydata(np.power(np.abs(Psi_FFT), 2))
  
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
  #Update time
  t = t+dt             
