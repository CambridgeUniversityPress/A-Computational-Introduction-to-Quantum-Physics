""" 
 This script aims at finding the ground state of a Hamiltonian of a 
 certain potential by means of adiabatic evolution. The system starts out 
 in the ground state of a harmonic oscillator. Then, the potential
 is gradually changed into another one - the one we wish to minimize. If 
 this is done sufficiently slowly, the system will remain in the current 
 ground state.

 The evolution is simulated using the split operator technique.

 Time inputs
 Nstep  -   the number time steps
 Ttotal -   the duration of interaction

 Numerical input parameters
 N     - number of grid points, should be 2^n for FFT's sake
 L     - the size of the numerical domain; it extends from -L/2 to L/2
 
 Function inputs
 Vinit     - intial, harmonic potential
 Vfinal    - potential to be minimized
 sFunk     - time-dependent function shiftling smootly from 1 to 0

 Vpot, not input: sFunk(t)*Vinit(x) + (1-sFunk(t))*Vfinal(x)

 All input parameters are hard coded initially.
 
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Spatial inputs
N = 128
L = 8

# Time inputs
Nstep  =  250
Ttotal = 20

# Window for plotting
xMin = -4
xMax = 4
yMin = -0.5
yMax = 1.5

# Initial potential
def Vinit(x):
    return 0.5*x**2

# Final potential - the one we want to minimize
def Vfinal(x):
    return (x**2-1)**2 - x/5

# Transition functions
# To change potential
def sFunk(t): 
    return 0.5*(1 + np.cos(np.pi/Ttotal*t))


# Time-dependent potential
def Pot(t, x):
    return sFunk(t)*Vinit(x) + (1-sFunk(t))*Vfinal(x)

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)


# Kinetic energy by means of the fast Fourier transform.
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

# Time step
dt = Ttotal/Nstep

# Initiate time
t=0

# Initial state, ground state of the harmonic oscillator with k=1
Psi = np.pi**(-.25)*np.exp(-x**2/2)      
Psi = Psi.reshape(N, 1)                        

# Initiate plots
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax))     # Fix window
# Plot wave function
line1, = ax.plot(x, np.abs(Psi)**2, '-', color='black')
# Plot the potential
line2, = ax.plot(x, Vinit(x), '-', color='blue')
plt.grid()
plt.xlabel('Position')

# Half propagator for kinetic energy
UT_half = linalg.expm(-1j*Tmat*dt/2)

# Propagate
t = 0
while t < Ttotal:
  # Half kinetic energy
  Psi = np.matmul(UT_half, Psi)              
  # Potential energy propagator
  Upot_full = np.diag(np.exp(-1j*Pot(t,x)*dt))
  Psi = np.matmul(Upot_full, Psi)
  # Half kinetic energy
  Psi = np.matmul(UT_half, Psi)              
  
  # Update data for plots
  line1.set_ydata(np.abs(Psi)**2)
  line2.set_ydata(Pot(t, x))
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  
  # Update time
  t = t+dt