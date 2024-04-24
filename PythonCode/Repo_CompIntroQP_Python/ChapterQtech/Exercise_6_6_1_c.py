""" 
 This script aims to find the minimum of a function by introducing this
 function as the potential of a Hamiltion for which the system
 undergoes adiabatic evolution. The system starts out 
 in the ground state of a harmonic oscillator. Then, the potential
 is gradually changed into another one - the one we wish to minimize. 
 If this is done sufficiently slowly, the system will remain in the 
 current ground state.
 
 Next, in order to localize the minimum, we allow for the mass of the
 quantum particle to increase - rendering a ground state with a much
 smaller width centred near the global minimum.

 In both these phases, the evolution is simulated using split operator 
 techniques. The potential and the kinetic energy change roles when it comes 
 to the splitting in the two phases.

 Time inputs
 Nstep   - the number time steps in the first phase
 Tphase1 - the duration of first phase
 Tphase2 - the duration of the second phase, the one with increasing mass

 Numerical input parameters
 N     - number of grid points, should be 2^n for FFT's sake
 L     - the size of the numerical domain it extends from -L/2 to L/2
 
 Function inputs
 Vinit     - intial, harmonic potential
 Vfinal    - potential to be minimized
 sFunk     - time-dependent function shiftling smootly from 1 to 0
 mFunk     - the time-dependent mass - for phase 2

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
Tphase1 = 20
Tphase2 = 100

# Window for plotting
xMin = -4
xMax = 4
yMin = -0.5
yMax = 4

# Initial potential
def Vinit(x):
    return 0.5*x**2

# Final potential - the one we want to minimize
def Vfinal(x):
    return (x**2-1)**2 - x/5

# Transition functions
# To change potential
def sFunk(t): 
    return 0.5*(1 + np.cos(np.pi/Tphase1*t))
# To change mass
def mFunk(t):
    return 1 + 0.01*t**2


# Time-dependent potential, phase 1
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
dt = Tphase1/Nstep

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
line1, = ax.plot(x, np.abs(Psi)**2, '-', color='black')
# Plotting the potential
line2, = ax.plot(x, Vinit(x), '-', color='blue')
plt.grid()
plt.xlabel('Position')

# Half propagator for kinetic energy
UT_half = linalg.expm(-1j*Tmat*dt/2)

# Propagate phase 1
t = 0
while t < Tphase1:
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
  fig.canvas.flush_events()

  # Update time
  t = t+dt


# Entering pahse 2, where the potential stays constant but the mass,
# and thus also the kinetic energy, changes in time

# Half propagator for potential
Upot_half = np.diag(np.exp(-1j*Vfinal(x)*dt/2))

# Propagate
while t < Tphase1 + Tphase2:
  # Half potential energy
  Psi = np.matmul(Upot_half, Psi)              
  # Kinetic energy propagator
  Mass = mFunk(t-Tphase1)
  UT = np.diag(np.exp(-1j*dt*(k**2)/(2*Mass)))
  Phi = np.fft.fft(Psi, axis = 0)
  Phi = np.matmul(UT, Phi)
  Psi = np.fft.ifft(Phi, axis = 0)
  # Half potential energy - again
  Psi = np.matmul(Upot_half, Psi)
  
  # Update data for plots
  line1.set_ydata(np.abs(Psi)**2)
  # Update plots
  fig.canvas.draw()
  fig.canvas.flush_events()
  
  # Update time
  t = t+dt