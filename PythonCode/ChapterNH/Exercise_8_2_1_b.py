"""
 This script simulates the evolution of an particle which hits a double 
 barrier. Each of the two identical parts of the barrier has a smooth 
 rectangular-like shape.
 
 The initial wave is a Gaussian with a finite width in momentum and 
 poistion. It is implemented such that you can run it repeatedly with 
 several initial mean momenta - or, equivalently, initial energies - and, 
 in the end, plot the transmission probability as a function of mean 
 energy.
 
 It imposes a complex absorbing potential and uses the accumulated
 absorption at each end in order to estimate reflection and trasmissiion
 probabilities.

 The absorbing potential is a quadratic monomial.
 
 Inputs for the initial Gaussian:
   x0      - The (mean) initial position
   sigmaP  - The momentum width of the initial, Gaussian wave packet
   tau     - The time at which the Gaussian is narrowest (spatially)
   E0min   - The minimal initial enwergy
   E0max   - The maximal initial energy
   dE0     - The energy step size
 
 Input for the barrier:
   V0      - The height of the barrier (can be negative)
   w       - The width of the barrier
   s       - Smoothness parameter
   d       - Distance between the barriers (centre to centre)

 Inputs for the absorbing potential
   eta     - The strength of the absorber
   Onset   - the |x| value beyond which absorption starts
 
 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   dt      - The step size in time

 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

# Numerical grid parameters
L = 100
N = 512         

# Numerical time parameter
dt = 0.25

# Inputs for the couble barrier potential
V0 = 4              # Heigth
s = 25              # Smoothness
width = 0.5         # Width
d = 3               # Half the distance between barriers

# Energies to calculate
dE0 = 0.05
E0min = 0.1
E0max = V0

# Inputs for the Gaussian 
x0 = -20
sigmaP = 0.1
tau = 0

# Inputs for the absorber
eta = 0.05
Onset = 40


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Set up potential (single barrier)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-width/2))+1)

# Seut up double barrier
def VpotDouble(x):
    return Vpot(x-d) + Vpot(x+d)

# Set up the absorbing potential
def Vabs(x):
    return  eta * (np.abs(x) > Onset) * (np.abs(x) - Onset)**2

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

# Full Hermitian Hamiltonian
Ham = Tmat + np.diag(VpotDouble(x))
# Add absorber
Ham = Ham - 1j*np.diag(Vabs(x))
# Propagator
U = linalg.expm(-1j*Ham*dt)



# Define function for determining the transmission probabbility
def TransmissionProb(E):
    # Initial momentum
    p0 = np.sqrt(2*E)
    
    # Initiate wave functon
    # Set up Gaussian - analytically
    InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
    # Initial Gaussian
    Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2 / 
                              (1-2j*sigmaP**2*tau)+1j*p0*x)
    # Turn Psi0 into N \times 1 matrix (column vector)
    Psi0 = np.matrix(Psi0)                 
    Psi0 = Psi0.reshape(N,1)
    Psi = Psi0

    # Initiate reflection and transmission probabilities
    Rprob = 0
    Tprob = 0

    # Loop which updates wave functions and plots in time
    # The limit of 99.5 # is set somewhat arbitrarily; it could be 
    # higher and it could be slightly lower
    while Rprob + Tprob < .995:
        # Update wave function
        Psi = np.matmul(U, Psi)
        
        # Update R and T
        # Reshape Psi for integral
        PsiForInt = np.asarray(Psi.T)
        Rprob = Rprob + \
            2*dt*np.trapz((x < -Onset)*Vabs(x)*np.abs(PsiForInt)**2, dx = h)
        Tprob = Tprob + \
            2*dt*np.trapz((x >  Onset)*Vabs(x)*np.abs(PsiForInt)**2, dx = h)
  
    # Assign output
    return Tprob

  
# Vector with meane energies for the initial wave packet
E0vector = np.arange(E0min, E0max, dE0)    
# Allocate vector with transmission probabilities
Tvector = np.zeros(len(E0vector))
  
# Loop over energies
index = 0
for E0 in E0vector:
    # Determine tranmsission probability and store in vector
    Tvalue = TransmissionProb(E0)
    Tvector[index] = Tvalue
    
    # Print transmission and reflection probability result to screen
    print(f'E = {E0:.3f}, T = {float(Tvalue):.4f} ')

    # Update index
    index = index + 1

# Plot transmission probabilit as a function of energy
plt.figure(1)
plt.clf()
plt.plot(E0vector, Tvector, '-', color = 'black', linewidth = 2)
plt.grid()
plt.xlabel('Energy', fontsize = 12)
plt.ylabel('Probabilty', fontsize = 12)
plt.show()