"""
This script solve the Schrödinger equation for a 1D model of an atom exposed 
to a linearly polarized laser pulse in the dipole approximation. It does so 
by solving the Schrödinger equation formulated within the basis consisting 
of the eigenstates of the field-free Hamiltonian H_0 (the spectral basis).

Time-propagation is implemented via the Crank-Nicolson (CN) propagator. We 
allow  ourselves to introduce a truncation of the spectral basis. All 
eigenstates of H_0 with eigenvalues beyond some Etrunc are removed from the 
basis. This reduces the computational time considerably, but care must be 
taken to ensure that the cutoff is not to harsh.

The purely time-dependent laser pulse is modelled as a sin^2-type envelope 
times a sine-carrier. 

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

Numerical input parameters
dt    - numerical time step
N     - number of grid points, should be 2^n
L     - the size of the numerical domain it extends from -L/2 to L/2
Etrunc - the energy cutoff

All input parameters are hard coded initially.
"""

# Import numpy
import numpy as np

# Inputs for the confining potential
w = 5
V0 = -1
s = 5

# Inputs for the laser pulse
Ncycl  = 10
omega = 1.0
E0 = 0.5

# Numerical input parameters
N = 1024
L = 400
dt = 0.05
Etrunc = 7


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Potential (as function)
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1.0)

# Laser field
Tpulse=Ncycl*2*np.pi/omega;
def Epulse(t):
    return E0*np.sin(np.pi*t/Tpulse)**2*np.sin(omega*t)
            
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
# Time-independent Hamiltonian
Ham0 = Tmat + np.diag(Vpot(x))

# Diagaonalize time-independent Hamiltonian (Hermitian matrix)
Evector, PhiMat = np.linalg.eigh(Ham0)
# Normalize eigenstates and shift to columns
PhiMat = PhiMat/np.sqrt(h)
# Write ground state and number of bound states to screen:
print(f'Ground state energy: {Evector[0]:.4f}.')
Nbound = sum(np.array(Evector < 0))
print(f'The potential supports {Nbound} bound states.')

# Impose truncation
Ntrunc = sum(np.array(Evector < Etrunc))
print(f'The basis features {Ntrunc} states.')
H0trunc = np.diag(Evector[0:(Ntrunc-1)])
PhiMatTrunc = PhiMat[:, 0:(Ntrunc-1)]
# Determine interaction matrix; 
# the elements are < \phi_m | x | \phi_n>
HintMat = np.matmul( np.diag(x), PhiMatTrunc)
HintMat= np.matmul(np.conj(PhiMatTrunc.T), HintMat)
HintMat = HintMat*h
# Identity matrix with the adequate dimension (for the CN propagator)
Imat = np.identity(Ntrunc-1)
                  
# Initiate wave functon, Hamiltonian and time
Avec = np.zeros((Ntrunc-1, 1))
Avec[0] = 1
HamNext = H0trunc
t = 0
ProgOld = 0

# Loop over time
while t < Tpulse:
  # Update Field and Hamiltonian
  Ham = HamNext;
  HamNext = H0trunc + Epulse(t+dt)*HintMat;
  # Forward half-step of the CN propagatpr
  Mat = Imat - 1j*Ham*dt/2
  Avec = np.matmul(Mat, Avec)
  # Backward half-step of the CN propagator:
  Mat = Imat + 1j*HamNext*dt/2
  Mat = np.linalg.inv(Mat)
  Avec = np.matmul(Mat, Avec)
  
  # Update time
  t=t+dt;
  # Write progress sto screen
  Prog = np.floor(t/Tpulse*10);
  if Prog != ProgOld:
    print(f'Progress: {10*Prog} %')
    ProgOld = Prog;

# Estimate ionization probability
Pion = 1 - sum(np.abs(Avec[0:Nbound-1])**2)
PionPercent = 100*float(Pion)
Pgs = np.abs(Avec[0])**2
PgsPercent = 100*float(Pgs)
PexcitePercent = 100 - PionPercent - PgsPercent
# Print result to screen
print(f'Ionization probability: {PionPercent:.2f} %')
print(f'Excitaton probability: {PexcitePercent:.2f} %')
print(f'Ground state probability: {PgsPercent:.2f} %')