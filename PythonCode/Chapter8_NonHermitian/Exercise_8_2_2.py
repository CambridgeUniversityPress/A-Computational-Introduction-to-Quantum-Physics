"""
 This script plots the determinant of a coefficient matrix of a set 
 of equations which determines the resonance energies for a potential 
 consisting of two rectangular barriers. 

 Inputs are
 For the barriers:
 V0 - The height of each barier (it is the same for both)
 d - half the distance between the barriers
 w - the width of each barrier.

 For the energy ranges:
 ImEMax - the maximal absolute value of negative imaginary energy
 dE - resolution in the enegy grid it is same for the real and the 
 imaginary part.
"""
# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Physical input parameters
V0 = 4     
d = 3     
w = .5     

# Energy grid
ReEMax = V0
ImEMax = .25            # Assign the absolute value of maximally negative
dE = .01

# Vectors of real and imaginary energy
ReEvector = np.arange(0, ReEMax, dE)
ImEvector = np.arange(-ImEMax, 0, dE)

# Alloacte
DetMatSymm = np.zeros((len(ReEvector), len(ImEvector)))
DetMatAntiSymm = np.zeros((len(ReEvector), len(ImEvector)))

# Define parmeters, onset and offset of the barriers
dPlus = d + w/2
dMinus = d - w/2


# Construct matrices with values of determinants
# Symmetric case
ReInd=0
for realE in ReEvector:
  ImInd = 0
  for imagE in ImEvector:
    # Energy-dependent parameters
    E = realE + 1j*imagE            # Complex energy
    kappa = np.sqrt(2*(V0-E))       
    k = np.sqrt(2*E)
    
    # Coefficient matrix
    Mat=[[np.cos(k*dMinus), -np.exp(-kappa*dMinus), -np.exp(kappa*dMinus), 0],
    [-k*np.sin(k*dMinus), kappa*np.exp(-kappa*dMinus), 
                             -kappa*np.exp(kappa*dMinus), 0],
    [0, np.exp(-kappa*dPlus), np.exp(kappa*dPlus), -np.exp(1j*k*dPlus)],
    [0, -kappa*np.exp(-kappa*dPlus), kappa*np.exp(kappa*dPlus), 
                             -1j*k*np.exp(1j*k*dPlus)]]
    
    # Calculate and assign determinant (in absolute value)
    DetMatSymm[ReInd,ImInd]=np.abs(np.linalg.det(Mat))
    ImInd = ImInd + 1              # Update index
  ReInd = ReInd + 1                # Update index


# Construct matrices with values of determinants
# Antisymmetric case
ReInd=0
for realE in ReEvector:
  ImInd = 0
  for imagE in ImEvector:
    # Energy-dependent parameters
    E = realE + 1j*imagE            # Complex energy
    kappa = np.sqrt(2*(V0-E))       
    k = np.sqrt(2*E)
    
    # Coefficient matrix
    Mat=[[np.sin(k*dMinus), -np.exp(-kappa*dMinus), -np.exp(kappa*dMinus), 0],
    [k*np.cos(k*dMinus), kappa*np.exp(-kappa*dMinus), 
                             -kappa*np.exp(kappa*dMinus), 0],
    [0, np.exp(-kappa*dPlus), np.exp(kappa*dPlus), -np.exp(1j*k*dPlus)],
    [0, -kappa*np.exp(-kappa*dPlus), kappa*np.exp(kappa*dPlus), 
                             -1j*k*np.exp(1j*k*dPlus)]]
    
    # Calculate and assign determinant (in absolute value)
    DetMatAntiSymm[ReInd,ImInd]=np.abs(np.linalg.det(Mat))
    ImInd = ImInd + 1              # Update index
  ReInd = ReInd + 1                # Update index


# Plot determinants - for both symmetric and antisymmetric case
fig, (ax1, ax2) = plt.subplots(2, 1)
# Symmetric case
ax1.pcolor(ReEvector, ImEvector, np.log(DetMatSymm.T))
ax1.set_xlabel(r'Re  $\varepsilon$')
ax1.set_ylabel(r'Im  $\varepsilon$')
# Anti-symmetric case
ax2.pcolor(ReEvector, ImEvector, np.log(DetMatAntiSymm.T))
ax2.set_xlabel(r'Re  $\varepsilon$')
ax2.set_ylabel(r'Im  $\varepsilon$')
plt.show()