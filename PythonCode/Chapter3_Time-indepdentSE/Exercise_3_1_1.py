"""
 This script plots the determinant which is to be set to zero
 to ensure non-trivial solutions for the time-independent 
 Schrödinger equation for a particle of unit mass trapped inside a 
 rectangular well potential.
 
 There are two input-parameters: The "height" of the well, V_0,
 and the width w of the well. As this is to be a well, not a barrier
 V_0 must be negative

 The zero points of the functions indicate admissible, quantized
 energies
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Input parameters
V0 = -4        # Negative debth
w = 5          # Widht

# Energy vector
Npoints = 200;
Energies = np.linspace(V0, 0, Npoints)

# kappa vector (derived from energy vector)
kappaVector = np.sqrt(-2*Energies)

# k vector (derived from energy vector)
kVector = np.sqrt(2*(Energies-V0))

# Determinant corresponding to the symmetric wave function
DetSymm = kappaVector*np.cos(kVector*w/2) - kVector*np.sin(kVector*w/2);

# Determinant corresponding to the anti-symmetric wave function
DetAntiSymm = kappaVector*np.sin(kVector*w/2) + kVector*np.cos(kVector*w/2)

# Plot the functions which are to be zero
plt.figure(1)
plt.clf()
plt.plot(Energies, DetSymm, '-', color = 'blue', label = r'Symmetric $\Psi$')
plt.plot(Energies, DetAntiSymm, '--', color = 'red', 
         label = r'Anti-symmetric $\Psi$')
plt.axhline(y = 0, color = 'black')
plt.grid()
plt.legend()
plt.show()

# Find approximate roots by brute force and write them to screen
for n in range(1, Npoints):
  if DetSymm[n]*DetSymm[n-1]<0 or DetAntiSymm[n]*DetAntiSymm[n-1]<0:
    EnergyMid = (Energies[n]+Energies[n-1])/2
    print(f'Permissible energy: {EnergyMid:.4f}')
