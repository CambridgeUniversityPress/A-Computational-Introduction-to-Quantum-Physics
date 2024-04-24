"""
 This script estimates the emitted radiation corresponding to the x-component 
 of a magnetic dipole of a spin 1/2-particle exposed to a mangetic field.
 It is a model for the NMR setup. The magnetic field has a static part 
 oriented along the z-axis, and an oscillating part point along the x-axis.

 The emitted radiation is assumed to be proportional to the time-average of 
 the the double time-derivative of the expectation value of the magnetic 
 dipole moment.

 Physical inputs - given in SI units
 Bz    - the strength of the static magnetic field
 B0    - the amplitude of the oscillating field
 Cycles- number of cycles of the driving field 
 fMin  - the minimal frequency of the oscillating field - in MHz
 fMax  - the maximal frequency of the oscillating field - in MHz
 df    - step size in frequency - in MHz

 Numerical inputs
 StepsPerCycle - Number of temporal steps  size used to estimate double 
 derivative and numerical integral (fixed by fMax here).

 Natural constants
 g     - the proton's g-factor
 m     - the electron mass
 e     - the elementary charge

 All inpust are hard coded initially
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Inputs
# Magnetic field strengths - given in T
Bz = 5
B0 = 0.01
# The number of cycles - determines interaction time
Cycles = 100
# Frequency parameters - given in MHz
fMin = 100
fMax = 300
df = 0.1

# Constants
g = 5.5857          # G-factor for the magnetic moment
m = 1.673e-27       # Proton mass
hbar = 1.055e-34    # Reduced Planck constant
e = 1.602e-19       # Elementary charge

# Numercal time-increment
StepsPerCycle = 150

# Derived parameters. Note that Omega, with capital O, differs
# from the angular frequency omega.
E = g*e*hbar/(2*m)*Bz
Omega = -g*e*hbar/(4*m)*B0

# Vector with frequencies
fVector = np.arange(fMin, fMax+df, df)
omegaVector = fVector*1e6*2*np.pi     # Angular frequency - in Hz

# Initialize vector with intensities of emitted radiation
L = len(omegaVector)
IntensityVector = np.zeros(L)

# Function to determine Intensity
def Intensity(omega):
    
  # Assign detuning and generalized Rabi frequency
  delta = E/hbar - omega
  OmegaG = np.sqrt(delta**2 + abs(Omega)**2)
  
  # Fix resolution in numerical estimates
  StepsPerCycle = 100               # Time steps per optical cycle
  Tfinal = Cycles*2*np.pi/omega     # Interaction duration
  t = np.linspace(0, Tfinal, StepsPerCycle*Cycles)
  dt = t[1]-t[0]
  
  # Calculate amplitudes for spin up and down as functions
  # of time
  a = np.exp( 1j*omega*t/2)*(np.cos(OmegaG*t/2) + 
      1j*delta/OmegaG*np.sin(OmegaG*t/2))
  b = np.exp(-1j*omega*t/2)*Omega/OmegaG*np.sin(OmegaG*t/2)
  
  # Dipole moment in x-direction
  DipoleX = np.real(a*np.conj(b))
  
  # Determine double derivative by means of the three-point
  # finite differencd scheme
  DipoleX_DoubleDeriv = (DipoleX[0:-2]-2*DipoleX[1:-1] \
      +DipoleX[2:])/dt**2
  
  # Calculate average radiation itensity (proportional to
  # the square of the double derivative of the dipole moment)
  return 1/Tfinal*dt*sum(DipoleX_DoubleDeriv**2)


# Loop over all input frequencies
index = 0
for omega in omegaVector:
  # Call function to determine radiated intensity
  IntensityVector[index] = Intensity(omega)
  
  # Update index
  index = index + 1


# Plot response - with resonance frequency
plt.figure(1)
plt.clf()
plt.plot(fVector, IntensityVector, '-', linewidth = 2, 
         color = 'black')
# Resonance frequency - in MHz
fRes = E/(2*np.pi*hbar*1e6)
print(f'Resonance frequency: {fRes:3.2f} MHz.')
MaxVal = 1.1*np.max(IntensityVector)
plt.vlines(fRes, 0, MaxVal, linewidth = 1.5, 
           color = 'red', linestyles = 'dashed')
plt.xlabel('Frequency [MHz]', fontsize = 12)
plt.ylabel('Intensity [Arbitrary units]', fontsize = 12)
plt.yticks([])
plt.grid()
plt.show()