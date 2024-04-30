"""
 This script determines which of the transitions between eigen states 
 within a helium atom produces photons with wavelengths in the visible 
 spectrum. Here we will take this to mean wavelengths in the interval 
 between 385 an 765 nanometres.
 
 The line spectrum is plotted. In doing so, selection rules in total 
 angular momentum quantum number L and total spin S are considered - in 
 addition to checking that the wavelength falls within the interval of 
 visible light.

 The script does not really take inputs, but it relies on the data file 
 Helium_S_L.dat, which includes L and S - in addition to the energies - 
 of all bounds states of the helium atom.
"""

# Import libraries
import numpy as np
from matplotlib import pyplot as plt


# Interval for visible ligth
LambdaMin = 385
LambdaMax = 765

# Convert from eV to nm
PlanckConstant = 6.626e-34
SpeedOfLight = 3.00e8
eVtoJoule = 1.602e-19
MeterToNanoMeter = 1e9
Factor = PlanckConstant*SpeedOfLight/eVtoJoule*MeterToNanoMeter


# Read the input
file = open('Helium_S_L.dat', 'r')
Sdata = []
Ldata = []
Edata = []
for Line in file:
     Vec = np.fromstring(Line, sep = ' ')
     Sdata.append(int(Vec[0]))
     Ldata.append(int(Vec[1]))
     Edata.append(Vec[2])
file.close()


# Loop over all energies
VisibleLambda = []
States = len(Edata)
for index1 in range(0,States):
    for index2 in range(index1+1, States):
        # Wavelength corresponding to the transition - in nm
        Ediff = Edata[index2]-Edata[index1]
        Lambda = Factor/Ediff
        # Change in quantum  numbers S and L
        DeltaS = np.abs(Sdata[index2]-Sdata[index1])       
        DeltaL = np.abs(Ldata[index2]-Ldata[index1])
        # Check if the transition is visible and allowed
        if Lambda > LambdaMin and Lambda < LambdaMax and \
            DeltaS == 0 and DeltaL ==1:
            # Add to list of visible transitions
            VisibleLambda.append(Lambda)

# Function which converts wavelength to RGB (red-green-blue) code
def Wl2RGB(wl):
  # Wl must be given in nanometres. Moreover, it 
  # must reside between 380 nm and 780 nm.
  # This function is an adaption of source code found here:
  # http://www.physics.sfasu.edu/astro/color/spectra.html

  # Determine colour by piecewise linear interpolation
  if wl >= 380 and wl <= 440:  
    R = -1.*(wl-440.)/(440.-380.)
    G = 0
    B = 1
  elif wl > 440 and wl <= 490:
    R = 0
    G = (wl-440.)/(490.-440.)
    B = 1
  elif wl > 490 and wl <= 510:
    R = 0
    G = 1
    B = -1.*(wl-510.)/(510.-490.)
  elif wl > 510 and wl <= 580:
    R = (wl-510.)/(580.-510.)
    G = 1
    B = 0
  elif wl > 580 and wl <= 645:
    R = 1
    G = -1.*(wl-645.)/(645.-580.)
    B = 0
  elif wl > 645 and wl <= 780:  
    R = 1
    G = 0
    B = 0

  # Reduce brightness near the edges
  if wl > 700:
    SSS=.3+.7* (780.-wl)/(780.-700.)
  elif wl < 420:
    SSS=.3+.7*(wl-380.)/(420.-380.)
  else:
    SSS=1  
    
  R = SSS*R
  G = SSS*G
  B = SSS*B

  return [R, G, B]
      
      
# Create line spectrum
plt.figure(1)
plt.clf()
# Set black background
plt.style.use('dark_background')
for Wl in VisibleLambda:
  # Colour for this wavelength
  RGBcode = Wl2RGB(Wl)
  # Horisontal line
  plt.vlines(Wl, 0, 1, linewidth = 3, 
             color = RGBcode)

# Labels, limits and ticks
plt.xlabel(r'$\lambda$ [nm]', fontsize = 12)
plt.xlim(LambdaMin, LambdaMax)
plt.ylim(0, 1)
plt.yticks(ticks = [])
plt.show()     