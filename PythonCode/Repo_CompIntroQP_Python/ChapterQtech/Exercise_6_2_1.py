"""
 This script determines which of the possible transitions between eigen 
 states within a hydrogen atoms produces photons with wavelengths in the 
 visible spectrum. Here we will take this to mean wavelengths in the interval 
 between 385 an 765 nanometres.
 
 The wavelenghs and transitions in question are written to screen. The 
 corresponding line spectrum is also plotted. To this end, a function which 
 converts wavelength to RGB code is introduced.

 The script does not really take inputs, save the maximum 
 quantum number n involved.
"""

# Import Pyplot
from matplotlib import pyplot as plt

# Maximum quantum number n
nMax = 20

# Interval for visible ligth
LambdaMin = 385
LambdaMax = 765

# Natural constants
c = 3.00e8              # The speed of light
B = 2.179e-18           # The Bohr constant
h = 6.63e-34            # The Planck constant (not the reduced one)


# Loop over all initial quantum numbers n1
VisibleLambda = []
Transition = []
# Write heading to screen
print('Transitions in the visible spectrum:')
for n1 in range(2, nMax+1):
  # Loop over all final quantum numbers
  for n2 in range(1, n1):
    # Wavelength of emitted photon - in nanometres
    LambdaNM = h*c/(B*(1/n2**2-1/n1**2))/1e-9
    if LambdaNM > LambdaMin and LambdaNM < LambdaMax:
      VisibleLambda.append(LambdaNM)
      Transition.append([n1, n2])
      # Write visible transition and wavelength to screen
      print(f'From n={n1} to n={n2}, wavelength: {LambdaNM:3.2f} nm.')


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