 """
 This scripts uses current data from a (fictious) scanning tunneling 
 microscipe (STM) setup to determine the shape of a surface. It does to by 
 using the WKB-approximation to estimate the tunneling rate.
 
 The current data is given by the file Current.dat, in which the first 
 column is the distance from the needle's startig point along one particular 
 direction along the surface. This quantity is given in Ångström. The second 
 column is the measured current - in arbitrary units.

 Finally, the surface data is interpolated using a cubic spline (from the 
 SciPy library) to render a more smooth surface.
 
 The input parameters are
 U     - The voltage between needle and surface - in Volt
 V0    - The work function energy gap between metal and 
 needle - in eV
 E     - The energy of the conductance electrons - in eV
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

# Inputs
U = 1       # Voltage - in V
E = 0.5     # Energy - in eV
V0 = 4.5    # Work function - in eV

# Constants
m = 9.109e-31        # Electron mass - in kg
hbar = 1.055e-34     # Reduced Planck constant - in Js
e = 1.602e-19        # Elemetary charge - in C

# Convert inputs to SI
EinSI = E*e
V0inSI = V0*e

# Read the input
file = open('Current.dat', 'r')
Ydata = []
Idata = []
for Line in file:
     Vec = np.fromstring(Line, dtype = float, sep = ' ')
     Ydata.append(Vec[0])
     Idata.append(Vec[1])
file.close()


# Prefactor in formula
Prefactor = 3*e*U*hbar/(4*np.sqrt(2*m)*((V0inSI-EinSI)**(3/2) \
    -(V0inSI-EinSI-e*U)**(3/2)))

# Vector with f(y)
Fdata = Prefactor*np.log(Idata)
# Set mean value to zero and convert from metres to Ångström
Fdata = Fdata-np.mean(Fdata)
Fdata = Fdata/1e-10

# Interploate (with 100 points)
Ydense = np.linspace(min(Ydata), max(Ydata), 100)
spline = CubicSpline(Ydata, Fdata)
Fdense = spline(Ydense)

# Plot the shape of the sruface
plt.figure(1)
plt.clf()
plt.plot(Ydense, Fdense, 'k-', linewidth = 2)
plt.xlabel('y [Å]', fontsize = 12)
plt.ylabel('f(y) [Å]', fontsize = 12)
plt.axis('equal')
plt.show()