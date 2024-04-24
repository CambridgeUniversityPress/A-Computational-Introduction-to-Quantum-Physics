"""
 This script plots the relativistic and non-relativistic kinetic energy 
 functions of momentum and energy. The former is done in two ways - in SI 
 units for an object of 1 kg mass and more genrically with (p/mc)^2 as 
 argument and with mc^2 as energy unit.
 
 500 points have been use for each of the plots. When the momentum is
 the argument, we have set 2mc as upper limit, while the velocity plot
 goes up to almost c, the speed of light.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Constants (SI units)
c = 3.00e8
m = 1


# Plot against momentum - SI
# Momentum vector
p = np.linspace(0, 2*m*c, 500)
# Energy
T_nonrel = p**2/(2*m)
T_rel = m*c**2*(np.sqrt(1+p**2/(m*c)**2)-1)

plt.figure(1)
plt.clf()
plt.plot(p, T_nonrel, '-', linewidth = 3, color = 'blue', 
         label = 'Non-relativistic')
plt.plot(p, T_rel, '-', linewidth = 3, color = 'red', 
         label = 'Relativistic')
plt.grid()
plt.xlabel('Momentum [kg m/s]', fontsize = 15)
plt.ylabel('Energy [J]', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

#
# Plot against momentum - generic units
#
# 'Momentum' vector
x = np.linspace(0, 2**2, 500)
# Energy
T_nonrel = x/2
T_rel = (np.sqrt(1+x)-1)

plt.figure(2)
plt.clf()
plt.plot(x, T_nonrel, '-', linewidth = 3, color = 'blue', 
         label = 'Non-relativistic')
plt.plot(x, T_rel, '-', linewidth = 3, color = 'red', 
         label = 'Relativistic')
plt.grid()
plt.xlabel('$(p/mc)^2$', fontsize = 15)
plt.ylabel('Energy [$mc^2$]', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

#
# Plot against velocity
#
# Velocity vector
v = np.linspace(0, 0.99*c, 500)
# Energy
T_nonrel = 0.5*m*v**2
T_rel = m*c**2*(1/np.sqrt(1-(v/c)**2)-1)

plt.figure(3)
plt.clf()
plt.plot(v, T_nonrel, '-', linewidth = 3, color = 'blue', 
         label = 'Non-relativistic')
plt.plot(v, T_rel, '-', linewidth = 3, color = 'red', 
         label = 'Relativistic')
plt.grid()
plt.ylim(0, 3e17)
plt.xlabel('Velocity [m/s]', fontsize = 15)
plt.ylabel('Energy [J]', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()