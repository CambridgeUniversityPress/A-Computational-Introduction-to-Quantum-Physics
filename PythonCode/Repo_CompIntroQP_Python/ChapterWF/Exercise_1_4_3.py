"""
This script estimates the probability that a 
position measurement will provide a result between
x=a and x=b for four different wave functions.
This is done using Bolean variables.

Inputs:
  PsiFunk - The expression for the unnormalized wave function.
  L       - The extension of the spatial grid 
  N       - The number of grid points
  a and b - The interval in which we seek the particle

Inputs are hard coded initialy, and the function in question is 
selected by commenting in and out the adequate lines.
"""

# Import numpy library
import numpy as np

# Define the interval in question
a = 1
b = 2

# Unnormalzied wave functions (Comment in and out as fit)
def  PsiFunk(x):
  # Psi_A:
  return 1/(1+(x-3)**2)**(3/2)
  # Psi_B:
  #return 1/(1+(x-3)**2)**(3/2)*np.exp(-4*1j*x)
  # Psi_C:
  #return np.exp(-x**2)
  # Psi_D:
  #return (x+1j)*np.exp(-(x-3j-2)**2/10)

# Numerical grid parameters
L = 20
N = 100

# Set up grid
x = np.linspace(-L/2, L/2, N)
Psi = PsiFunk(x);                 # Vector with function values

# Normalization
Norm = np.sqrt(np.trapz(np.abs(Psi)**2, x))
Psi = Psi/Norm;


# Determine the probability of measuring the particle between a and b
Prob = np.trapz((x>a)*(x<b)*np.abs(Psi)**2, x)
print(f'Probability: {Prob*100:.2f} %')