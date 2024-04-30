"""
This script estimates expectation values for the kinetic energy
for four given wave functions.

 Inputs:
   PsiFunk - The expression for the unnormalized wave function.
   L       - The extension of the spatial grid 
   N       - The number of grid points
   
Inputs, indluding the expression for the wave function, are hard coded 
initially.   
"""   
 
# Import the numpy library 
import numpy as np

# Unnormalzied wave functions (Comment in and out as fit)
def  PsiFunk(x):
  # Psi_A:
  #return 1/(1+(x-3)**2)**(3/2)
  # Psi_B:
  #return 1/(1+(x-3)**2)**(3/2)*np.exp(-4*1j*x)
  # Psi_C:
  #return np.exp(-x**2)
  # Psi_D:
  return (x+1j)*np.exp(-(x-3j-2)**2/10)


# Numerical grid parameters
L = 20
N = 64                          # Should be 2^n for FFT's sake

# Set up grid
x = np.linspace(-L/2, L/2, N)
Psi = PsiFunk(x)                 # Vector with function values
h = L/(N-1)

# Normalization
Norm = np.sqrt(np.trapz(np.abs(Psi)**2, x))
Psi = Psi/Norm

# Determine double derivative by means of three point-formula
PsiDD_FD3 = np.zeros(N, dtype=complex)
PsiDD_FD3[0] = (-2*Psi[0]+Psi[1])/h**2
PsiDD_FD3[N-1] = (Psi[N-2]-2*Psi[N-1])/h**2
for n in range(1, N-1):
  PsiDD_FD3[n] = (Psi[n-1]-2*Psi[n]+Psi[n+1])/h**2

# Determine double derivative by means of five point-formula
PsiDD_FD5 = np.zeros(N, dtype=complex)
PsiDD_FD5[0] = (-30*Psi[0]+16*Psi[1]-Psi[2])/(12*h**2)
PsiDD_FD5[1] = (16*Psi[0]-30*Psi[1]+16*Psi[2]-Psi[3])/(12*h**2)
PsiDD_FD5[N-2] = (-Psi[N-4]+16*Psi[N-3]-30*Psi[N-2]+16*Psi[N-1])/(12*h**2)
PsiDD_FD5[N-1] = (-Psi[N-3]+16*Psi[N-2]-30*Psi[N-1])/(12*h**2)
for n in range(2, N-2): 
  PsiDD_FD5[n]=(-Psi[n-2]+16*Psi[n-1]-30*Psi[n]+
                16*Psi[n+1]-Psi[n+2])/(12*h**2)

# Determine double derivative by means of the fast Fourier transform.
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
PsiDD_FFT = np.fft.ifft((1j*k)**2*np.fft.fft(Psi))


# Calculate expectation values
MeanT_FD3 = -1/2*np.trapz(np.conj(Psi)*PsiDD_FD3, x)
MeanT_FD5 = -1/2*np.trapz(np.conj(Psi)*PsiDD_FD5, x)
MeanT_FFT = -1/2*np.trapz(np.conj(Psi)*PsiDD_FFT, x)


# Print results to screen (only real part):
print('Kinetic energy energy estimates:')
print(f'Three point-forumla: {np.real(MeanT_FD3):.4f}')
print(f'Five point-forumla: {np.real(MeanT_FD5):.4f}')
print(f'Fast Fourier transform: {np.real(MeanT_FFT):.4f}')
