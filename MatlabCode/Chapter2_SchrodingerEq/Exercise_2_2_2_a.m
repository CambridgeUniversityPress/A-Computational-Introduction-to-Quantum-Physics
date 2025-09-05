% This script estimates expectation values for the kinetic energy
% for four given wave functions.
%
% Inputs:
%   PsiFunk - The expression for the unnormalized wave function.
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%
% The expression for the various wave functions and the value of the two
% numerical input parameters are hard coded initially.

% Unnormalzied wave functions 
% Psi_A:
%PsiFunk = @(x) 1./(1+(x-3).^2).^(3/2);
% Psi_B:
%PsiFunk = @(x) 1./(1+(x-3).^2).^(3/2).*exp(-4*i*x);
% Psi_C:
PsiFunk = @(x) exp(-x.^2);
% Psi_D:
PsiFunk = @(x) (x+i).*exp(-(x-3j-2).^2/10);

% Numerical grid parameters
L = 20;
N = 64;                 % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Set up grid
x = linspace(-L/2, L/2, N);
Psi = PsiFunk(x);                 % Vector with function values

% Normalization
Norm = sqrt(trapz(x, abs(Psi).^2));
Psi = Psi/Norm;

% Determine double derivative by means of three point-formula
PsiDD_FD3 = zeros(1, N);
PsiDD_FD3(1) = (-2*Psi(1)+Psi(2))/h^2;
PsiDD_FD3(N) = (Psi(N-1)-2*Psi(N))/h^2;
for n=2:(N-1)
  PsiDD_FD3(n) = (Psi(n-1)-2*Psi(n)+Psi(n+1))/h^2;
end

% Determine double derivative by means of five point-formula
PsiDD_FD5 = zeros(1, N);
PsiDD_FD5(1) = (-30*Psi(1)+16*Psi(2)-Psi(3))/(12*h^2);
PsiDD_FD5(2) = (16*Psi(1)-30*Psi(2)+16*Psi(3)-Psi(4))/(12*h^2);
PsiDD_FD5(N-1) = (-Psi(N-3)+16*Psi(N-2)-30*Psi(N-1)+16*Psi(N))/(12*h^2);
PsiDD_FD5(N) = (-Psi(N-2)+16*Psi(N-1)-30*Psi(N))/(12*h^2);
for n=3:(N-2) 
  PsiDD_FD5(n)=(-Psi(n-2)+16*Psi(n-1)-30*Psi(n)+...
  16*Psi(n+1)-Psi(n+2))/(12*h^2);
end

% Determine double derivative by means of the fast Fourier transform.
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
PsiDD_FFT = ifft((1i*k).^2.*fft(Psi));

% Calculate expectation values
MeanT_FD3 = -1/2*trapz(x,conj(Psi).*PsiDD_FD3);
MeanT_FD5 = -1/2*trapz(x,conj(Psi).*PsiDD_FD5);
MeanT_FFT = -1/2*trapz(x,conj(Psi).*PsiDD_FFT);

% Print results to screen (only real part):
disp('Kinetic energy estimates (real parts):')
disp(['Three point-forumla: ', num2str(real(MeanT_FD3))])
disp(['Five point-forumla: ', num2str(real(MeanT_FD5))])
disp(['Fast Fourier transform: ', num2str(real(MeanT_FFT))])
