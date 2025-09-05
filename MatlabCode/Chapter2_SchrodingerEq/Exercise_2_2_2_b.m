% This script estimates expectation values for the kinetic energy
% for four given wave functions. It does so by constructing matrix
% representations for the kinetic energy operator and then estimate
% the expectation value by means of matrix products.
%
% Inputs:
%   PsiFunk - The expression for the unnormalized wave function.
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%
% All inputs, including the expressions for the wave functions, are hard
% coded initially.

% Unnormalzied wave functions 
% Psi_A:
%PsiFunk = @(x) 1./(1+(x-3).^2).^(3/2);
% Psi_B:
%PsiFunk = @(x) 1./(1+(x-3).^2).^(3/2).*exp(-4*i*x);
% Psi_C:
PsiFunk = @(x) exp(-x.^2);
% Psi_D:
%PsiFunk = @(x) (x+i).*exp(-(x-3*i-2).^2/10);

% Numerical grid parameters
L = 20;
N = 32;                     % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector
Psi = PsiFunk(x);                           % Vector with function values

% Normalization
Norm = sqrt(trapz(x, abs(Psi).^2));
Psi = Psi/Norm;

% Set up kinetic energy matrix by means of three point-formula
Tmat_FD3 = zeros(N, N);
Tmat_FD3(1, [1 2]) = [-2 1];
Tmat_FD3(N, [N-1 N]) = [1 -2];
for n=2:(N-1)
  Tmat_FD3(n, [n-1 n n+1]) = [1 -2 1]; 
end
Tmat_FD3 = Tmat_FD3/h^2;        % Correct duble derivative
Tmat_FD3 = -1/2*Tmat_FD3;       % Correct prefactor

% Set up kinetic energy matrix by means of five point-formula
Tmat_FD5 = zeros(N, N);
Tmat_FD5(1, [1:3]) = [-30 16 -1];
Tmat_FD5(2, [1:4]) = [16 -30 16 -1];
Tmat_FD5(N-1, [(N-3):N]) = [-1 16 -30 16];
Tmat_FD5(N, [(N-2):N]) = [-1 16 -30];
for n=3:(N-2)
  Tmat_FD5(n, [(n-2):(n+2)]) = [-1 16 -30 16 -1]; 
end
Tmat_FD5 = Tmat_FD5/(12*h^2);        % Correct duble derivative
Tmat_FD5 = -1/2*Tmat_FD5;            % Correct prefactor

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;            % Correct prefactor

% Calculate expectation values
MeanT_FD3 = h*Psi'*Tmat_FD3*Psi;
MeanT_FD5 = h*Psi'*Tmat_FD5*Psi;
MeanT_FFT = h*Psi'*Tmat_FFT*Psi;

% Print results to screen:
disp('Kinetic energy estimates (real parts):')
disp(['Three point-forumla: ', num2str(real(MeanT_FD3))])
disp(['Five point-forumla: ', num2str(real(MeanT_FD5))])
disp(['Fast Fourier transform: ', num2str(real(MeanT_FFT))])
