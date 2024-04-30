% This script estimates expectation value for the 
% momentum p for four given wave functions. 
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
%PsiFunk = @(x) exp(-x.^2);
% Psi_D:
PsiFunk = @(x) (x+i).*exp(-(x-3*i-2).^2/10);

% Numerical grid parameters
L = 20;
N = 100;

% Set up grid
x = linspace(-L/2, L/2, N);
h = L/(N-1);                        % Increment
Psi = PsiFunk(x);                   % Vector with function values

% Normalization
Norm = sqrt(trapz(x, abs(Psi).^2));
Psi = Psi/Norm;

% Set up vector with Psi'(x)
PsiDeriv = zeros(1,N);              % Allocate
% End points (assume Psi = 0 outside the interval)
PsiDeriv(1) =  Psi(2)/(2*h);
PsiDeriv(N) = -Psi(N-1)/(2*h);
% Estimate the derivative with the midpoint rule
for n = 2:(N-1)
  PsiDeriv(n) = (Psi(n+1)-Psi(n-1))/(2*h);
end

% Calculate expectation value
MeanP = -i*trapz(x, conj(Psi).*PsiDeriv);   % Mean momentum
% Write result to screen
disp(['Mean momentum (real part): ', num2str(real(MeanP))])
disp(['Imaginary part: ', num2str(imag(MeanP))])
