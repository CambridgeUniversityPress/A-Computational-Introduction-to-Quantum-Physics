% This script calculates the width in position and momentum 
% for four given wave functions. For the momentum width, it 
% does so by means of a three point finite difference formula.
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
h = L/(N-1);                            % Increment
Psi = PsiFunk(x);                       % Vector with function values

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

% Set up vector with Psi''(x)
PsiDoubleDeriv = zeros(1,N);              % Allocate
% End points (assume Psi = 0 outside the interval)
PsiDoubleDeriv(1) =  (-2*Psi(1)+Psi(2))/h^2;
PsiDoubleDeriv(N) = (Psi(N-1)-2*Psi(N))/h^2;
% Estimate the derivative with the midpoint rule
for n = 2:(N-1)
  PsiDoubleDeriv(n) = (Psi(n-1)-2*Psi(n)+Psi(n+1))/h^2;
end


% Calculate x expectation value
MeanX = trapz(x, x.*abs(Psi).^2); 
% Calculate x^2 expectation value
MeanXsq = trapz(x, x.^2.*abs(Psi).^2); 
% Calculate p expectation value
MeanP = -i*trapz(x, conj(Psi).*PsiDeriv);
% Calculate p^2 expectation value
MeanPsq = -trapz(x, conj(Psi).*PsiDoubleDeriv);

% Determine widths
SigmaX = sqrt(MeanXsq - MeanX^2);
SigmaP = sqrt(MeanPsq - MeanP^2);
% Write result to screen
disp(['Position width: ', num2str(SigmaX)])
disp(['Momentum width (real part): ', num2str(real(SigmaP))])
disp(['Imaginary part: ', num2str(imag(SigmaP))])

% Check uncertainty principle
Product = SigmaX*SigmaP;
% Write product to screen
disp(['Product: ', num2str(real(Product))])