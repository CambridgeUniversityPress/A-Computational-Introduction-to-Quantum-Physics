% This script estimates the probability that a 
% position measurement will provide a result between
% x=a and x=b for four different wave functions.
%
% Inputs:
%   PsiFunk - The expression for the unnormalized wave function.
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   a and b - The interval in which we seek the particle
% 
% All inputs, including the expressions for the wave functions, are hard
% coded initially.

% Interval
a = 1;
b = 2;

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
L = 50;
N = 200;

% Set up grid - for normalization
x = linspace(-L/2, L/2, N);
Psi = PsiFunk(x);                 % Vector with function values

% Normalization
Norm = sqrt(trapz(x, abs(Psi).^2));
Psi = Psi/Norm;


% Use logical variables to set Psi to zero outside the interval
PsiSqBetween = (x>a) .* (x<b) .* abs(Psi).^2; 
% Probability
P = trapz(x, PsiSqBetween);
% Write result to screen
disp(['Probability to find particle between x=',num2str(a),' and x=',...
    num2str(b),': ',num2str(100*P), '%'])