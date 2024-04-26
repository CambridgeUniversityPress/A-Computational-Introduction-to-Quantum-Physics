% This script estimates expectation values for r
% and p for the ground state of Hydrogen.
% It does so using atomic units.
%
% Inputs:
%   L       - The extension of the radial grid 
%   N       - The number of grid points

% Numerical grid parameters
L = input('Please provide the extension of your grid (in a.u.): ');
N = input('Please provide the number of grid points: ');

% Provide wave function
PsiFunk = @(r) 2*r.*exp(-r);

% Set up grid
r = linspace(0,L,N);
Psi = PsiFunk(r);               % Vector with function values
h = L/N;                        % Increment 

% Calculate expectation value for r
MeanR = sum(r.*abs(Psi).^2)*h;                 % Mean position
% Write result to screen
disp(['Mean distace: ', num2str(MeanR), ' a.u.'])

% Set up vector with Psi'(r)
PsiDeriv = zeros(1, N);              % Allocate
% End points
PsiDeriv(1) = (Psi(2)-Psi(1))/h;    % Two point formula
PsiDeriv(N) = -Psi(N-1)/(2*h);
% Estimate the derivative with the midpoint rule
for n = 2:(N-1)
  PsiDeriv(n) = (Psi(n+1)-Psi(n-1))/(2*h);
end

% Calculate expectation value for p
MeanP = -1i*sum(conj(Psi).*PsiDeriv)*h;   % Mean momentum
% Write result to screen
disp(['Mean momentum (real part): ', num2str(real(MeanP)), ' a.u.'])
