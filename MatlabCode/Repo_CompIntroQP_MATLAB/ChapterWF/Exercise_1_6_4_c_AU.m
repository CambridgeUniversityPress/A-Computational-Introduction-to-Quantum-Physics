% This script estimates the energy expectation value
% for the ground state of the hydrogen atom - in atomic units.
%
% Inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points

% Numerical grid parameters
L = input('Please provide the extension of your grid: ');
N = input('Please provide the number of grid points: ');


% Provide wave function (normalized)
PsiFunk = @(r) 2*r.*exp(-r);

% Set up grid
h = L/N;                    % Increment
r = linspace(h, L, N);
Psi = PsiFunk(r);           % Vector with function values

%
% Estimate kinetic energy expectation value
%
% Set up vector with Psi''(x)
PsiDoubleDeriv = zeros(1,N);              % Allocate
% End points (use Psi(r=0) = 0)
PsiDoubleDeriv(1) =  (Psi(2)-2*Psi(1))/h^2;
PsiDoubleDeriv(N) = (Psi(N-1)-2*Psi(N))/h^2;
% Estimate the derivative with the midpoint rule
for n = 2:(N-1)
  PsiDoubleDeriv(n) = (Psi(n+1)-2*Psi(n)+Psi(n-1))/h^2;
end
%
ExpectKinEn = -1/2*trapz(r, conj(Psi).*PsiDoubleDeriv);

%
% Estimate potential energy exectation value
%
ExpectPotEn = -trapz(r, 1./r.*abs(Psi).^2);

% Total energy
ExpectEn = ExpectKinEn + ExpectPotEn;
% Write energy to screen
disp(['Energy expectation value: ', num2str(ExpectEn), ' a.u.'])