% This script estimates the energy expectation value
% for the ground state of the hydrogen atom - in SI units.
%
% Inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points

% Numerical grid parameters
L = input('Please provide the extension of your grid (in metres): ');
N = input('Please provide the number of grid points: ');

% Relevant constants in SI units
a0 = 5.292e-11;                 % The Bohr radius
hbar = 1.055e-34;               % The reduced Planck constant 
ElemCharge = 1.602e-19;         % Elementary charge                  
Mass = 9.109e-31;               % Electron mass
Eps0 = 8.854e-12;               % Coulomb constant

% Provide wave function (normalized)
PsiFunk = @(r) 2/a0^(3/2)*r.*exp(-r/a0);

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
ExpectKinEn = -hbar^2/(2*Mass)*trapz(r, conj(Psi).*PsiDoubleDeriv);

%
% Estimate potential energy exectation value
%
ExpectPotEn = -ElemCharge^2/(4*pi*Eps0) * trapz(r, 1./r.*abs(Psi).^2);

% Total energy
ExpectEn = ExpectKinEn + ExpectPotEn;
% Write energy to screen
disp(['Energy expectation value: ', num2str(ExpectEn), ' J'])
