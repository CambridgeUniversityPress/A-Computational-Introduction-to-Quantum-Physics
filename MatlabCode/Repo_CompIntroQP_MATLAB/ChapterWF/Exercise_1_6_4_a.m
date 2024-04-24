% This script estimates expectation values for r
% and p for the ground state of Hydrogen.
% It does so using SI units.
%
% Inputs:
%   L       - The extension of the radial grid 
%   N       - The number of grid points

% Numerical grid parameters
L = input('Please provide the extension of your grid (in metres): ');
N = input('Please provide the number of grid points: ');

% Relevant constants in SI units
a0 = 5.292e-11;                 % The Bohr radius
hbar = 1.055e-34;               % The reduced Planck constant 

% Provide wave function
PsiFunk = @(r) 2/a0^(3/2)*r.*exp(-r/a0);

% Set up grid
r = linspace(0, L, N);
Psi = PsiFunk(r);               % Vector with function values
h = L/(N-1);                    % Increment 

% Calculate expectation value for r
MeanR = trapz(r, r.*abs(Psi).^2);                 % Mean position
% Write result to screen
disp(['Mean distace: ', num2str(MeanR), ' m'])

% Set up vector with Psi'(r)
PsiDeriv = zeros(1, N);             % Allocate
% End points (Assuming Psi(0) = Psi(R) = 0)
PsiDeriv(1) =  Psi(2)/(2*h);    
PsiDeriv(N) = -Psi(N-1)/(2*h);
% Estimate the derivative with the midpoint rule
for n = 2:(N-1)
  PsiDeriv(n) = (Psi(n+1)-Psi(n-1))/(2*h);
end

% Calculate expectation value for p
MeanP = -1i*hbar*trapz(r, conj(Psi).*PsiDeriv);   % Mean momentum
% Write result to screen
disp(['Mean momentum (real part): ', num2str(real(MeanP)), ' kg*m/s'])
