% This script determines the tunneling rate for a particle of unit mass 
% hitting a potential supported in a finite interval.
% It solves the time-independent Schrödinger equation in the region where 
% the potential is supported by formulating it as a first order coupled 
% ODE in \psi(x) and \phi(x), where \phi(x) = \psi'(x). It does so for a 
% set of several energy values.
%
% Inputs:
% Vpot  - The potential; this input is an inline function - one
% that features certain additional input parameters
% D     - The width of the interval in which the potential 
% is nonzero
% Emin  - The minimal energy of the particle, must be positive
% Emax  - The maximal energy
% dE    - The increment of the vector containing input energies
% 
% These input parameters are hard coded initially.

% Input parameters
D = 3;
Emin = 0.1;
Emax = 4;
dE = 0.05;

% The potential
V0 = 2;
a = 0.4;
Vpot = @(x) V0-a*x;

% Allocate/initiate vectors
EnergyVector = Emin:dE:Emax;
len = length(EnergyVector);
TransVector = zeros(1, len);

index = 1;
for Energy = EnergyVector
  TransVector(index) = TransProb(Energy, Vpot, D);
  index = index + 1;
end

% Plot result
figure(1)
subplot(1,2,1)
plot(EnergyVector, TransVector, 'k-', 'linewidth', 2)
set(gca, 'fontsize', 20)
xlabel('Energy')
ylabel('Transmission Probability')
grid on
subplot(1,2,2)
semilogy(EnergyVector, TransVector, 'k-', 'linewidth', 2)
set(gca, 'fontsize', 20)
xlabel('Energy')
ylabel('Transmission Probability')
grid on

% Function which calculates T for a given energy
function T = TransProb(E, Vpot, D)

  % Wave number
  k = sqrt(2*E);

  % Coefficient matrix for ODE
  % ODE
  RHS = @(x,y) [0 1; 2*Vpot(x)-k^2 0]*y;
  % "Initial value"
  yD = [1; 1i*k];
  % Solve ODE
  [x y] = ode45(RHS, [D 0], yD);

  % Read off psi(0) and psi'(0) = phi(0)
  Psi0 = y(end, 1);
  Phi0 = y(end, 2);
  % Transmission probability
  T = 4*k^2/abs(k*Psi0-1i*Phi0)^2;
end