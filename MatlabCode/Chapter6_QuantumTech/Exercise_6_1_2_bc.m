% This script is intended to check the validity of the WKB approximation 
% when it comes to estimating tunneling rates. It relates to a specific 
% model for tunneling from a metal surface towards a needle used in the 
% STM setup. 
%
% The inputs are given in units which are converted to atomic units for 
% the calculations.
% 
% Inputs:
% d     - The avarage distance from needle to surface
% V0    - Work function of the metal
% E     - Enery of the conductance electron 
% U     - Voltage between surface and needle
% fMin  - Minimal value the surface function
% fMax  - Maximal value for the surface function
% df    - Step size used in vector with data
% 
% These input parameters are hard coded initially.

% Input parameters
dInAngstrom = 4;
V0_in_eV = 4.5;
Ein_eV = 0.5;
UinV = 1;
fMinInAngstrom = -1;
fMaxInAngstrom = 1;
dfInAngstrom = 0.01;

% Atomic units
a0 = 5.292e-11;          % Length in metres
E0 = 4.360e-18;          % Energy in Joule
e = 1.602e-19;           % Elementary charge in Coulomb

% Convert to atomic units
d = dInAngstrom*1e-10/a0;
fMin = fMinInAngstrom*1e-10/a0;
fMax = fMaxInAngstrom*1e-10/a0;
df = dfInAngstrom*1e-10/a0;
V0 = V0_in_eV*e/E0;
E = Ein_eV*e/E0;
eU = UinV*e/E0;

% Allocate/initiate vectors
fVector = fMin:df:fMax;
len = length(fVector);
TransVectorNum = zeros(1, len);
TransVectorWKB = zeros(1, len);

% Loop over f-varlues and calculate T
index = 1;
for f = fVector
  % The potential
  a = eU/(d-f);
  Vpot = @(x) V0 - a*x;
  D = d-f;
  % Determine transmission rate numerically
  TransVectorNum(index) = TransProb(E, Vpot, D);
  
  % WKB estimate  
  TransVectorWKB(index) = exp(-4*sqrt(2)/(3*a)*...
      ((V0-E)^(3/2)-(V0-E-eU)^(3/2)));
  
  % Update index
  index = index + 1;
end

% Plot result
figure(1)
semilogy(fVector*a0/1e-10, TransVectorNum, 'k-', 'linewidth', 1.5)
hold on
semilogy(fVector*a0/1e-10, TransVectorWKB, 'r--', 'linewidth', 2)
hold off
set(gca, 'fontsize', 15)
xlabel('f [Å]')
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