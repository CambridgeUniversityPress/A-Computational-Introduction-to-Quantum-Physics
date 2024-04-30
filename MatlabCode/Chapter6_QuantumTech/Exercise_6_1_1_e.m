% This script determines the tunneling rate for a particle of unit mass 
% hitting a barrier supported in a finite interval. It solves the 
% time-independent Schrödinger equation in the region where the potential 
% is supported by formulating it as a first order coupled ODE in \psi(x) 
% and \phi(x), where \phi(x) = \psi'(x). 
%
% In this particular case, the potential is constant. This allows us to 
% test our implementation by comparing it to an analytical, exact solution.
%
% Inputs:
% V0 - The height of the potential
% D  - The width of the interval in which the potential 
% is nonzero
% E  - The energy of the particle, must be positive
% 
% These input parameters are hard coded initially.

% Inputs
V0 = 3;
D = 2;
E = 1.75;

% Wave number
k = sqrt(2*E);

% Coefficient matrix for ODE
Mat = [0 1; 2*V0-k^2 0];
% ODE
RHS = @(x,y) Mat*y;
% "Initial value"
yD = [1; 1i*k];
% Solve ODE
[x y] = ode45(RHS, [D 0], yD);

% Read off psi(0) and psi'(0) = phi(0)
Psi0 = y(end, 1);
Phi0 = y(end, 2);
% Transmission probability
Tnumerical = 4*k^2/abs(k*Psi0-1i*Phi0)^2;

% Analytical expression for T
alpha = sqrt(2*(V0-E));
Tanalytical = 1/(1 + V0^2/(4*E*(V0-E))*sinh(alpha*D)^2);

% Write results to screen
disp(['Numerical transmission rate:  ', num2str(100*Tnumerical), ' %.'])
disp(['Analytical transmission rate: ', num2str(100*Tanalytical), ' %.'])