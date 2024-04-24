% This script simulates the evolution of an particle which hits a double 
% barrier. Each of the two identical parts of the barrier has a smooth 
% rectangular-like shape.
% 
% The transmission probability is determined by semi-analytical means 
% where plane wave solutions are imposed in the regions where the double 
% barrier potential is no longer supported. The script loops over several
% energy values for the solution and and, in the end, plot the 
% transmission probability as a function of mean energy.
% 
% It imposes a complex absorbing potential and uses the accumulated
% absorption at each end in order to estimate reflection and trasmissiion
% probabilities.
%
% The absorbing potential is a quadratic monomial.
% 
% Input for the barrier:
%   V0      - The height of the barrier (can be negative)
%   w       - The width of the barrier
%   s       - Smoothness parameter
%   d       - Distance between the barriers (centre to centre)
%   D       - The length of the interval, starting from x = 0, where the
%   potential is supported. This quantity may very well be related to d 
% and  w.
%
% Inputs for the energy grid
%   Emin    - Minmal energy 
%   Emax    - Maximal energy
%   dE      - Energy increment
% 
% All inputs are hard coded initially.

% Input parameters for the barrier
V0 = 4;
w = 0.5;
s = 25;
d = 3;
D = 2*d + 4*w;

% Energies to calculate
dE = 0.01;
Emin = 0.01;
Emax = V0;

% Set up barrier
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);
Vdouble = @(x) Vpot(x-d) + Vpot(x+d);
VdoubleShifted = @(x) Vdouble(x-D/2);

% Vector with energies
Evector = Emin:dE:Emax;
Tvector = zeros(1, length(Evector));       % Allocate
index = 1;

% Loop over various initial energies and estimate transmission 
% probabilites
for E = Evector
  % Estimate T
  Tprob = TransProb(E, VdoubleShifted, D);
  Tvector(index) = Tprob;
  
  % Print result to screen
  disp(['E and T: ', num2str([E Tprob])])  
  
  % Update index
  index = index + 1;
end

% Plot the result - T as a function of E
figure(1)
plot(Evector, Tvector, 'k-', 'linewidth', 2)
grid on
xlabel('Initial energy')
ylabel('Tranmission probability')
set(gca, 'fontsize', 15)

% Function which determines transmission probability as a function of
% energy.
function T = TransProb(E, Vpot, D)

% Wave number
k = sqrt(2*E);

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