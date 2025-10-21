% This script simulates a spin 1/2-particle which is exposed to a 
% magnetic field. This field has both a static part and a dynamic part. 
% The dynamic part oscillates in time as a sine function.
% The static field points in the z-direction, thus lifting the degenerecy 
% between spin up and spin down. The oscillating field is taken to point 
% along the x-axis, corresponding to a real coupling in the Hamiltonian.
%
% The initial state is the spin up-state.
% 
% The implementation solves the Schrödinger equation (TDSE) by using the 
% MATLAB routine ode45. It plots the probability of a spin up-measurement 
% as a function of time.
%
% The inputs are
% E     - the energy separation induced by the static field
% Omega - the strength of the oscillating field
% w     - the angular frequency of the oscillating field
% OpticalCycles - the duration of the simulation, given
% by the number of periods of the oscillating field.
%
% All inputs are hard coded initially.

% Parameters for the magnetic field
E = 1;
Omega= 0.1;
w = 1.1;

% Duration of the simulation
OpticalCycles=15;
Tfinal = OpticalCycles*2*pi/w;

% Set up the equation - with the Hamiltonian
H0 = [-E/2 0; 0, E/2];                  % Static part
InteractionMatrix = [0 1; 1, 0];        % Coupling matrix
% Right hand side of the TDSE
F = @(t,x) -i*(H0+Omega*sin(w*t)*InteractionMatrix)*x;

% Initial state
x0=[1; 0];                              % Start with spin up

% Solve using built-in function (Runge-Kutta)
[T X]=ode45(F, [0 Tfinal], x0);         % Solves the TDSE

% Plot result
figure(1)
plot(T,abs(X(:,1)).^2, 'k-', 'linewidth', 2)
grid on

% Explain the axes, set limits and fontsizes
xlabel('Time')
ylabel('Spin up-probability')
grid on
set(gca, 'fontsize', 15)
axis([0 Tfinal 0 1.1])
