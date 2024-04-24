% This script simulates a spin 1/2-particle which is exposed to a 
% magnetic field. It does so in two ways: 1) by direct, numerical 
% solution of the Schroedinger equation (TDSE) and 2) approximatively 
% by the analytical solution within the rotating wave approximation (RWA).
%
% The magnetic field has both a static part and a dynamic part. The 
% dynamic, which points along the x axis, part oscillates in time as a 
% sine function. The static field points in the z-direction, thus 
% lifting the degenerecy between spin up and spin down.
%
% The initial state is a spin up-state.
%
% The implementation solves the Schrödinger equation by using the MATLAB 
% routine ode45. It plots the probability of a spin up-measurement as a 
% function of time.
%
% The inputs are
% E     - the energy separation induced by the static field
% Omega - the strength of the oscillating field
% w     - the angular frequency of the oscillating field
% OpticalCycles - the duration of the simulation, given
% by the number of periods of the oscillating field.
%
%All inputs are hard coded initially.

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

% Analytical, approximate solution (RWA)
delta = w - E;
SpinUpProbRWA = 1/(delta^2 + Omega^2) * (delta^2 + Omega^2 * ...
                    cos(sqrt(delta^2 + Omega^2)*T/2).^2);

% Plot results
figure(1)
% Numerical TDSE solution
plot(T, abs(X(:,1)).^2, 'k-', 'linewidth', 2)
% Approximate RWA solution
hold on
plot(T, SpinUpProbRWA, 'r--', 'linewidth', 2)
hold off
grid on
% Explain the axes and graphs and set limits and fontsizes
xlabel('Time')
ylabel('Spin up-probability')
legend('TDSE', 'RWA')
set(gca, 'fontsize', 15)
axis([0 Tfinal 0 1.1])