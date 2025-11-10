% This script simulates a system consisting of two 
% spin 1/2-particles exposed to the same magnetic field. 
% This field has both a static part and a dynamic part. 
% The dynamic part oscillates in time as a sine function.
% The static field points in the z-direction, thus lifting
% the degenerecy between spin up and spin down. The
% oscillating field is taken to point along the x-axis.
%
% The two particle interacts thorough the spin-spin interaction.
% 
% The implementation solves the Schrödinger equation (TDSE)
% by using the MATLAB routine ode45. It plots the 
% probability of both spins pointing upwards and the probability 
% of opposite alignment along the z-axis as functions of time.
%
% Several choices of initial state are listed; the desired one is 
% selected by commenting out the other ones.
%
% The inputs are
% E     - the energy separation induced by the static field
% Omega - the strength of the oscillating field
% w     - the angular frequency of the oscillating field
% OpticalCycles - the duration of the simulation, given
% by the number of periods of the oscillating field.
% u     - the strength of the spin-spin interaction.
% 
% All inputs are hard coded initially.

% Parameters for the magnetic field
E = 1;
Omega= 0.2;
w = 1.1;

% Spin-spin interaction strength
u = 0.025;

% Duration of the simulation
OpticalCycles=15;
Tfinal = OpticalCycles*2*pi/w;

% Set up the equation - with the Hamiltonian
% Static part
H0 = [-E+u 0 0 0; 0 -u 2*u 0; 0 2*u -u 0; 0 0 0 E+u];                  
% Coupling matrix
InteractionMatrix = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0];        
% Right hand side of the TDSE
F = @(t,x) -i*(H0+Omega*sin(w*t)*InteractionMatrix)*x;

% Initial state
%x0 = [1; 0; 0; 0];              % Up-up
x0 = [0; 1; 1; 0]/sqrt(2);     % Entangled, exchange symmetric
%x0 = [0; 1; -1; 0]/sqrt(2);    % Entangled, exchange anti-symmetric
%x0 = [0; 1; 0; 0];             % Up-down

% Solve using built-in function (Runge-Kutta)
[T X]=ode45(F, [0 Tfinal], x0);         % Solves the TDSE

% Plot result
figure(1)
% Up-Up probability
plot(T, abs(X(:,1)).^2, 'k-', 'linewidth', 2)
hold on
% Probability for opposite spin alingment along the z-axis
plot(T,abs(X(:,2)).^2 + abs(X(:,3)).^2, 'r-', 'linewidth', 2)
% Up-down probability
plot(T,abs(X(:,2)).^2, 'b-', 'linewidth', 2)
hold off
% Explain the axes, set limits and fontsizes
grid on
xlabel('Time')
ylabel('Probability')
%legend('Up-up', 'Opposite')
legend('Up-up', 'Opposite', 'Up-down')
set(gca, 'fontsize', 15)
axis([0 Tfinal 0 1.1])

% Check for symmetry breaking
figure(2)
plot(T, abs(X(:,2)-X(:,3)), 'k-', 'linewidth', 2)
grid on
xlabel('Time')
ylabel('|b(t)-c(t)|')
set(gca, 'fontsize', 15)
