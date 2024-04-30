% This script simply tests our numerical solution of the Schrödinger 
% equation against an exact analytical soltuion. The test case 
% corresponds to a constant Hamiltonian.
%
% The initial state is a spin up-state.
% 
% The implementation solves the Schrödinger equation (TDSE) by using 
% the MATLAB routine ode45. It plots the probability of a spin 
% up-measurement as a function of time - along with the corresponding 
% analytical solution.
%
% The inputs are
% E         - the difference betweeen the diagonal elements 
% of the Hamiltonian
% Omega     - twice the coupling element
% Tfinal    - the duration
%
% All inputs are hard coded initially.

% Parameters for the magnetic field
E = 0.5;
Omega= 1;             % taken to be real

% Duration of the simulation
Tfinal = 40;

% Set up the equation - with the Hamiltonian
H0 = [-E/2 0; 0, E/2];                  % Static part
InteractionMatrix = [0 1; 1, 0];        % Coupling matrix
% Right and side of the TDSE
F = @(t,x) -i*(H0+Omega/2*InteractionMatrix)*x;

% Initial state
x0=[1; 0];                              % Start with spin up

% Solve using built-in function (Runge-Kutta)
[T X]=ode45(F, [0 Tfinal], x0);         % Solves the TDSE

% Plot result
figure(1)
plot(T,abs(X(:,1)).^2, 'k-', 'linewidth', 2)

% Analytical solution
Tvector = linspace(0,Tfinal,250);       
Analytical = 1/(E^2 + Omega^2)*...
    (E^2 + Omega^2*cos(sqrt(E^2+Omega^2)/2*Tvector).^2);
hold on
% Plot the analytical solution
plot(Tvector, Analytical, 'r--', 'linewidth', 2)               
hold off
grid on

% Explain the graphs and axes, set limits and fontsizes
xlabel('Time')
ylabel('Spin up-probability')
legend('Numerical','Analytical')   
set(gca, 'fontsize', 15)
grid on
axis([0 Tfinal 0 1.1])