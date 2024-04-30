% This script solves the GKLS equation for a qubit with amplitude damping. 
% The unitary part of the evolution is given by a time-independent 
% Hamiltonian. The rate at which the 1-state decays spontaneously into 
% the 0-state is also constant.
% 
% The equation is solved by writing the density matrix as a column vector, 
% in which case the time derivative may be written as a matrix times the 
% vector itself.
%
% The inputs are
% E     - the diagonal elements of the Hamiltonian
% W     - the coupling (in the Hamiltonian), taken to be real
% Gamma     - the decay rate (amplitude damping)
% Ttotal    - the duration of the interaction
%
% All inputs are hard coded initially.

% Parameters for the Hamiltonian
E = 0.2;
W = 1;

% Decay rate
Gamma = 0.05;

% Duration of the simulation
Tfinal = 75;

% Set up the equation - with the Hamiltonian
% Allocate
M = zeros(4, 4);
% Diagonal part (in the Hamiltonian)
M(2, 2) = -E;
M(3, 3) = E;
% Add the coupling (in the Hamiltonian)
M = M + W/2*[0, -1, 1, 0;
            -1, 0, 0, 1; 
            1, 0, 0, -1;
            0, 1, -1, 0];
% Add decay (in the Lindbladian)
M = M + 1i*Gamma/2*[0, 0, 0, 2;
                    0, -1, 0, 0;
                    0, 0, -1, 0;
                    0, 0, 0, -2];

% Right and side of the dyamical equation
F = @(t,x) -1i*M*x;

% Initial state (a pure |1> state)
x0 = [0; 0; 0; 1];

% Solve using built-in function (Runge-Kutta)
[T Rho]=ode45(F, [0 Tfinal], x0);         % Solves the TDSE

% Analytical solution for Gamma = 0
AnalyticalNoDecay = (E^2 + W^2*cos(0.5*sqrt(E^2 + W^2)*T).^2)/(E^2 + W^2);

% Determine the steady state
% Reduced row echelon form
%Augmented Coefficient matrix for linear system, 4 variables, 5 quations
Aug = zeros(5, 5);
Aug(1:4, 1:4) = M;
% Trace equal to one
Aug(5, 1) = 1;
Aug(5, 4) = 1;
Aug(5, 5) = 1;
AugRREF = rref(Aug);
% Extract last column (except for the last entry, which shold be zero)
RhoSteady = AugRREF(1:4, 5);

% Plot the probability of remaining in the |1> state -
% along with the analytical solution for Gamma = 0 and the steady state
figure(1)
% Numerical solution, with decay
plot(T, Rho(:,4), 'k-', 'linewidth', 2)
hold on
% Analytical solution without decay
plot(T, AnalyticalNoDecay, 'b-.', 'linewidth', 1.5)
% Steady state solution
yline(RhoSteady(4),'r-');
hold off
grid on
xlabel('Time')
ylabel('Probability')
legend('With decay', 'Without decay', 'Steady state')
set(gca, 'fontsize', 15)