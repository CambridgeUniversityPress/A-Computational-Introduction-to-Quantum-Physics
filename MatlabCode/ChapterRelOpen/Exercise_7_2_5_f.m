% This implementation investigates how amplitude damping has a detrimental 
% effect on the implementation of the NOT gate. Within a model with 
% degenerate states with constant coupling W, the NOT gate may be 
% implemented by tuning the duration such that T = pi \hbar/W.
% With a finite decay rate Gamma, it is no longer that simple.
%
% The script solves the GKLS equation for this model and determines the 
% probability of measuring 1 with 0 as the initial state. It does so for
% various values of the decay rate Gamma
%
% The inputs are
% W     - the coupling (in the Hamiltonian), taken to be real
% GammaMax     - the maximal decay rate (amplitude damping)
% GammaStep    - the step size for Gamma
%
% All inputs are hard coded initially.

% Coupling strength
W = 1;

% Gamma parameters
GammaMax = 1.5;
GammaStep = 1e-3;

% Final time
Tfinal = pi/W;

% Gamma vector
GammaVector = 0:GammaStep:GammaMax;

% Matrix with couplings
Coupl = W/2*[0, -1, 1, 0;
            -1, 0, 0, 1; 
            1, 0, 0, -1;
            0, 1, -1, 0];
  
% Initial state (a pure |0> state)
x0 = [1; 0; 0; 0];

% Allocate fidelity vector and index
FidelityVector = zeros(1, length(GammaVector));
index = 1;

% Loop over gamma
for Gamma = GammaVector
  % Add decay (in the Lindbladian)
  M = Coupl + 1i*Gamma/2*[0, 0, 0, 2;
                    0, -1, 0, 0;
                    0, 0, -1, 0;
                    0, 0, 0, -2];

  % Right and side of the dyamical equation 
  F = @(t,x) -1i*M*x;

  % Solve using built-in function (Runge-Kutta)
  [T Rho]=ode45(F, [0 Tfinal], x0);         % Solves the TDSE

  % Extract the probability of measuring 1
  Rho11 = Rho(end, 4);
  FidelityVector(index) = Rho11;
  index = index + 1;
end

% Plot result
figure(1)
plot(GammaVector, FidelityVector, 'k-', 'linewidth', 2)
xlabel('Decay rate, \Gamma')
ylabel('Flip probability')
set(gca, 'fontsize', 15)
grid on
axis([0 GammaMax 0 1])