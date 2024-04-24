% This script simulates a system consisting of two spin 1/2-particles 
% exposed to the same magnetic field. From the two-particle state, the 
% reduced density matrix for the first particle is determined, and 
% various quantieis such as its spin projection expectation avlues and 
% purity, are determined from this reduced density matrix.
% 
% The initial state is one in which both particles are eigenstates
% of their respective spin projection operators s_z. The first particle 
% has positive spin projection (spin up) while the other one has negative
% spin projection (spin down).
%
% The external magnetic field has both a static part and a dynamic part. 
% The dynamic part oscillates in time as a sine function. The static 
% field points in the z-direction, thus lifting the degenerecy between 
% spin up and spin down. The oscillating field is taken to point along 
% the x-axis.
%
% The two particle interact thorough the spin-spin interaction.
% 
% From this solution, the time-dependent reduced density matrix for the 
% first particle is calculated - at each end every time.
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
Omega = 0.2;
w = 1.1;

% Spin-spin interaction strength
u = .025;

% Duration of the simulation
OpticalCycles = 15;
Tfinal = OpticalCycles*2*pi/w;

% Set up the equation - with the Hamiltonian
% Static part
H0 = [-E+u 0 0 0; 0 -u 2*u 0; 0 2*u -u 0; 0 0 0 E+u];                  
% Coupling matrix
InteractionMatrix = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0];        
% Right and side of the TDSE
F = @(t,x) -i*(H0+Omega*sin(w*t)*InteractionMatrix)*x;

% Initial state
x0 = [0; 1; 0; 0];          % Up-down

% Solve using built-in function (Runge-Kutta)
[T X]=ode45(F, [0 Tfinal], x0);         % Solves the TDSE

% Spin projection operators (Pauli matrices times \hbar/2, where \hbar = 1)
Sx = 1/2*[0 1; 1 0];
Sy = 1/2*[0 -1i; 1i 0];
Sz = 1/2*[1 0; 0 -1];

% Allocate
lenT = length(T);
SpinUpA = zeros(1, lenT);
MeanSpinA_x = zeros(1, lenT);
MeanSpinA_y = zeros(1, lenT);
MeanSpinA_z = zeros(1, lenT);
OffDiagonalA = zeros(1, lenT);
PurityA = zeros(1, lenT);

% Loop over the time vector T and calculate quantities from the reduced 
% density matrix
for index = 1:lenT
  % Amplitudes
  a = X(index, 1);
  b = X(index, 2);
  c = X(index, 3);
  d = X(index, 4);
  % rhoA
  rhoA = [abs(a)^2 + abs(b)^2, a*conj(c) + b*conj(d);
      conj(a)*c + conj(b)*d, abs(c)^2 + abs(d)^2];
  % Calculate the various quantities from rhoA
  SpinUpA(index) = rhoA(1,1);
  MeanSpinA_x(index) = trace(Sx*rhoA);
  MeanSpinA_y(index) = trace(Sy*rhoA);
  MeanSpinA_z(index) = trace(Sz*rhoA);
  OffDiagonalA(index) = abs(rhoA(1,2));
  PurityA(index) = trace(rhoA^2);
end

% Plot results

% Spin-up probability, size of off-diagonal elements and purity
figure(1)
% Spin up
plot(T, SpinUpA, 'b-.', 'linewidth', 2)
hold on
% Off-diagonal element
plot(T, OffDiagonalA, 'k--', 'linewidth', 2)
% Purity
plot(T, PurityA, 'r-', 'linewidth', 2)
hold off
set(gca, 'fontsize', 15)
xlabel('Time')
legend('P_\uparrow', '|\rho_{0,1}|', '\gamma(t)')
grid on
set(gca, 'fontsize', 15)
xlabel('Time')
grid on

% Plot the spin expectation values - along each axis
figure(2)
plot(T, MeanSpinA_x, 'k-', 'linewidth', 2)
hold on
plot(T, MeanSpinA_y, 'r--', 'linewidth', 2)
plot(T, MeanSpinA_z, 'b-.', 'linewidth', 2)
hold off
set(gca, 'fontsize', 15)
xlabel('Time')
ylabel('Spin projection')
legend('<s_x>', '<s_y>','<s_z>')
grid on