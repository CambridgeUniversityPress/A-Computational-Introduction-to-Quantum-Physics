% This script sets out to fix the duration of the spin-spin interaction 
% between to spin 1/2-particles so that it, in effect, implements the 
% SWAP gate. The parameters are rigged such that the Hamiltonian assumes a 
% particularly simple form. The only inputs are the upper limit for the 
% interaction time T - and dT, the resolution in the vector with 
% time-durations.
% 
% It plots the cost function
% C(T) = 1 - |1/4 * Tr(U_target^\dagger U(T))|^2
% as a function of duration T; when C=0, our gate coincides
% with the target gate, and the fidelity is 100%.

% Inputs
Tmax = 3;
dT = 0.01;

% Vector with durations
Tvector = 0:dT:Tmax;

% Hamiltonian
H = [1 0 0 0; 0 -1 2 0; 0 2 -1 0; 0 0 0 1];
% Gate of one step
UdT = expm(-i*H*dT);

% Target gate
Uswap = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1];

% Initiate variables
T = 0;
U = eye(4);         
index = 1;
InFidelity = zeros(1,floor(Tmax/dT));
% 
% Loop over durations T
while T < Tmax
  InFidelity(index) = 1 - abs(1/4*trace(Uswap'*U))^2;
  % Update gate, duration and index
  U = UdT*U;
  T = T+dT;
  index = index+1;
end

% Plot cost function
figure(1)
plot(Tvector, InFidelity, 'k-', 'linewidth', 2)
hold on
% Indicate exact zero-points
xline(pi/4, 'r--', 'linewidth', 1.5);
xline(3*pi/4, 'r--', 'linewidth', 1.5);
hold off
% Labels, fontsize and grid
xlabel('Interaction duration T')
ylabel('C(T)')
grid on
set(gca, 'fontsize', 15)
