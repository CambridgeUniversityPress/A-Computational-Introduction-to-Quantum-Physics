% This script determines the ground state for a specific potential 
% by means of propagation in inmaginary time.
% The potential barrier has a "smoothly rectangular" shape.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   Tfinal  - The duration of the simulation - in imaginary time
%   dt      - The step size in imaginary time
%
% Input for the potential:
%   V0      - The "height" of the barrier (must be negative)
%   w       - The width of the barrier
%   s       - Smoothness parameter
%   Vpot    - The potential (function variable)
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 20;
N = 512;                % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Numerical time parameters
Tfinal = 10;
dt = 1e-2;

% Input parameters for the barrier
V0 = -3;
w = 8;
s = 5;

% The potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;            % Correct prefactor


% Set up propagator for the approximation
Ham = Tmat_FFT + diag(Vpot(x));         % Hamiltonian
U = expm(-Ham*dt);                      % Propagator

% Set up initial state - just a random "wave function"
Psi0 = rand(N, 1);
InitialNorm = trapz(x, abs(Psi0).^2);
% Impose proper norm
Psi0 = Psi0/sqrt(InitialNorm);

% Initiate plots
figure(1)
pl = plot(x,abs(Psi0).^2, 'k-', 'linewidth', 1.5);
set(gca, 'fontsize', 15)
xlabel('x')
ylabel('|\Psi|^2')

% Initiate wave functons and "time"
Psi = Psi0;
t = 0;

% Initiate/allocate vectors for plotting
TimeVector = 0:dt:Tfinal;
EnergyVector = zeros(1, length(TimeVector));

% Loop which updates wave functions and plots in time
index = 1;
for t = TimeVector  
  % Update time
  t = t+dt;             
  % Update numerical approximation
  Psi = U*Psi;
  
  % Normalize again
  N2 = trapz(x, abs(Psi).^2);
  Energy = -1/(2*dt)*log(N2);
  EnergyVector(index) = Energy;  % Energy estimate
  Psi = Psi/sqrt(N2);
  
  % Update plot
  set(pl, 'ydata', abs(Psi).^2);
  drawnow
  
  % Update index
  index = index + 1;
end


% Determine actual (numerical) ground state
% Diagonalize to find eigen states and energies
[EigVectors EigValues] = eig(Ham);
% Extract eigen values from diagonal matrix
EigValues = diag(EigValues);
% Sort them
[EigValues, Indexes] = sort(real(EigValues));
EigVectors = EigVectors(:,Indexes);
% "Exact" groundState
GroundStateEnergy = EigValues(1);
GroundStateWF = EigVectors(:,1)/sqrt(h);  % Normalize

% Plot energy as a function of "time"
figure(2)
plot(TimeVector, EnergyVector, 'k-', 'linewidth', 2)
hold on
yline(GroundStateEnergy, 'r--', 'linewidth', 2);
hold off
grid on
set(gca, 'fontsize', 15)
xlabel('Imaginary time')
ylabel('Energy')
legend('Estimate', 'True ground state energy')

% Compare wave functions
figure(3)
plot(x, abs(Psi).^2, 'k-', 'linewidth', 2)
hold on
% Plott "exact" wave function
plot(x, abs(GroundStateWF).^2, 'r--', 'linewidth', 2)          
% Fix strength of the potential for plotting
MaxWalPot = -.2*max(abs(Psi)).^2/V0;          
% Plot confining potential
plot(x, Vpot(x)*MaxWalPot, 'b');      
hold off
grid on
set(gca,'fontsize',15)
legend('Estimate','Exact','Potential')

% Write estimates to screen
disp(['Energy from imaginary time: ', num2str(Energy)])
disp(['Energy from direct diagonalization: ', num2str(GroundStateEnergy)])
