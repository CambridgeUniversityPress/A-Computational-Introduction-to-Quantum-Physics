% This script simulates the evolution of an intially Gaussian wave packet 
% placed between two barriers. The barriers have a "smoothly rectangular" 
% shape. In addition to simulating the evolution of the wave packet, the 
% script estimates the probability of remaining between the barriers.
%
% As we go aloing, the evolution of the wave packet is shown - with a 
% logarithmic y-axis.
% 
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   Tfinal  - The duration of the simulation
%   dt      - The step size in time
%
% Inputs for the initial Gaussian:
%   sigmaP  - The momentum width of the initial, Gaussian wave packet
% 
% Input for the barriers:
%   V0      - The height of the barriers (can be negative)
%   w       - The width of the barriers
%   s       - Smoothness parameter
%   d       - Distance between barriers (centre to centre)
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 200;
N = 2048;                % Should be 2^k, k integer, for FFT's sake

% Numerical time parameters
Tfinal = 150;
dt = 0.05;

% Input parameter for the Gaussian (the rest is fixed)
sigmaP = 0.2;

% Input parameters for the barrier
V0 = 1;
w = 0.5;
s = 25;
d = 10;

% Fixed parameters for the Gaussian
x0 = 0;
p0 = 0;
tau = 0;

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector
h = L/(N-1);

% Set up barrier
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);
VpotDouble = @(x) Vpot(x-d) + Vpot(x+d);

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
Ham = Tmat_FFT + diag(VpotDouble(x));       % Hamiltonian
U = expm(-1i*Ham*dt);                       % Propagator

% Set up Gaussian - analytically
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
% Initial Gaussian
Psi0 = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);

% Initiate plots
figure(1)
% Plot wave function
plFFT = semilogy(x,abs(Psi0).^2, 'k-', 'linewidth', 1.5);
% Plot potential
MaxValPsi0 = max(abs(Psi0).^2);     % For scaling the barrier plot
hold on
semilogy(x, 0.4*VpotDouble(x)/V0*MaxValPsi0, 'r-', 'linewidth', 2)
set(gca, 'fontsize', 15)
xlabel('x')
% Fix window
axis([-L/2 L/2 1e-4 1])
hold off

% Initiate wave functons and time
Psi = Psi0;
t = 0;

% Vector with time and allocation of vector with probabilities
tVector = 0:dt:Tfinal;
Pvector = 0*tVector;
index = 1;

% Loop which updates wave functions and plots in time
while t < Tfinal
  % Update numerical approximations
  Psi = U*Psi;
  
  % Update plot
  set(plFFT, 'ydata', abs(Psi).^2);
  drawnow
  
  % Calculate probability remaining
  Pbetween = trapz(x, (abs(x)<d).*abs(Psi).^2);
  Pvector(index) = Pbetween;
  
  % Update time and index
  t = t+dt;             
  index = index + 1;
end

% Plot probability of reamaning
figure(2)
plot(tVector, Pvector, 'k-', 'linewidth', 2)
set(gca, 'fontsize', 15)
grid on
xlabel('Time')