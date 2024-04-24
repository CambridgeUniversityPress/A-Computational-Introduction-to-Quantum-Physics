% This script simulates the evolution of an intially Gaussian wave packet 
% which hits a barrier. The barrier has a "smoothly rectangular" shape.
% In addition to simulating the evolution of the wave packet, the script
% estimates the transmission and reflection probabilities afther the 
% collision.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   Tfinal  - The duration of the simulation
%   dt      - The step size in time
%
% Inputs for the initial Gaussian:
%   x0      - The mean position of the initial wave packet
%   p0      - The mean momentum of the initial wave packet
%   sigmaP  - The momentum width of the initial, Gaussian wave packet
%   tau     - The time at which the Gaussian is narrowest (spatially)
% 
% Input for the barrier:
%   V0      - The height of the barrier (can be negative)
%   w       - The width of the barrier
%   s       - Smoothness parameter
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 400;
N = 2048;                % Should be 2^k, k integer, for FFT's sake

% Numerical time parameters
Tfinal = 100;
dt = 0.5;

% Input parameters for the Gaussian
x0 = -20;
p0 = 1;
sigmaP = 0.2;
tau = 0;

% Input parameters for the barrier
V0 = 3;
w = 2;
s = 5;

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector
h = L/(N-1);

% Set up barrier
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;           % Correct prefactor


% Set up propagator for the approximation
Ham = Tmat_FFT + diag(Vpot(x));     % Hamiltonian
U = expm(-1i*Ham*dt);               % Propagator

% Set up Gaussian - analytically
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
% Initial Gaussian
Psi0 = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);

% Initiate plots
figure(1)
plFFT = plot(x,abs(Psi0).^2, 'k-', 'linewidth', 1.5);
MaxValPsi0 = max(abs(Psi0).^2);     % For scaling the barrier plot
hold on
plot(x, 0.4*Vpot(x)/V0*MaxValPsi0, 'r-', 'linewidth', 2)
axis([-L/2 L/2 0 1.5*MaxValPsi0])
hold off

% Initiate wave functons and time
Psi = Psi0;
t = 0;

% Loop which updates wave functions and plots in time
while t < Tfinal
  % Update time
  t = t+dt;             
  % Update numerical approximations
  Psi = U*Psi;
  % Update plot
  set(plFFT, 'ydata', abs(Psi).^2);
  drawnow
end

% Estimate transmission and relflection probabilities
T = trapz(x, (x>0) .* abs(Psi).^2);
R = trapz(x, (x<0) .* abs(Psi).^2);
% Print result to screen
disp(['Estimated reflection probability: ', num2str(100*R), ' %'])
disp(['Estimated transmission probability: ', num2str(100*T), ' %'])
