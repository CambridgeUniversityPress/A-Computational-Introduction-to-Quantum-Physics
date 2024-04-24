% This script simulates the evolution of two Gaussian wave packets who meet. 
% It does so by solving the Schrödinger equation by approximating the kinetic 
% energy operator using the Fast Fourier transform. The two Gaussians have 
% their own sets of initial mean position, mean momentum, and momentum widths.
% The mean momenta should be such that the two waves travel towards each other, 
% and the mean position and widths should be such that they do not overlap 
% initially.
%
% The simulation displays both |\Psi(x; t)|^2 and [Re (\Psi(x; t))]^2.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   Tfinal  - The duration of the simulation
%   dt      - The step size in time
%
% Physical inputs, wave 1:
%   x01      - The mean position of the initial wave packet
%   p02      - The mean momentum of the initial wave packet
%   sigmaP1  - The momentum width of the initial, Gaussian wave packet
%   tau1     - The time at which the Gaussian is narrowest (spatially)
%
% -Correspondingly for wave 2.    
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 150;
N = 1024;                % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Numerical time parameters
Tfinal = 50;
dt = 0.1;

% Input parameters for the 1st Gaussian
x01 = -20;
p01 = 0.75;
sigmaP1 = 0.2;
tau1 = 0;

% Input parameters for the 2nd Gaussian
x02 = 20;
p02 = -1.25;
sigmaP2 = 0.1;
tau2 = 0;

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

% Set up propagator
U_FFT = expm(-1i*Tmat_FFT*dt);

% Set up Gaussian - analytically
% 1st Gaussian:
InitialNorm1 = nthroot(2/pi, 4) * sqrt(sigmaP1/(1-2i*sigmaP1^2*tau1));
Psi1 = InitialNorm1*exp(-sigmaP1^2*(x-x01).^2/(1-2i*sigmaP1^2*tau1)+ ...
    1i*p01*x);
% 2nd Gaussian:
InitialNorm2 = nthroot(2/pi, 4) * sqrt(sigmaP2/(1-2i*sigmaP2^2*tau2));
Psi2 = InitialNorm2*exp(-sigmaP2^2*(x-x02).^2/(1-2i*sigmaP2^2*tau2)+ ...
    1i*p02*x);
% Total initial wave packet - with proper normalization (as long as they 
% do not overlap)
Psi0 = 1/sqrt(2)*(Psi1 + Psi2);

% Initiate plot
figure(1)
plAbs = plot(x, abs(Psi0).^2, 'k-', 'linewidth', 2);
hold on
plReal = plot(x, real(Psi0).^2, 'b-', 'linewidth', 1);
hold off
set(gca, 'fontsize', 15)
xlabel('x')
axis([-L/2 L/2 0 .15])
% Initiate wave functon and time
Psi_FFT = Psi0;
t = 0;

% Loop which updates wave functions and plots in time
while t < Tfinal
  % Update numerical wave function
  Psi_FFT = U_FFT*Psi_FFT;
  
  % Update plot
  set(plAbs, 'ydata', abs(Psi_FFT).^2);
  set(plReal, 'ydata', real(Psi_FFT).^2);
  drawnow

  % Update time
  t = t+dt;             
end
