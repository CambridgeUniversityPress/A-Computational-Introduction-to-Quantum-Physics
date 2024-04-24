% This script simulates the evolution of an particle which hits a barrier. 
% It does so by simulating it both as a classical process and as quantum
% physical prociess - solving Newtons 2nd law and the Schrödinger equation,
% respectively.
% The classical initial momentum and position are taken as initial mean
% values in the quantum physical case.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   Tfinal  - The duration of the simulation
%   dt      - The step size in time
%
% Inputs for the initial Gaussian:
%   x0      - The (mean) initial position
%   p0      - The (mean) initial momentum
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
L = 200;
N = 2048;                % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Numerical time parameters
Tfinal = 60;
dt = 0.025;

% Input parameters for the Gaussian
x0 = -20;
p0 = 1;
sigmaP = 0.2;
tau = 0;

% Input parameters for the barrier
V0 = 1;
w = 2;
s = 5;

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up barrier
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);
% The derivative of the barrier (finite difference formula)
VpotDeriv = @(x) (Vpot(x+h) - Vpot(x-h))/(2*h);

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];        % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;               % Correct prefactor


% Set up propagator for the approximation
Ham = Tmat_FFT + diag(Vpot(x));         % Hamiltonian
U = expm(-1i*Ham*dt);                   % Propagator

% Set up Gaussian - analytically
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
% Initial Gaussian
Psi0 = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);

% Initiate wave functon, classical variables and time
t = 0;
Psi = Psi0;
xCl = x0;
pCl = p0;

% Initiate plots
figure(1)
% Plot wave packet
plWF = plot(x,abs(Psi).^2, 'k-', 'linewidth', 1.5);
MaxValPsi0 = max(abs(Psi).^2);     % For scaling the barrier plot
hold on
% Plot potential
plot(x, 0.4*Vpot(x)/V0*MaxValPsi0, 'r-', 'linewidth', 2)
% Plot classical position
plClassical = plot(xCl, 0, 'b*', 'linewidth', 5);
axis([-L/2 L/2 0 1.5*MaxValPsi0])                % Set axes
xlabel('Position x')
set(gca, 'fontsize', 15)
hold off

% Loop which updates wave functions and plots in time
nTimes = round(Tfinal/dt);
for n = 1:nTimes  
  % Update numerical wave function
  Psi = U*Psi;
  
  % Update classical position and momentum
  xClOld = xCl;     % Copy position
  pClOld = pCl;     % Copy momentum
  xCl = xClOld + pClOld*dt - 1/2*VpotDeriv(xClOld)*dt^2;
  pCl = pClOld - VpotDeriv(xClOld+pClOld*dt/2)*dt;
  
  % Update plot
  set(plWF, 'ydata', abs(Psi).^2);
  set(plClassical, 'xdata', xCl)
  drawnow
  
  % Update time
  t = t+dt;             
end

% Estimate transmission and relflection probabilities
T = trapz(x((N/2+1):N), abs(Psi((N/2+1):N)).^2);
R = trapz(x(1:N/2), abs(Psi(1:N/2)).^2);
% Print result to screen
disp(['Estimated reflection probability: ', num2str(100*R), ' %'])
disp(['Estimated transmission probability: ', num2str(100*T), ' %'])

% Conclusion for the classicle particle
if xCl < 0
  disp('The classical particle was reflected.')
else
  disp('The classical particle was transmitted.')
end