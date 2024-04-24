% This script simulates the evolution of an particle which hits a double 
% barrier. Each of the two identical parts of the barrier has a smooth 
% rectangular-like shape.
% 
% It imposes a complex absorbing potential and uses the accumulated
% absorption at each end in order to estimate reflection and trasmissiion
% probabilities.
%
% The absorbing potential is a quadratic monomial.
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
%   d       - Distance between the barriers (centre to centre)
%
% Inputs for the absorbing potential
%   eta     - The strength of the absorber
%   Onset   - the |x| value beyond which absorption starts
% 
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   dt      - The step size in time
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 100;
N = 1024;                % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Numerical time parameter
dt = 0.1;

% Input parameters for the Gaussian
x0 = -20;
%p0 = 1;
p0 = sqrt(2*3);
sigmaP = 0.1;
tau = 0;

% Input parameters for the barrier
V0 = 4;
w = 0.5;
s = 25;
d = 3;

% Inputs for the absorber
eta = 0.05;
Onset = 40;
% Check that Onset is admissible
if Onset > L/2
  error('This value for Onset is too large.')
end

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up barrier
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);
Vdouble = @(x) Vpot(x-d) + Vpot(x+d);

% Set up the absorbing potential
Vabs = @(x) eta * (abs(x) > Onset) .* (abs(x) - Onset).^2;

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];        % Vector with k-values
% Fourier transform the identity matrix:
Tmat = fft(eye(N));
% Multiply by (ik)^2
Tmat = diag(-k.^2)*Tmat;
% Transform back to x-representation
Tmat = ifft(Tmat);
Tmat = -1/2*Tmat;               % Correct prefactor


% Set up propagator - with a non-Hermitian Hamiltonian
Ham = Tmat + diag(Vdouble(x)) - 1i*diag(Vabs(x));  % Hamiltonian
U = expm(-1i*Ham*dt);                               % Propagator

% Set up Gaussian - analytically
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
% Initial Gaussian
Psi0 = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);

% Initiate wave functon
Psi = Psi0;

% Initiate plots
figure(1)
% Plot wave packet
plWF = plot(x,abs(Psi).^2, 'k-', 'linewidth', 1.5);
MaxValPsi0 = max(abs(Psi).^2);     % For scaling the barrier plot
hold on
% Plot absorber
plot(x, 0.5*Vabs(x)/max(Vabs(x))*MaxValPsi0, 'r--', 'linewidth', 2)
% Plot potential
plot(x, 0.7*Vdouble(x)/V0*MaxValPsi0, 'b-', 'linewidth', 2)
axis([-L/2 L/2 0 1.2*MaxValPsi0])                % Set axes
xlabel('Position x')
set(gca, 'fontsize', 15)
hold off

% Initiate reflection and transmission probabilities
Rprob = 0;
Tprob = 0;

% Loop which updates wave functions and plots in time
% The limit of 99.5 % is set somewhat arbitrarily; it could be 
% higher and it could be slightly lower
while Rprob + Tprob < .995
  % Update numerical wave function
  Psi = U*Psi;
  
  % Update R and T
  Rprob = Rprob + 2*dt*trapz(x, (x < -Onset).*Vabs(x).*abs(Psi).^2);
  Tprob = Tprob + 2*dt*trapz(x, (x >  Onset).*Vabs(x).*abs(Psi).^2);
  
  % Update plot
  set(plWF, 'ydata', abs(Psi).^2);
  drawnow
end

% Estimate transmission and relflection probabilities
% Print result to screen
disp(['Estimated reflection probability: ', num2str(100*Rprob), ' %'])
disp(['Estimated transmission probability: ', num2str(100*Tprob), ' %'])
