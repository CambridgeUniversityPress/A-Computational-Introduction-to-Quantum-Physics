% This script simulates the evolution of an particle which hits a double 
% barrier. Each of the two identical parts of the barrier has a smooth 
% rectangular-like shape.
% 
% The initial wave is a Gaussian with a finite width in momentum and 
% poistion. It is implemented such that you can run it repeatedly with 
% several initial mean momenta - or, equivalently, initial energies - and, 
% in the end, plot the transmission probability as a function of mean 
% energy.
% 
% It imposes a complex absorbing potential and uses the accumulated
% absorption at each end in order to estimate reflection and trasmissiion
% probabilities.
%
% The absorbing potential is a quadratic monomial.
% 
% Inputs for the initial Gaussian:
%   x0      - The (mean) initial position
%   sigmaP  - The momentum width of the initial, Gaussian wave packet
%   tau     - The time at which the Gaussian is narrowest (spatially)
%   E0min   - The minimal initial enwergy
%   E0max   - The maximal initial energy
%   dE0     - The energy step size
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
N = 1024;                
h = L/(N-1);

% Numerical time parameter
dt = 0.1;

% Input parameters for the Gaussian
x0 = -20;
sigmaP = 0.1;
tau = 0;

% Input parameters for the barrier
V0 = 4;
w = 0.5;
s = 25;
d = 3;

% Energies to calculate
dE0 = 0.1;
E0min = 0.05;
E0max = V0;

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

% Set up the absorbing potential - to the left and the right
Vabs = @(x) eta * (abs(x) > Onset) .* (abs(x) - Onset).^2;
VabsL = @(x) eta * (-x > Onset) .* (-x - Onset).^2;
VabsR = @(x) eta * (x > Onset) .* (x - Onset).^2;

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


% Loop over various initial energies and estimate transmission 
% probabilites
% Vector with energies
E0vector = E0min:dE0:E0max;
Tvector = zeros(1, length(E0vector));       % Allocate
index = 1;
for E0 = E0vector
  % Estimate T
  Tprob = TransmissionProb(E0, U, x0, sigmaP, tau, x, dt, VabsL, VabsR);
  Tvector(index) = Tprob;
  
  % Print result to screen
  disp(['E and T: ', num2str([E0 Tprob])])  
  
  % Update index
  index = index + 1;
end

% Plot the result - T as a function of E0
figure(1)
plot(E0vector, Tvector, 'k-', 'linewidth', 2)
grid on
xlabel('Initial energy')
ylabel('Tranmission probability')
set(gca, 'fontsize', 15)

% Function which determines transmission probability as a function of
% initial energy
function Tprob = TransmissionProb(E0, U, x0, sigmaP, tau, x, ...
    dt, VabsL, VabsR)

% Initial mean momentum
p0 = sqrt(2*E0);

% Initial Gaussian
% Set up Gaussian - analytically
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
% Initial Gaussian
Psi = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);

% Initiate probabilities
Rprob = 0;
Tprob = 0;

% Loop which updates wave functions and plots in time
while Rprob + Tprob < .995
  % Update numerical wave function
  Psi = U*Psi;
  
  % Update R and T
  Rprob = Rprob + 2*dt*trapz(x, VabsL(x).*abs(Psi).^2);
  Tprob = Tprob + 2*dt*trapz(x, VabsR(x).*abs(Psi).^2);
end

% End function
end
