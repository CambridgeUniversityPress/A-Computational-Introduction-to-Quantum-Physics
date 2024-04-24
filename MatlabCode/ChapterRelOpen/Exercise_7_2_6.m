% This script simulates the evolution of an intially Gaussian wave 
% passing a well. In addition to scattering, the simulation features
% the possibility of capturing the incident particle in the ground 
% state. This comes about via a jumb operator of Lindblad form.
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
% Input for the capture model:
%   Gamma0  - the over all strength of the decay rate
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 400;
N = 1024;                % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Numerical time parameters
Tfinal = 50;
dt = 0.05;

% Input parameters for the Gaussian
x0 = -20;
p0 = 1;
sigmaP = 0.2;
tau = -x0/p0;

% Input parameters for the barrier
V0 = -1;
w = 2;
s = 5;

% Capture rate
Gamma0 = 0.1;

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up well (V0 should be negative)
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat = fft(eye(N));
% Multiply by (ik)^2
Tmat = diag(-k.^2)*Tmat;
% Transform back to x-representation
Tmat = ifft(Tmat);
Tmat = -1/2*Tmat;            % Correct prefactor
% Total Hamiltonian
Ham = Tmat + diag(Vpot(x));    % Hamiltonian

% Diagonalize Hamiltonian
[B E] = eig(Ham);               % Diagonalization
E = diag(E);                    % Extract diagonal energies
[E SortInd] = sort(E);          % Sort energies
B = B(:,SortInd)/sqrt(h);       % Sort eigenvectors and normalize
% Ground state
GroundState = B(:, 1);
disp(['Ground state energy: ',num2str(E(1)),'.'])         
Nbound = length(find(E<0));     % The number of bound states
disp(['The potential supports ',num2str(Nbound),' bound state(s).'])

% Construct non-Hermitian contribution to effective Hamiltonian
% Vector with couplings, <\phi_l | x | \phi_0>
ProjectVect = h*B'*diag(x)*B(:,1); 
ProjectVect(1) = 0;                              % No decay from ground state
% Matrix with Gamma coefficients
GammaKL = Gamma0*ProjectVect*ProjectVect';       
% Anti-Hermitian "interaction matrix"
HamNH = h/2*B*GammaKL*B';

% Total, Non-Hermitian Hamiltonian:
HamTot = Ham - 1i*HamNH;
% Non-unitary propagator
U = expm(-1i*HamTot*dt);                   

% Set up intial Gaussian - analytically
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
Psi0 = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);

% Initiate plots
figure(1)
plWF = plot(x,abs(Psi0).^2, 'k-', 'linewidth', 1.5);
MaxValPsi0 = max(abs(Psi0).^2);     % For scaling the barrier plot
hold on
ScalingFactor = 0.5*MaxValPsi0/abs(V0);
plot(x, ScalingFactor*Vpot(x), 'r-', 'linewidth', 2)
axis([-50 50  -1.2*ScalingFactor 1.5*MaxValPsi0])
hold off
xlabel('x')
ylabel('Particle density')
set(gca, 'fontsize', 15)

% Initiate and allocate
Psi = Psi0;
NstepTime = floor(Tfinal/dt);
Tvec = 0:dt:((NstepTime-1)*dt);
rho00Vec = zeros(1, length(Tvec));
rho00 = 0;

% Loop which updates wave functions and plots in time
for timeIndX = 1:NstepTime
  % Update wave function
  Psi = U*Psi;

  % Update norm
  Norm = trapz(x, abs(Psi).^2);
  % Update gound state population
  rho00 = rho00 + dt*2*h*Psi'*HamNH*Psi;
  rho00Vec(timeIndX) = rho00;
  
  % Update plot
  set(plWF, 'ydata', abs(Psi).^2 + ...
      (1-Norm)*abs(GroundState).^2);
  drawnow
end

figure(2)
plot(Tvec, real(rho00Vec), 'k-', 'linewidth', 1)
xlabel('Time')
ylabel('Capture probability')
grid on
set(gca, 'fontsize', 15)