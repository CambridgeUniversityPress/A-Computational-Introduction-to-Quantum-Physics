% This script simulates the time-evolution of the wave packet for a 
% particle trapped in a harmonic potential. The initial state is fixed by
% more or less randomly select the coefficients in a linear combination 
% of theoin the eigen states of the Hamiltonian.
%
% The particle is assumed to have unit mass.
%
% Numerical input parameters: 
% Ttotal    - the duration of the simulation
% dt        - numerical time step, here it serves to tune the speed of the
% simulation
% N         - number of grid points, should be 2^n
% L         - the size of the numerical domain; it extends from -L/2 to L/2
% 
% Physical input parameters:
% Avector   - the set of coefficients defining the initial state 
% Kpot      - strength of the harmonic potential
% 
% All input parameters are hard coded initially.
% 

clear all
format short e

% Numerical time parameters:
Ttotal = 30;
dt = 0.05;

% Grid parameters
L = 30;
N = 512;              % For FFT's sake, we should have N=2^n

% Physical parameters:
Kpot = 1;
% Allocate vector with coefficients
Avector = zeros(1, N);
% Assign values to the first few
Avector(1:10) = [4 1 .1 2 1 .7 .3 .5 2 1];
% Ensure normalization
Avector = Avector/sum(abs(Avector).^2);

% Set up the grid.
x = linspace(-L/2, L/2, N)';
h = L/(N+1);
wavenumFFT = 2*(pi/L)*1i*[(0:N/2-1), (-N/2:-1)]';       % Momentum vector, FFT

% Set up Hamiltonian
% Kinetic energy
T = ifft(-0.5*diag(wavenumFFT.^2)*fft(eye(N)));
% Potential energy
V = diag(Kpot/2*x.^2);

% Total Hamiltonian
H = T+V;

% Diagonalize to find eigen states and energies
[EigVectors EigValues] = eig(H);
% Extract eigen values from diagonal matrix
EigValues = diag(EigValues);
% Sort them
[EigValues, Indexes] = sort(EigValues);
EigVectors = EigVectors(:,Indexes);
% Normalize eigen vectors
EigVectors = EigVectors/sqrt(h);
% Check and correct sign og eigen states
for n=1:N
  if abs(min((x>0).*EigVectors(:,n))) > max((x>0).*EigVectors(:,n))
    EigVectors(:,n) = -EigVectors(:,n);
  end
end

%
% Construct initial condition
%
% Wave function, linear combination of eigenstates with coefficients given
% in Avector
Psi = EigVectors*Avector.';

% Initiate time
t=0;

% Create plot
figure(1)
MaxVal=max(abs(Psi).^2);
pl1 = plot(x,abs(Psi).^2,'k-', 'linewidth', 2);         % Plot wave packet
% Adjust axes
axis([-L/2 L/2 0 1.5*MaxVal]);
set(gca, 'fontsize', 15)
xlabel('x')

%
% Propagate
%
ProgressOld=0;

while t < Ttotal
  % Update time
  t=t+dt;

  % Propagation
  Psi=EigVectors*(Avector.'.*exp(-1i*EigValues*t));

  % Plot wave function and classical position on the fly
  set(pl1, 'ydata', abs(Psi).^2);
  axis([-L/2 L/2 0 1.5*MaxVal])
  drawnow
end