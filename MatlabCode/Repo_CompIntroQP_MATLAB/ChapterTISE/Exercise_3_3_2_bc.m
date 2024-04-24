% This script simulates the time-evolution of the wave packet for a 
% particle trapped in a harmonic potential in the special case that it is
% in a Glauber state, or a Coherent state as it is also called.
%
% The script also simulate the evolution of a classical particle with the 
% corresponding initial condition.
%
% In order to determine the truncation index for the expansion in eigen 
% states, a plot which compares numerical and analytical eigenenergies is 
% displayed; the truncation should be set such that the these coincide up 
% to the truncation index.
%
% The particle is assumed to have unit mass.
%
% Numerical input parameters: 
% Ttotal - the duration of the simulation
% dt     - numerical time step
% N      - number of grid points, should be 2^n
% L      - the size of the numerical domain; it extends from -L/2 to L/2
% Ntrunc - the number of states to include in our basis 
%
% Physical input parameters:
% alpha  - the complex parameter specifying the Glauber state
% Kpot   - strength of the harmonic potential
% 
% All input parameters, except Ntrunc, are hard-coded initially.

clear all
format short e

% Physical parameters:
alpha = 2-i;
Kpot = 1;

% Numerical time parameters:
Ttotal = 30;
dt = 0.05;

% Grid parameters
L = 30;
N = 512;              % For FFT's sake, we should have N=2^n


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

% Plot eigen values - along with the analytical ones
AnalyticalEigValues = (0.5+[0:(N-1)])*sqrt(Kpot);
figure(1)
plot(EigValues,'rx')
hold on
plot(AnalyticalEigValues, 'ko')
hold off
xlabel('Quantum numer n')
ylabel('Energy')

% Fix truncation - takes input from the screen
Ntrunc = input('Where should we truncate the numerical energies? ');

%
% Construct initial condition
%
% Allocate vector with coefficients
Avector = zeros(1,N);
Nvector = [0:(Ntrunc-1)];
Avector(1:Ntrunc) = exp(-abs(alpha)^2/2)*alpha.^(Nvector)./...
      sqrt(factorial(Nvector));

% Initiate wave function, classical position and momentum and time
Psi = EigVectors*Avector.';
% Classical initial condition
xCl = real(alpha)*sqrt(2);
pCl = imag(alpha)*sqrt(2);
t=0;

% Create plot
figure(2)
MaxVal=max(abs(Psi).^2);
% Plot wave packet
pl1 = plot(x, abs(Psi).^2, 'k-', 'linewidth', 2);         
hold on
% Plot classical position
pl2 = plot(xCl, 0, 'r*', 'linewidth', 2);
hold off
% Adjust axes
axis([-L/2 L/2 0 1.5*MaxVal]);
xlabel('Position x')
ylabel('|\Psi(x; t)|^2')
%
% Propagate
%
ProgressOld=0;

while t < Ttotal
  % Update time
  t=t+dt;

  % Propagation, quantum physics. Given by simple phase factors
  Psi=EigVectors*(Avector.'.*exp(-1i*EigValues*t));

  % Propagation, classical (Heuns method)
  xClOld = xCl;
  pClOld = pCl;
  Force = -Kpot*xCl;
  xCl = xClOld + pClOld*dt - 1/2*Kpot*xClOld*dt^2;
  pCl = pClOld - Kpot*(xClOld+pClOld*dt/2)*dt;
  
  % Plot wave function and classical position on the fly
  set(pl1, 'ydata', abs(Psi).^2);
  set(pl2, 'xdata', xCl)
  axis([-L/2 L/2 0 1.5*MaxVal])
  drawnow
end