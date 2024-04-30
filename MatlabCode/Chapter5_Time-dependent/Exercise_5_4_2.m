% This script estimates the ground state energy for a two-particle 
% system consisting of two identical fermions. The particles interacts
% via a smooth Coulomb-like interaction, and they are confined by a 
% smoothly rectangular potential 
% 
% We determine the ground state energy by propagation in imaginary time.
% The implementation also has the option of finding the lowest energy 
% for an exchange anti-symmetric state.
% 
% The one-particle Hamiltonian is estimated numerically 
% using an FFT representation of the kinetic energy.
%
% 
% Numerical inputs:
% L         - size of domain 
% N         - number of grid points (should be 2^n)
% dt        - step size in "time"-propagation
% Tmax      - duration of simulation
% 
%
% Physical inputs:
% V0        - the height of the potential (should be negative)
% w         - the width of the potential
% s         - the "smoothness" of the potential 
% W0        - the strength of the interaction
% Xsymm     - the exchange symmetry, should be +1 or -1
%
% All parameters and functions are hard coded initially

% Potential:
V0 = -1;
w = 4;
s = 5;

% Interaction strength
W0 = 1;

% Excange symmetry
Xsymm = +1;

% Grid parameters
L = 10;
N = 128;              

% "Time" parameters
dt = 1e-3;
Tmax = 10;

% The potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% The interaction
Wint = @(x1, x2) W0./sqrt((x1-x2).^2+1);

% Set up the grids.
x = linspace(-L/2,L/2,N);
x = x.';                            % Transpose
h = L/(N-1);
% Two-particle grid
[X1 X2] = meshgrid(x.', x.');

% Matrix with interaction
WintMat = Wint(X1, X2);

% Set up kinetic energy matrix by means of the fast Fourier transform
k = 2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat = fft(eye(N));
% Multiply by (ik)^2
Tmat = diag(-k.^2)*Tmat;
% Transform back to x-representation
Tmat = ifft(Tmat);
Tmat = -1/2*Tmat;            % Correct prefactor

% Potential energy
V = diag(Vpot(x));
% Total one-particle Hamiltonian
H_OnePart = Tmat + V;

% Initiate Wave function
Psi = rand(N, N);
% Impose the symmetry in question
Psi = 0.5*(Psi + Xsymm*Psi.');
% Normalize
NormSq = h^2*sum(sum(abs(Psi).^2));
Psi = Psi/sqrt(NormSq);

% Initiate time variables
t = 0;
Tvec = 0:dt:Tmax;
Evec = 0*Tvec;
index = 1;

% "Time"-loop
while t < Tmax
  % Hamiltonian acting on the wave function
  HamPsi = H_OnePart*Psi + Psi*H_OnePart + WintMat.*Psi;
  
  % Take a step in time
  Psi = Psi - HamPsi*dt;
  
  % Enfocrce correct symmetry
  Psi = 0.5*(Psi + Xsymm*Psi.');
  
  % Renormalize and estimate energy
  NormSq = h^2*sum(sum(abs(Psi).^2));
  Psi = Psi/sqrt(NormSq);
  Energy = 1/dt*(1-sqrt(NormSq));
  Evec(index) = Energy;
  
  % Update time and index
  t = t + dt;
  index = index + 1;
end

% Write energy to screen
disp(['Resulting energy: ', num2str(Energy)])

% Plot final wave fuction
figure(1)
pcolor(x, x, Psi)
shading interp
xlabel('x_1')
ylabel('x_2')
set(gca, 'fontsize', 15)
axis square
colorbar

% Plot convergence
figure(2)
plot(Tvec, Evec, 'k-', 'linewidth', 2)
grid on
xlabel('Imaginary time')
ylabel('Energy')
set(gca, 'fontsize', 15)
axis([0 Tmax -2 2])