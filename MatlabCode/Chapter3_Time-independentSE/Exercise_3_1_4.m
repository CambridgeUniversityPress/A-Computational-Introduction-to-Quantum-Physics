% This script determines the energies for a quantum particle of unit mass
% confined within a harmonic oscillator potential. It does so by setting 
% up the Hamiltonian, numerically, and diagonalizing it.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
% 
% Input for the harmonic potential:
%   kStrength - the strength of the potential
%
% Inputs for plotting
%   NplotE  - The number of energies to display
%   NplotWF - The number of wave functions to display
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 15;
N = 256;                % Should be 2^k, k integer, for FFT's sake

% Input parameter for the potential
kStrength = 1;

% Number of plots
NplotE = 50;
NplotWF = 3;

% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up barrier
Vpot = @(x) 1/2*kStrength*x.^2;

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;            % Correct prefactor

% Full Hamiltonian
Ham = Tmat_FFT + diag(Vpot(x));         % Hamiltonian

% Diagonalization
[U E] = eig(Ham);
% Extract eigen energies, the diagonal of E
E = diag(E);
% Sort eigenvalues - and rearrange eigenvectors accordingly
[E Ind] = sort(real(E));
U = U(:, Ind);
% Ensure proper normalization
U = U/sqrt(h);

% Plott some of the bound energies together with the analytical formula
figure(1)
for n = 1:min(NplotE, N)
  % Numerical values
  plot(1:NplotE, E(1:NplotE), 'rx', 'linewidth', 2)
  % Analytical values
  hold on
  plot(1:NplotE, sqrt(kStrength)*([0:(NplotE-1)]+1/2), 'ko', 'linewidth', 2)
  hold off
end
grid on
set(gca, 'fontsize', 15)
xlabel('n quantum number')
ylabel('Energy')
legend('Numerical eigenenergies', 'Exact eigenenergies', 'location', 'northwest')

% Plot some of the wave functions
figure(2)
for n = 1:min(N,NplotWF)
  % Write energy to screen
  plot(x, U(:,n), 'k-', 'linewidth', 2)
  title(['Energy index: ', num2str(n), ', energy: ', num2str(E(n))])
  set(gca, 'fontsize', 12)
  xlabel('r [a.u.]')
  ylabel('Wave function')
  grid on
  pause
end
