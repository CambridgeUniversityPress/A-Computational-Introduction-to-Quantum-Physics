% This script determines the admissible energies for a bound quantum
% particle confined in a rectuangular-like confining potential. It does so
% by setting up the Hamiltonian, numerically, and diagonalizing it.
%
% In addition to determining the bound energies, the script also plot
% the corresponding eigen states.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
% 
% Input for the confining potential:
%   V0      - The "height" of the potential (must be negative for a well)
%   w       - The width of the potetial
%   s       - Smoothness parameter, rectangular well for s -> infinity
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 20;
N = 512;                % Should be 2^k, k integer, for FFT's sake

% Input parameters for the potential
V0 = -4;
w = 5;
s = 100;

% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];        % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;                   % Correct prefactor

% Full Hamiltonian, with potential
Ham = Tmat_FFT + diag(Vpot(x));             % Hamiltonian

% Diagonalization
[U E] = eig(Ham);
% Extract eigen energies, the diagonal of E
E = diag(E);
% Sort eigenvalues - and rearrange eigenvectors accordingly
[E Ind] = sort(E);
U = U(:, Ind);
% Ensure proper normalization
U = U/sqrt(h);

% Write bound energies to screen and plot wave functions
n = 1;
while E(n) < 0
  % Write energy to screen
  disp(['Energy nr. ', num2str(n),': ', num2str(E(n))])
  % Plot wave function
  plot(x, U(:,n), 'k-', 'linewidth', 2)
  title(['Energy index: ', num2str(n), ', energy: ', num2str(E(n))])
  grid on
  set(gca, 'fontsize', 15)
  % Update n
  n = n+1;
  pause
end