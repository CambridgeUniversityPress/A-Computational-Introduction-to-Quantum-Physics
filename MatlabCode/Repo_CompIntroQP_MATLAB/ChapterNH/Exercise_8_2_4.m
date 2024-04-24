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
%   Sigma   - The width of the potetial
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 20;
N = 512;                % Should be 2^k, k integer, for FFT's sake

% Input parameters for the potential
V0 = -1;
Sigma = sqrt(2);

% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up potential
Vpot = @(x) V0*exp(-x.^2/(2*Sigma^2));

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];        % Vector with k-values
% Fourier transform the identity matrix:
Tmat = fft(eye(N));
% Multiply by (ik)^2
Tmat = diag(-k.^2)*Tmat;
% Transform back to x-representation
Tmat = ifft(Tmat);
Tmat = -1/2*Tmat;                   % Correct prefactor

% Full Hamiltonian
Ham = Tmat + diag(Vpot(x));             % Hamiltonian

% Diagonalization
[U E] = eig(Ham);
% Extract eigen energies, the diagonal of E
E = real(diag(E));
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
  plot(x, U(:,n))
  title(['Energy index: ', num2str(n), ', energy: ', num2str(E(n))])
  % Update n
  n = n+1;
end