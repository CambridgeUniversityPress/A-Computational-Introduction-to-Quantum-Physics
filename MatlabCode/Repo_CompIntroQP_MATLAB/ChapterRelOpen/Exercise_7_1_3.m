% This script determines the eigen energies for a quantum particle
% with a rectuangular-like confining potential. It does so by setting up 
% both the non-relativistic Schrödigner Hamiltonian and the relativistic 
% Dirac Hamiltonian numerically and diagonalizing them.
%
% It uses the Fast Fourier transform for estimating the kinetic energy.
% A comparison between relativistic and non-relativistic bound states is
% plotted - in addition to the full relativistic spectrum. The latter 
% consist of both pseudo continuum states and bound states.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
% 
% Input for the confining potential:
%   V0      - The "height" of the potetial (must be negative for a well)
%   w       - The width of the potential
%   s       - Smoothness parameter, rectangular well for s -> infinity
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 20;
N = 2048;                % Should be 2^k, k integer, for FFT's sake

% Input parameters for the potential
V0 = -100;
w = 5;
s = 20;

% Mass (chosen to be mass unit)
m = 1;

% The speed of light (in atomic units)
c = 137;

% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];        % Vector with k-values
% Fourier transform the identity matrix:
IdentityTrans = fft(eye(N));
% Multiply by -ik and (-ik)^2 - corresponding to single and double
% derivative, respectively
Pmat = diag(-1i*k)*IdentityTrans;
Tmat = diag(-k.^2)*IdentityTrans;
% Transform back to x-representation
Pmat = ifft(Pmat);
Tmat = ifft(Tmat);
% Correct prefactors
Pmat = -1i*Pmat;
Tmat = -1/2*Tmat;                   

% Non-relativistic Hamiltonian
HamSchrod = Tmat + diag(Vpot(x));             

%
% Relativistic Hamiltonian
%
HamDirac = zeros(2*N, 2*N);         % Allocate
% Upper left block
HamDirac(1:N, 1:N) = diag(Vpot(x)) + m*c^2*eye(N);
% Upper right block
HamDirac(1:N, (N+1):(2*N)) = c*Pmat;
% Lower left block
HamDirac((N+1):(2*N), 1:N) = c*Pmat;
% Lower right block
HamDirac((N+1):(2*N), (N+1):(2*N)) = diag(Vpot(x)) - m*c^2*eye(N);

% Diagonalization
Eschrod= eig(HamSchrod);
Edirac = eig(HamDirac);
% Sort in ascending order
Eschrod = sort(real(Eschrod));
Edirac = sort(real(Edirac));

% Plot relativistic eigen energies
figure(1)
plot(Edirac, 'ro', 'markersize', 3)
hold on
yline(c^2, 'k--');
yline(-c^2, 'k--');
hold off
grid on
xlabel('Index')
ylabel('Energy [a.u.]')
set(gca, 'fontsize', 15)

% Plot bound states
% Non-relativistic
BoundIndicesSchrod = find(Eschrod < 0);
% Relativistic (positive but less than the mass energy)
BoundIndicesDirac = find(Edirac > -m*c^2 & Edirac < m*c^2);

% Plot which compares bound states (mass energy subtracted)
figure(2)
plot(1:length(BoundIndicesSchrod), Eschrod(BoundIndicesSchrod), ...
    'b.', 'markersize', 15) 
hold on
plot(1:length(BoundIndicesDirac), Edirac(BoundIndicesDirac)-m*c^2, ...
    'ro', 'markersize', 5, 'linewidth', 2) 
hold off
grid on
xlabel('Index')
ylabel('Energy [a.u.]')
%legend('Non-relativistic', 'Relativistic', 'location', 'southeast')
set(gca, 'fontsize', 15)