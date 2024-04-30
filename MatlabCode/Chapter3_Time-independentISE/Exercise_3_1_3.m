% This script determines the admissible energies for a hydrogen atom.
% It does so by setting up the Hamiltonian, numerically, and 
% diagonalizing it. The kinetic energy operator is estimated using
% a three point finite difference formula. The boundary conditions at 
% r = 0 and r = L, where L is the size of the numerical domain, is 
% included manifestly in our approximation.
%
% In addition to determining the bound energies, the script also plots the
% admissible energies and compares the spectrum with the Bohr formula.
%
% Numerical inputs:
% L       - The extension of the spatial grid 
% N       - The number of grid points
% 
% Nplot   - The number of wave functions to plot
% 
% All inputs are hard coded initially.

% Numerical grid parameters
L = 50;
N = 1000;                

% The number of plots
Nplots = 8;

% Set up grid. Here r = 0 and r = L are excluded. It does, however, use
% that psi(0) = psi(L) = 0.
h = L/(N+1);
r = transpose(linspace(h, L-h, N));      % Column vector

% Set up potetial
Vpot = -1./r;

% Set up kinetic energy matrix by means of three point-formula
Tmat_FD3 = zeros(N, N);
Tmat_FD3(1, [1 2]) = [-2 1];
Tmat_FD3(N, [N-1 N]) = [1 -2];
for n=2:(N-1)
  Tmat_FD3(n, [n-1 n n+1]) = [1 -2 1]; 
end
Tmat_FD3 = Tmat_FD3/h^2;        % Correct duble derivative
Tmat_FD3 = -1/2*Tmat_FD3;       % Correct prefactor

% Full Hamiltonian
Ham = Tmat_FD3 + diag(Vpot);         % Hamiltonian

% Diagonalization
[U E] = eig(Ham);
% Extract eigen energies, the diagonal of E
E = diag(E);
% Sort eigenvalues - and rearrange eigenvectors accordingly
[E Ind] = sort(E);
U = U(:, Ind);
% Ensure proper normalization
U = U/sqrt(h);

% Determine the number of bound states
Nbound = length(find(E<0));
disp(['We found ', num2str(Nbound), ' numerical bound states.'])
% Plott the bound energies together with the Bohr formula
figure(1)
for n = 1:Nbound
  % Numerical values
  plot(1:Nbound, E(1:Nbound), 'rx', 'linewidth', 2)
  % Analytical values
  hold on
  plot(1:Nbound, -1./(2*[1:Nbound].^2), 'ko', 'linewidth', 2)
  hold off
end
grid on
set(gca, 'fontsize', 15)
xlabel('n quantum number')
ylabel('Energy')
legend('Numerical eigen energies', 'Bohr formula', 'location', 'southeast')

% Plot some of the wave functions
figure(2)
for n = 1:min(N,Nplots)
  % Write energy to screen
  plot(r, U(:,n), 'k-', 'linewidth', 2)
  title(['Energy index: ', num2str(n), ', energy: ', num2str(E(n))])
  set(gca, 'fontsize', 12)
  xlabel('r [a.u.]')
  ylabel('Wave function')
  grid on
  pause
end