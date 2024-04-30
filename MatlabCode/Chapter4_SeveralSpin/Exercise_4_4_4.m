% This script estimates the ground state energy of a one-dimensional
% two-particle system by the method self-consistent field. We assume 
% that the ground can be approxmated by a simple product of two identical 
% wave functions for each particle. From this wave function an effective 
% potential is calculated. The Hamiltonian of the resulting effective 
% one-particle Hamiltonian is diagonalsed and the ground state of this 
% is taken as our updated one-particle wave function. This is iterated 
% until the effective one-particle energy is converged.
%
% As initial guess we have taken the one-particle ground state.
% 
% This is a simple example of a Hartree-Fock calculation.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   Epres   - The energy precision, used to set convergence criteria
%
% Input for the confining potential:
%   V0      - The "height", the the barrier (must be negative for a well)
%   w       - The width of the barrier
%   s       - Smoothness parameter, rectangular well for s -> infinity
%
% All inputs are hard coded initially.

% Numerical grid parameters
L = 20;
N = 512;                
Epres = 1e-5;

% Input parameters for the confining potential
V0 = -1;
w = 4;
s = 5;

% Input for the interaction
W0 = 1;

% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      % Column vector
% Meshgrid
[X1 X2] = meshgrid(x, x);

% Confining potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Interaction
Wint = @(x1, x2) W0./sqrt((x1-x2).^2+1);

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;            % Correct prefactor

% Full one-particle Hamiltonian
HamOnePart = Tmat_FFT + diag(Vpot(x));         % Hamiltonian

% Matrix for the interaction
Wmat = Wint(X1, X2);

% Diagonalization - in order to set initial state
[U E] = eig(HamOnePart);
% Extract eigen energies, the diagonal of E
E = diag(E);
% Sort eigenvalues - and rearrange eigenvectors accordingly
[E Ind] = sort(E);
U = U(:, Ind);
% Select initial wave function and normalize it
Psi = U(:,1)/sqrt(h);
EnergyOnePart = E(1);
% Initialte old energy (to get the loop going)
EnergyOnePartOld = 1e6;

% Allocate effective potential
Veff = zeros(N, 1);

% Iterate until self consistency
iterations = 0;
while abs(EnergyOnePart-EnergyOnePartOld) > Epres
  % Set up effective interaction potential
  PsiSq = abs(Psi).^2;
  Veff = h*Wmat*PsiSq;
  Heff = HamOnePart + diag(Veff);
  
  % Diagonalize effective Hamiltonian
  [U E] = eig(Heff);
  E = diag(E);
  [E Ind] = sort(E);
  U = U(:, Ind);
  Psi = U(:,1)/sqrt(h);
  
  % Update energy
  EnergyOnePartOld = EnergyOnePart;
  EnergyOnePart = E(1);
  iterations = iterations + 1;
  
  % Write progress to screen
  disp(['Iteration:', num2str(iterations), ', one-particle energy: ', ...
      num2str(EnergyOnePart)])
end

% Determine energy
EonePart = h*Psi'*HamOnePart*Psi;
PsiTwoPart = Psi*Psi.';
EtwoPart = h^2*sum(sum(Wmat.*abs(PsiTwoPart).^2));
Etotal = 2*EonePart + EtwoPart;
% Write result to screen:
disp(['Two-particle energy estimate: ', num2str(Etotal)])
disp(['Iterations: ', num2str(iterations)])