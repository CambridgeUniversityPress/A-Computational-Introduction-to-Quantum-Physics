% This script solve the Schrödinger equation for a 1D model of an atom 
% exposed to a linearly polarized laser pulse in the dipole approximation. 
% It does so by expressing the wave function within a spectral basis; 
% by writing it as a linear combination of the eigen states of the 
% unperturbed part of the Hamiltonian, H0.
% The interaction, -q E(t) x, is calculated as matrix elements within
% this basis.
%
% For the time-stepping, the Crank-Nicolson (CN) approximation to the 
% propagator is used.
%
% The spectral basis is truncated; we remove all eigen states of H0 
% beyond a threshold value from our basis in order to speed up the 
% calculation. 
%
% The purely time-dependent laser pulse is modelled as a sin^2-type 
% envelope times a sine-carrier. 
%
% All parameters are given in atomic units.
%
% Inputs for the confining potential
% w     -   the width of the potential
% V0    -   the "height" of the potential (should be negative)
% s     -   "smoothness" parameter
%
% Inputs for the laser pulse
% Ncycl  -   the number of optical cycles
% Tafter -   time propagation after pulse 
% omega  -   the central frequency of the laser
% E0     -   the strength of the pulse
%
% Numerical input parameters
% dt    - numerical time step
% N     - number of grid points, should be 2^n
% L     - the size of the numerical domain; it extends from -L/2 to L/2
% Etrunc - the cutoff value for the basis trunction.
%
% All input parameters are hard coded initially.

% Inputs for the confining potential
w = 5;
V0 = -1;
s = 5;

% Inputs for the laser pulse
Ncycl  = 10;
omega = 1.0;
E0 = 0.5;

% Numerical input parameters
dt = .05;
N = 1024;
L = 400;
Etrunc = 7;


% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Confining potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Laser field
Tpulse=Ncycl*2*pi/omega;
Epulse = @(t) (t>0 & t<Tpulse)*E0.*sin(pi*t/Tpulse).^2.*...
    sin(omega*t);

% Set up kinetic energy matrix by means of the fast 
% Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];        % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;                   % Correct prefactor
% H0 - the time-independent part of the Hamiltonian
H0 = Tmat_FFT + diag(Vpot(x));

% Diagonalize time-independent Hamiltonian
[BasisMat E] = eig(H0);                  % Diagonalization
E = diag(E);                             % Extract diagonal energies
[E SortInd] = sort(E);                   % Sort energies
BasisMat = BasisMat(:,SortInd)/sqrt(h);  % Sort eigenvectors and normalize
Psi = BasisMat(:,1);                     % Initiate wave function
disp(['Ground state energy: ',num2str(E(1)),'.'])         
Nbound = length(find(E<0));              % The number of bound states
disp(['The potential supports ',num2str(Nbound),' bound states.'])
clear Tmat_FFT H0;                       % Clear obsolete matrices

% Impose truncation
Ntrunc = max(find(E<Etrunc));
E = E(1:Ntrunc);
BasisMat = BasisMat(:, 1:Ntrunc);
% Write basis size to screen
disp(['The truncated basis has ', num2str(Ntrunc), ' states.'])

% Matrices in spectral representation
H0 = diag(E);
Hint = BasisMat'*diag(x)*BasisMat*h;
% Identity
Imat = eye(Ntrunc);

% Initiate intial state and other variables
Avec = zeros(Ntrunc, 1);
Avec(1) = 1;
ProgOld = 0;
t = 0;
HamNext = H0;

% Loop over time
while t < Tpulse
  % Update Hamiltonian
  Ham = HamNext;
  HamNext = H0 + Epulse(t+dt)*Hint;
  % Half-step forward with the CN propagator
  Avec = (Imat - 1i*Ham*dt/2)*Avec;
  % Half-step backwards with the CN propagator
  Avec = inv(Imat + 1i*HamNext*dt/2)*Avec;
  
  % Update time
  t=t+dt;
  % Write progress sto screen
  Prog = floor(t/Tpulse*10);
  if Prog ~= ProgOld
    disp(['Progress: ',num2str(10*Prog),'%'])
    ProgOld = Prog;
  end
end

% Calculate probabilities
Pion = 1-sum((E<0).*abs(Avec).^2);
P_GroundState = abs(Avec(1))^2;
Pexcite = 1-Pion-P_GroundState;
% Write results to screen
disp(['Ionization probability: ', num2str(Pion*100), '%'])
disp(['Ground state probability: ', num2str(P_GroundState*100), '%'])
disp(['Excitation probability: ', num2str(Pexcite*100), '%'])
