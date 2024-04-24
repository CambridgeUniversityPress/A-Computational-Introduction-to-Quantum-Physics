% This script calculates the probability to remain in the the 
% initial state for a particle in a harmonic osciallator which
% is shaken. It does so by first expressing the kinetic energy term
% via FFT and then approximating the second order Magnus propagator as 
% a split operator.
% 
% The harmonic oscillator potential is subject to a time-dependent 
% translation which shifts it in both directions before restoring it 
% in its original position. In addition to time, it takes omega as an 
% input; the duraton of the time-dependent shift is related to omega by 
% T = 2*pi/omega.
%
% Inputs
% L         - size of domain 
% N         - number of grid points (should be 2^n)
% Nt        - the number of time steps for each run
% k         - the strength of the potential
% omegaMin  - minimal omega value
% omegaMax  - maximal omega value
% domega    - step size in omega
% InitialState - the index of the initial state;  InitialState = 0 is 
% the ground state
%
% All inputs are hard coded initially

% Grid parameters
L = 10;
N = 512;              

% Strength of the harmonic oscillator
k = 1;

% Time parameters
Nstep = 500;
omegaMin = 0.05;
omegaMax = 1.5;
domega = 0.025;

% Index of initial state (ground state corresponds to zero)
InitialState = 0;

% Shape of the potential
Vpot = @(x) 0.5*k*x.^2;

% The displacement of the potential
Ftrans = @(t, omega) 8/3^(3/2)*sin(omega*t/2).^2.*sin(omega*t);

% Set up the grid
x = linspace(-L/2,L/2,N);
x = x.';                            % Transpose
h = L/(N-1);

% Set up kinetic energy matrix by means of the fast Fourier transform
k = 2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;            % Correct prefactor

% Diagonalize initial Hamiltonian
Hinit = Tmat_FFT + diag(Vpot(x));
[U E] = eig(Hinit);
E = diag(E);
[E Ind] = sort(E);
U = U(:, Ind)/sqrt(h);

% InitialState
Psi0 = U(:, InitialState+1);

% Loop over the omega values
omegaVector = omegaMin:domega:omegaMax;
Pvector = zeros(1, length(omegaVector));         % Allocate    
index = 1;
for omega = omegaVector
  % Duration
  tTotal = 2*pi/omega;
  % Time-step
  dt = tTotal/Nstep;
  % Propagator for half-step with kinetic energy
  UkinHalf = expm(-1i*Tmat_FFT*dt/2);
  t = 0;
  % Initial state
  Psi = Psi0;
  
  % Do the dynamics
  while t < tTotal
    % Half step with kinetic energy
    Psi = UkinHalf*Psi;  
    % Full step with potential energy
    Translation = Ftrans(t+dt/2, omega);
    UpotFull = diag(exp(-1i*Vpot(x-Translation)*dt));
    Psi = UpotFull*Psi;
    % Another half step with kinetic energy
    Psi = UkinHalf*Psi;
    % Update time
    t = t + dt;
  end
  % Probability for remaining in the initial state
  Pinit = abs(trapz(x,conj(Psi).*Psi0))^2;
  % Write result to screen
  disp(['omega: ', num2str(omega), ', Pinit: ', num2str(Pinit)])
  Pvector(index) = Pinit;
  index = index + 1;
end

% Plot Pinit as a function of omega
figure(1)
plot(omegaVector, Pvector, 'k-', 'linewidth', 2)
grid on
xlabel('\omega')
ylabel('P_{init}')
set(gca, 'fontsize', 15)