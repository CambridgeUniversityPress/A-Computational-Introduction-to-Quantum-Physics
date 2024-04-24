% This script calculates the probability to remain in the the 
% initial state for a particle in a harmonic osciallater which
% is shaken. It does so by solving the Schrödinger equation formulated
% within the adiabatic basis. The time-stepping is performed
% using a propagator of Crank-Nicolson form.
% 
% The harmonic oscillated potential is subject to a 
% time-dependent translation which shifts it in both directions
% before restoring it in its original position. The dynamics is
% resolved repeatedly with varying duration.
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
% InitialState - the index of the initial state; 
% InitialState = 0 is the ground state
%
% All inputs are hard coded initially

% Grid parameters
L = 15;
N = 512;              

% Strength of the harmonic oscillator
k = 1;

% Time parameters
Nstep = 250;
omegaMin = 0.05;
omegaMax = 0.5;
domega = 0.05;

% Index of initial state (ground state corresponds to zero)
InitialState = 0;

% Shape of the potential
Vpot = @(x) 0.5*k*x.^2;

% The derivative of displacement of the potential
FtransDeriv = @(t, omega) (t<2*pi/omega)*4/3^(3/2)*omega*...
    (2*sin(omega*t/2).^2.*cos(omega*t) + sin(omega*t).^2);

% Set up the grid
x = linspace(-L/2,L/2,N);
x = x.';                            % Transpose
h = L/(N-1);

% Set up momentum and kinetic energy matrix
k = 2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Itrans = fft(eye(N));
% Multiply by (ik)^2 for T and just ik for p
Tmat = diag(-k.^2)*Itrans;
Pmat = diag(1i*k)*Itrans;
% Transform back to x-representation
Tmat = ifft(Tmat);
Tmat = -1/2*Tmat;            % Correct prefactor
Pmat = -1i*ifft(Pmat);

% Diagonalize initial Hamiltonian (in grid representation)
Hinit = Tmat + diag(Vpot(x));
[B E] = eig(Hinit);
E = diag(E);
% Sort and normalize
[E Ind] = sort(E);
B = B(:, Ind)/sqrt(h);

% Hamiltonian 
Dmat = diag(E);             % Diagonal energies (time-independent)
CoupMat = h*B'*Pmat*B;      % Coupling matrix

% Initiate and allocate
% Identity matrix
Imat = eye(N);
% Vectors with omega and Pinit values
omegaVector = omegaMin:domega:omegaMax;
Pvector = zeros(1, length(omegaVector));
index = 1;

% Loop over the omega values
for omega = omegaVector
  % Initial state
  Avec = zeros(N,1);
  Avec(InitialState + 1) = 1;
  
  % Time parameters
  tTotal = 2*pi/omega;
  dt = tTotal/Nstep;
  t = 0;
  Fderiv = 0;
  Ham = Dmat;
  
  % Do the dynamics
  while t < tTotal
    % Update Hamiltonian and the derivative of the displament
    FderivOld = Fderiv;
    HamOld = Ham;
    Fderiv = FtransDeriv(t+dt, omega);
    Ham = Dmat - Fderiv*CoupMat;
    % The Crank-Nicolson scheme
    Avec = (Imat - 1i*HamOld*dt/2)*Avec;
    Avec = inv(Imat + 1i*Ham*dt/2)*Avec;
    % Update time
    t = t + dt;
  end
  % Final probability
  Pinit = abs(Avec(InitialState+1)).^2;
  
  % Write propbability to screen
  disp(['omega: ', num2str(omega), ', Pinit: ', num2str(Pinit)])
  Pvector(index) = Pinit;
  index = index + 1;
% Plot Pinit as a function of omega
figure(1)
plot(omegaVector, Pvector, 'k-', 'linewidth', 2)
grid on
xlabel('\omega')
ylabel('P_{init}')
set(gca, 'fontsize', 15)
end

% Plot Pinit as a function of omega
figure(1)
plot(omegaVector, Pvector, 'k-', 'linewidth', 2)
grid on
xlabel('\omega')
ylabel('P_{init}')
set(gca, 'fontsize', 15)