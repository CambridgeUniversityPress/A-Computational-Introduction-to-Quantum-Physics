% This script estimates the ground state energy for a two-particle 
% system consisting of two identical fermions. The particles interact 
% via a smooth Coulomb-like interaction, and they are confined by a 
% "smooth" rectangular potential. The ground state energy is estimated 
% by means of the variational principle. 
% As trial function we use a product of two idential Gaussian wave 
% packets with a variable width.
%
% The Hamiltonian is estimated numerically using an FFT representation 
% of the kinetic energy.
%
% The estimated energy is plotted as a function of the width.
%
% Numerical inputs:
% SigmaMin  - minimal value for the widht
% SigmaMax  - maximal value for the width
% SigmaStep - step size used for the width
% L         - size of domain 
% N         - number of grid points (should be 2^n)
%
% Physical inputs:
% V0        - the height of the potential (should be negative)
% w         - the width of the potential
% s         - the "smoothness" of the potential 
% W0        - the strength of the interaction
%
% All parameters and functions are hard coded initially
%

SigmaMin = 0.1;
SigmaMax = 3;
SigmaStep = 0.005;

% One-particle test function (normalized Gaussian)
PsiTrial = @(x, sigma) (2*pi*sigma^2)^(-0.25)*exp(-x.^2/(4*sigma^2));

% Potential:
V0 = -1;
w = 4;
s = 5;

% Interaction strength
W0 = 1;

% Grid parameters
L = 15;
N = 128;              % For FFT's sake, we should have N=2^n


% The confining potential
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
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;            % Correct prefactor

% Potential energy
V = diag(Vpot(x));
% Total one-particle-Hamiltonian
H_OnePart = Tmat_FFT + V;


% Calculate expectation value of the Hamiltonian
% for various sigmas
index = 1;
for sigma = SigmaMin:SigmaStep:SigmaMax
  Psi = PsiTrial(x,sigma);                 % One-particle wave function
  
  % One-particle part of the energy
  OnePartContribution = h*Psi'*H_OnePart*Psi;
  
  % Two-particle wave function (absolute value squared)
  PsiTwoSquare = abs(Psi*Psi.').^2;
  % Expectation value of interaction
  InteractionContribution = h^2*sum(sum(WintMat.*PsiTwoSquare));
  
  % Total energy expectation value
  EnergyExpectation = 2*OnePartContribution + InteractionContribution;
  
  % Vectors with energy and sigma values
  Energy(index) = EnergyExpectation;      % Expectation value
  SigmaVector(index) = sigma;             % Vector with sigma-values
  index=index+1;                          % Update index
end

% Determine estimate - and write to screen
[EnergyEstimate, MinInd] = min(Energy);
OptimalSigma = SigmaVector(MinInd);
disp(['Ground state energy estimate: ', ...
    num2str(EnergyEstimate)])
disp(['Optimal sigma value: ', num2str(OptimalSigma)])

%
% Plot ground state energy estimate
%
figure(1)
plot(SigmaVector, Energy, 'k-', 'linewidth', 2)
grid on
set(gca,'fontsize',15)
xlabel('\sigma')
ylabel('<E>')
