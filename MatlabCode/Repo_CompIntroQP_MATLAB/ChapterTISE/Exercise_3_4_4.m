% This script estimates the ground state energy of a specific potential 
% by means of the variational principle. As a trial function we use a 
% Gaussian wave packet with variable width mean position.
%
% The Hamiltonian is estimated numerically using an FFT representation 
% of the kinetic energy.
%
% The ground state energy is also calculated "exactly" by diagoalizing 
% the full, numerical Hamiltonian.
%
% Inputs for the gradient decent approach:
% Sigma0    - starting point for the width
% Mu        - starting point for the mean position
% Gamma     - the so-called learning rate
% GradientMin  - the lower limit for the length of the gradient
% DerivStep    - the numerical step size for estimating the gradient
%
% Grid parameters
% L         - size of domain 
% N         - number of grid points (should be 2^n)
%
% Physical inputs:
% V0        - the height of the smoothl rectangular potential (negative)
% w         - the width of the potential
% s         - the "smoothness" of the potential 
% Vpot      - the potential - which has an additional square term
%
% All parameters and functions are hard coded initially

% Parameters for gradient descent
Sigma0 = 2;
Mu0 = 4;
Gamma = 0.1;
GradientMin = 1e-3;
DerivStep = 1e-1;

% Parameters for the smoothly rectangular part of the potential:
V0 = -5;
w = 6;
s = 4;

% Grid parameters
L = 20;
N = 512;              

% Shape of the potential
Vsmooth = @(x) V0./(exp(s*(abs(x)-w/2))+1);
Vpot = @(x) Vsmooth(x-2) + x.^2/50;

% Set up the grid.
x = linspace(-L/2, L/2, N)';
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

% Potential energy
V = diag(Vpot(x));
% Total Hamiltonian
H = Tmat_FFT + V;

% The Gaussian shape used in the trial function
PsiTest = @(x,sigma, mu) (2*pi*sigma^2)^(-0.25)*...
    exp(-(x-mu).^2/(4*sigma^2));

% Function which calculates energy expectation value
EnergyExpectation = @(sigma, mu) h*PsiTest(x, sigma, mu)'*H* ...
    PsiTest(x, sigma, mu);

% 
% Gradient descent
%
% Initiate parameters for gradient descent
index = 1;
sigma = Sigma0;
mu = Mu0;
LengthGradient = 1e6;           % Fix high value to get the loop going

% Iterate while the gradient is large enough
while LengthGradient > GradientMin
  % Calculate new energy estimate
  EnergyEstimate = EnergyExpectation(sigma, mu);
  
  % Estimate partial derivatives by means of midpoint rule
  dEdSigma = (EnergyExpectation(sigma+DerivStep, mu) - ...
      EnergyExpectation(sigma-DerivStep, mu))/(2*DerivStep);
  dEdMu = (EnergyExpectation(sigma, mu+DerivStep) - ...
      EnergyExpectation(sigma, mu-DerivStep))/(2*DerivStep);
  % Calculate length of the gradient
  LengthGradient = sqrt(dEdSigma^2 + dEdMu^2);
  
  % Update sigma and mu
  sigma = sigma - Gamma*dEdSigma;
  mu = mu - Gamma*dEdMu;

  % Write estimate to screen for every 10th iteration
  if mod(index, 10) == 0
      disp(['Iteration: ', num2str(index), ', energy: ',...
          num2str(EnergyEstimate)])
  end
  
  % Update index
  index=index+1;                          
end

% Diagonalize to find actual eigenstates and energies
[EigVectors EigValues] = eig(H);
% Extract eigen values from diagonal matrix
EigValues = diag(real(EigValues));
% Sort them
[EigValues, Indexes] = sort(EigValues);
EigVectors = EigVectors(:, Indexes);
% "Exact" groundState
GroundStateEnergy = EigValues(1);
% Normalized ground state wave function:
GroundStateWF = EigVectors(:,1)/sqrt(h);

% Write estimates and error to screen
disp(['Ground state energy estimate: ', ...
    num2str(EnergyEstimate)])
disp(['Actual ground state energy (on the grid): ', ...
    num2str(GroundStateEnergy)])
Error = EnergyEstimate - GroundStateEnergy;
disp(['The error in ground state energy is ', num2str(Error),'.'])
RelError = abs(Error/GroundStateEnergy)
disp(['This is corresponds to the relative error ',...
    num2str(RelError*100),' %.'])

%
% Plot ground state energy estimate and compare with "exact" result
%
figure(2)
% Plot estimate
PsiEstimate = PsiTest(x, sigma, mu);
plot(x, abs(PsiEstimate).^2, 'k-', 'linewidth', 2)
hold on
% Plot true ground state on the grid
plot(x, abs(GroundStateWF).^2, 'r--', 'linewidth', 2)
% Plot scaled potential
MaxVal = max(abs(GroundStateWF).^2);
plot(x, Vpot(x)/abs(V0)*MaxVal*0.2, 'b-.', 'linewidth', 2)
hold off
% Cosmetics
grid on
xlabel('x')
ylabel('|\Psi(x)|^2')
legend('Estimate','Exact', 'Potential', 'location', ...
    'northwest')
set(gca,'fontsize',15)