% This script estimates the ground state energy for a two-particle 
% system i one dimensions. The trial function is a symmetrized product of
% Gaussian single-particle functions with mean positions of opposite signs.
% Specifically: 
% Psi ~ 
% \psi(x_1; +mu) \psi(x_2; -mu) + \psi(x_1; -mu) \psi(x_2; +mu)
%
% The Hamiltonian is estimated numerically using an FFT representation 
% of the kinetic energy. It features an interaction term which is a 
% "softened" Coulomb interaction.
%
% The ground state energy is estimated using the variational principle 
% taking the width sigma and the mean position mu of the Gaussians as 
% variational parameters.
%
% The minimization, in turn, is done via the gradient descent method.
%
% Inputs for the gradient decent approach:
% Sigma0    - starting point for the width
% Mu0       - starting point for the mean position
% Gamma     - the so-called learning rate
% GradientMin  - the lower limit for the length of the gradient
% DerivStep    - the numerical step size for numerical integration
%
% Grid parameters
% L         - size of domain 
% N         - number of grid points (should be 2^n)
%
% Physical inputs:
% V0        - the height of the smoothly rectangular potential (negative)
% w         - the width of the potential
% s         - the "smoothness" of the potential 
% W0        - the strength of the two-particle interaction
%
% All parameters and functions are hard coded initially
%

% Parameters for gradeient descent
Sigma0 = 2;
Mu0 = 2;
Gamma = 0.05;
GradientMin = 1e-4;
DerivStep = 1e-3;

% Parameters for the smoothly rectangular part of the potential:
V0 = -1;
w = 4;
s = 5;
% Interaction strength
W0 = 1;

% Grid parameters
L = 20;
N = 512;

% Functions:
%
% Confining potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);
% Interaction
Wint = @(x1, x2) W0./sqrt((x1-x2).^2 + 1);
% Function for one-particle functions (Gaussians)
PsiGauss = @(x,sigma, mu) (2*pi*sigma^2)^(-0.25)*...
    exp(-(x-mu).^2/(4*sigma^2));

% Set up the grid.
x = linspace(-L/2, L/2, N).';
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
% Single particle Hamiltonian
H_OnePart = Tmat_FFT + V;
%H_OnePart = 0.5*(H_OnePart+H_OnePart');        % Enforce Hermicity

% Matrix for interaction
[X1 X2] = meshgrid(x, x);                 % Matrix with x and y values
IntMatrix = Wint(X1, X2);

% 
% Gradient descent
%
% Initiate parameters for gradient descent
index = 1;
Sigma = Sigma0;
Mu = Mu0;
LengthGradient = 1e6;           % Fix high value to get the loop going

% Iterate while the gradient is large enough
while LengthGradient > GradientMin
  % Calculate new energy estimate (calls function defined below)
  EnergyEstimate = EnergyExpectation(Sigma, Mu, x, h, H_OnePart, ...
  IntMatrix, PsiGauss);
  
  % For plotting
  SigmaVect(index) = Sigma;
  MuVect(index) = Mu;
  EnergyVect(index) = EnergyEstimate;
  
  % Estimate partial derivatives by means of midpoint rule
  dEdSigma = (EnergyExpectation(Sigma+DerivStep, Mu, x, h, H_OnePart, ...
  IntMatrix, PsiGauss) - ...
  EnergyExpectation(Sigma, Mu, x, h, H_OnePart, ...
  IntMatrix, PsiGauss))/(2*DerivStep);
  dEdMu = (EnergyExpectation(Sigma, Mu+DerivStep, x, h, H_OnePart, ...
  IntMatrix, PsiGauss) - ...
  EnergyExpectation(Sigma, Mu-DerivStep, x, h, H_OnePart, ...
  IntMatrix, PsiGauss))/(2*DerivStep);  
  % Calculate length of the gradient
  LengthGradient = sqrt(dEdSigma^2 + dEdMu^2);
  
  % Update Sigma and Mu
  Sigma = Sigma - Gamma*dEdSigma;
  Mu = Mu - Gamma*dEdMu;
  
  % Write estimate to screen for every 10th iteration
  if mod(index, 10) == 0
      disp(['Iteration: ', num2str(index), ', energy: ',...
          num2str(EnergyEstimate)])
  end
  
  % Update index
  index=index+1;                          
end

% Write results to screen:
disp(['Minmal energy: ', num2str(EnergyEstimate)])
disp(['Optimal value for sigma:', num2str(Sigma)])
disp(['Optimal value for mu:', num2str(Mu)])
disp(['Total number of iterations: ', num2str(index-1)])

% Figure illustrating the convergence
figure(1)
Emin = EnergyVect(end);
% Curve in the sigma, mu, E-space
plot3(SigmaVect, MuVect, EnergyVect, 'k-', 'linewidth', 2)
hold on
% Plane of the lowest energy
surf([min(SigmaVect) max(SigmaVect)], [min(MuVect) max(MuVect)], ...
    [Emin Emin; Emin Emin])
alpha(0.3)
colormap([0 .6 0])
% Plot projection of the convergence curve
plot3(SigmaVect, MuVect, Emin*ones(index-1), 'r--', 'linewidth', 2)
hold off
xlabel('\sigma')
ylabel('\mu')
zlabel('<E>')
set(gca, 'fontsize', 15)
grid on

%
% Function which calculates energy expectation value
%
function Eval = EnergyExpectation(Sigma, Mu, x, h, H_OnePart, ...
  IntMatrix, PsiGauss);  
  
  % One-particle functions
  PsiPlus  = PsiGauss(x, Sigma, +Mu);
  PsiMinus = PsiGauss(x, Sigma, -Mu);
  
  % "Diagonal" single partilcle energy
  OnePart1 = h*PsiPlus'*H_OnePart*PsiPlus;
  % Mixed single particle enegy
  OnePart2 = h*PsiPlus'*H_OnePart*PsiMinus;
  Overlap = h*PsiMinus'*PsiPlus;
  OnePart2 = OnePart2*Overlap;
  
  % Two-partile wave function parts
  PsiMatPlusMinus = PsiMinus*transpose(PsiPlus);
  PsiMatMinusPlus = PsiPlus*transpose(PsiMinus);
  
  % "Diagonal" two-particle energy
  TwoPart1 = h^2*sum(sum(conj(PsiMatPlusMinus).*IntMatrix.*...
      PsiMatPlusMinus));
  
  % Mixed two-particle energy
  TwoPart2 = h^2*sum(sum(conj(PsiMatPlusMinus).*IntMatrix.*...
      PsiMatMinusPlus));
  
  % Total Energy
  % Symmetric version
  Eval = (2*OnePart1+2*OnePart2+TwoPart1+TwoPart2) / (1 + Overlap^2);
  % Anti-symmetric version
  %Eval = (2*OnePart1-2*OnePart2+TwoPart1-TwoPart2) / (1 - 1*Overlap^2);
end