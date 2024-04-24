% This script estimates the ground state energy of a specific 
% two-dimensional potential by means of the variational principle. 
% As trial function we use a Gaussian wave packet with variable 
% width mean position.
%
% The Hamiltonian is estimated numerically using an FFT representation 
% of the kinetic energy.
%
% A trial function of product form is assumed. Two functional forms
% are set: A Gaussian and a cosine shaped one.
%
% The gradient descent implementation calls a function which calculates
% the energy expectation value. This function is implemented towards
% the very end of the script. (Because MATLAB inisists on this, unless
% the function is implemented as a separate file.)
%
% Inputs for the gradient decent approach:
% SigmaX0    - starting point for the width in the x-direction
% SigmaY0    - starting point for the width in the y-direction
% Gamma      - the so-called learning rate
% GradientMin  - the lower limit for the length of the gradient
% DerivStep    - the numerical step size for numerical integration
%
% Grid parameters
% L         - size of domain 
% N         - the number of grid points (should be 2^n)
%
% Physical inputs:
% V0        - the height of the smoothl rectangular potential (negative)
% wX        - the width of the potential in the x-direction
% wY        - the width of the potential in the y-direction
% s         - the "smoothness" of the potential 
%
% All parameters and functions are hard coded initially

% Parameters for gradeient descent
SigmaX0 = 2;
SigmaY0 = 2;
Gamma = 0.1;
GradientMin = 1e-5;
DerivStep = 1e-1;

% Trial function
% Gaussian version
%PsiTrial = @(x,sigma) (2*pi*sigma^2)^(-0.25)*...
%    exp(-(x).^2/(4*sigma^2));
% Cosine version
% (sigma will here play the role of an inverse width)
%PsiTrial = @(x, sigma) sqrt(2*sigma/pi)*cos(sigma*x).*(abs(x)<pi/(2*sigma));

% Parameters for the smoothly rectangular part of the potential:
V0 = -1;
wX = 4;
wY = 2;
s = 4;

% Grid parameters
L = 10;
N = 1024;              % For FFT's sake, we should have N=2^n

% Shape of the potential
Vpot = @(x, y) V0./(exp(s*(abs(x)-wX/2))+1)./(exp(s*(abs(y)-wY/2))+1);

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
Tmat_FFT = -1/2*Tmat_FFT;                       % Correct prefactor

% Matrix for potential
[X Y] = meshgrid(x, x);                 % Matrix with x and y values
Vmat = Vpot(X, Y);

% 
% Gradient descent
%
% Initiate parameters for gradient descent
index = 1;
sigmaX = SigmaX0;
sigmaY = SigmaY0;
LengthGradient = 1e6;     % Fix high value to get the loop going
% Iterate while the gradient is large enough
while LengthGradient > GradientMin
  % Calculate new energy estimate
  EnergyEstimate = EnergyExpectation(sigmaX, sigmaY, x, h, ...
      Tmat_FFT, Vmat, PsiTrial);
  
  % Estimate partial derivatives by means of midpoint rule
  dEdSigmaX = (EnergyExpectation(sigmaX+DerivStep, sigmaY, x, h, ...
               Tmat_FFT, Vmat, PsiTrial) - ...
               EnergyExpectation(sigmaX-DerivStep, sigmaY, x, h, ...
               Tmat_FFT, Vmat, PsiTrial))/(2*DerivStep);
  dEdSigmaY = (EnergyExpectation(sigmaX, sigmaY+DerivStep, x, h, ...
               Tmat_FFT, Vmat, PsiTrial) - ...
               EnergyExpectation(sigmaX, sigmaY-DerivStep, x, h, ...
               Tmat_FFT, Vmat, PsiTrial))/(2*DerivStep);  
  % Calculate length of the gradient
  LengthGradient = sqrt(dEdSigmaX^2 + dEdSigmaY^2);
  
  % Update sigmaX and sigmaY
  sigmaX = sigmaX - Gamma*dEdSigmaX;
  sigmaY = sigmaY - Gamma*dEdSigmaY;
  
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
disp(['Optimal value for the x-width:', num2str(sigmaX)])
disp(['Optimal value for the y-width:', num2str(sigmaY)])
disp(['Total number of iterations: ', num2str(index)])


%
% Function which calculates energy expectation value
%
function Eval = EnergyExpectation(sigmaX, sigmaY, x, h, Tmat, ...
    Vmat, PsiTrial)
  
  % The x-part and the y-part of the two-variable trial wave function
  PsiX = PsiTrial(x, sigmaX);
  PsiY = PsiTrial(x, sigmaY);
  
  % Kinetic energy in the x-direction
  Tx = h * PsiX'*Tmat*PsiX;
  % Kinetic energy in the Y-direction
  Ty = h * PsiY'*Tmat*PsiY;
  
  % Construct matrix for the two-variable wave function
  PsiTwo = PsiX*transpose(PsiY);
  PsiTwoAbs2 = abs(PsiTwo).^2;
  Vval = h^2 * sum(sum(Vmat.*PsiTwoAbs2));
  Eval = Tx + Ty + Vval;
end