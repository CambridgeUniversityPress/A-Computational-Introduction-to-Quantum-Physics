% This script determines the energies for a particle in a 
% periodic potential. In such situations, the energies becomes functions 
% of the parameter kappa. 
% In this particular example, the potential consists of a equivistant
% sequence of Gaussian confining potentials.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
% 
% Input for the potential:
%   V0      - The depth of the potential
%   SigmaP  - The width of the potential
%
% Input for plotting
%   NplotE  - The number of energies to display
%
% All inputs are hard coded initially.
%

% Periodicity and resolution
L=10;
N=256;

% Number of energies to plot
NplotE = 5;

% Parameters for the potential
V0 = -1;
SigmaPot = .1;
Vpot = @(x) V0*exp(-x.^2/2/SigmaPot^2);

% Position vector
x=linspace(-L/2,L/2,N);

% FFT kVector  
wavenumFFT = 2*(pi/L)*[(0:N/2-1), (-N/2:-1)]';       % Momentum vector, FFT


% Set up vector with kappa values (note: These differ from the k-values)
kappaMax = pi/L;
Nkappa = 250;                           % Number of kappa values to include
kappaVector = linspace(-kappaMax, kappaMax, Nkappa);

% Initiate and allocate
indX = 1;
EigMat = zeros(Nkappa, NplotE);
% Loop over kappa
for kappa=kappaVector
  % Set up Hamiltonian
  % Kinetic energy
  T = -1/2*ifft(diag((1i*(wavenumFFT+kappa)).^2)*fft(eye(N)));
  Ham = T + diag(Vpot(x));          % Add potential
  % Diagonalize and sort
  E=eig(Ham);
  E=real(E); 
  E=sort(E);
  % Get the lowest energies
  EigMat(indX,:)=E(1:NplotE);
  indX=indX+1;
end

% Plot energies
figure(1)
plot(kappaVector, EigMat, 'linewidth',2)
grid on
set(gca, 'fontsize', 15)
xlabel('\kappa')
% Adjust axis to only include relevant kappa-values
AxisVect = axis;
axis([-kappaMax kappaMax AxisVect(3) AxisVect(4)]);
