% This script estimates the ground state energy of a "smooth" 
% rectangular potential by means of the variational principle. 
% As a trial function we use a Gaussian wave packet centered at the 
% origin with a variable width.
%
% The Hamiltonian is estimated numerically using an FFT representation 
% of the kinetic energy.
%
% The ground state energy is also calculated "exactly" by diagoalizing 
% the full, numerical Hamiltonian.
%
% The estimated energy is plotted as a function of the width - along 
% with the "exact" ground state energy
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
% Vpot      - the actual potential (function variable)
%
% All parameters and functions are hard coded initially

SigmaMin = 0.1;
SigmaMax = 5;
SigmaStep = 0.05;

% Trial function
PsiTrial = @(x,sigma) (2*pi*sigma^2)^(-0.25)*exp(-x.^2/(4*sigma^2));

% Potential:
V0 = -3;
w = 8;
s = 5;

% Grid parameters
L = 30;
N = 1024;              % For FFT's sake, we should have N=2^n

% Shape of the potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Set up the grid.
x = linspace(-L/2,L/2,N);
x = x.';                            % Traspose
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

% Total Hamiltonian
H = Tmat_FFT + diag(Vpot(x));

% Calculate expectation value of the Hamiltonian
% for various sigmas
index = 1;
for sigma = SigmaMin:SigmaStep:SigmaMax
  Psi = PsiTrial(x,sigma);                 % Wave function
  Sigma(index) = sigma;                   % Vector with sigma-values
  Energy(index) = h*Psi'*H*Psi;           % Expectation value
  index=index+1;                          % Update index
end

% Diagonalize to find eigen states and energies
[EigVectors EigValues] = eig(H);
% Extract eigen values from diagonal matrix
EigValues = diag(EigValues);
% Sort them (and remove spurious imaginary parts)
[EigValues, Indexes] = sort(real(EigValues));
EigVectors = EigVectors(:,Indexes);
% "Exact" groundState
GroundStateEnergy = EigValues(1);
GroundStateWF = EigVectors(:,1)/sqrt(h);  % Normalize

% Determine estimate - and error
[EnergyEstimate, MinInd] = min(Energy);
Error = EnergyEstimate - GroundStateEnergy;
disp(['Ground state energy estimate: ', ...
    num2str(EnergyEstimate)])
disp(['Actual ground state energy (on the grid): ', ...
    num2str(GroundStateEnergy)])
disp(['The error in ground state energy is ', num2str(Error),'.'])
disp(['This is corresponds to the relative error ',...
    num2str(abs(Error/GroundStateEnergy)*100),' %.'])

%
% Plot ground state energy estimate and compare with "exact" result
%
figure(1)
plot(Sigma, Energy, 'k-', 'linewidth', 2)
hold on
yline(GroundStateEnergy, 'r--', 'linewidth', 2);
hold off
grid on
set(gca,'fontsize',15)
xlabel('\sigma')
legend('<E(\sigma)>','Exact')

%
% Plot wave functions - the estimate and the "exact" one
%
% Estimated ground state
Psi = PsiTrial(x, Sigma(MinInd));               
figure(2)
plot(x, abs(Psi).^2, 'k-', 'linewidth', 2)
hold on
% Plott "exact" wave function
plot(x, abs(GroundStateWF).^2, 'r--', 'linewidth', 2)          
% Fix strength of the potential for plotting
MaxWalPot = -.2*max(abs(Psi)).^2/V0;          
% Plot confining potential
plot(x, Vpot(x)*MaxWalPot, 'b');      
hold off
grid on
set(gca,'fontsize',15)
legend('Estimate','Exact','Potential')
