% This script calculates position and momentum expectation values 
% and widths for a Gaussian wave packet moving freely in one dimension. 
% It plots these four quantities after the simulation is over - in 
% addition to the product of the position and momentum widths. This is 
% done in order to check that the Heisenberg uncertainty relation holds 
% for this wave packet.
%
% The numerical derivatives, which are involved both in solving the 
% Schrödinger equation and in calculating momentum expectation values 
% and widhts, are approximated by means of the Fast Fourier Transform.
%
% The time-dependent expectation value for x and the position width 
% are also calculated analytically and compared with the numerical 
% solution.
% 
%  Numerical inputs:
%    L       - The extension of the spatial grid 
%    N       - The number of grid points
%    Tfinal  - The duration of the simulation
%    dt      - The step size in time
% 
%  Physical inputs:
%    x0      - The mean position of the initial wave packet
%    p0      - The mean momentum of the initial wave packet
%    sigmaP  - The momentum width of the initial, Gaussian wave packet
%    tau     - The time at which the Gaussian is narrowest (spatially)
%  
%  All inputs are hard coded initially.

% Numerical grid parameters
L = 100;
N = 512;                % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Numerical time parameters
Tfinal = 15;
dt = 0.2;

% Input parameters for the Gaussian
x0 = -20;
p0 = 3;
sigmaP = 0.2;
tau = 5;

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector
% Vectors used to calculate expectation values
% (Far from optimal, but theoretically consistent)
Xmat = diag(x);
X2mat = diag(x.^2);

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Aux = fft(eye(N));
% Multiply by ik and (ik)^2
D1_FFT = diag(1i*k)*Aux;
D2_FFT = diag(-k.^2)*Aux;
% Transform back to x-representation
D1_FFT = ifft(D1_FFT);
D2_FFT = ifft(D2_FFT);

% Set up propagator for the approximation
U_FFT = expm(-1i*(-1/2)*D2_FFT*dt);

% Set up Gaussian - analytically
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
% Initial Gaussian
Psi0 = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);

% Initiate wave functon and time
Psi = Psi0;
t = 0;

% Allocate vectors with data to plot
Npoint = ceil(Tfinal/dt);
Tvector = zeros(1, Npoint);
MeanXvector = zeros(1, Npoint);
MeanPvector = zeros(1, Npoint);
WidthXvector = zeros(1, Npoint);
WidthPvector = zeros(1, Npoint);

% Loop which updates wave functions and plots in time
counter = 1;
while t < Tfinal
  % Expectation values and widths
  MeanX = h*real(Psi'*Xmat*Psi);
  MeanX2 = h*real(Psi'*X2mat*Psi);
  MeanP = h*real(Psi'*(-1i*D1_FFT)*Psi);
  MeanP2 = h*real(Psi'*(-D2_FFT)*Psi);

  % Store data in vectors
  Tvector(counter) = t;
  MeanXvector(counter) = MeanX;
  MeanPvector(counter) = MeanP;
  WidthXvector(counter) = sqrt(MeanX2 - MeanX^2);
  WidthPvector(counter) = sqrt(MeanP2 - MeanP^2);
  
  % Update wave function
  Psi = U_FFT*Psi;

  % Update time
  t = t+dt;             
  counter = counter + 1;
end

% Plot expectation values and widths
% Mean x
figure(1)
plot(Tvector, MeanXvector, 'k-', 'linewidth', 2)
% Analytical solution
MeanXanalytical = x0 + p0*Tvector;
hold on
plot(Tvector, MeanXanalytical, 'r--', 'linewidth', 2)
hold off
xlabel('Time')
ylabel('<x>')
set(gca, 'fontsize', 15)
grid on

% Width in x
figure(2)
plot(Tvector, WidthXvector, 'k-', 'linewidth', 2)
SigmaXanalytical = sqrt(1+4*sigmaP^4*(Tvector-tau).^2)/(2*sigmaP);
hold on
plot(Tvector, SigmaXanalytical, 'r--', 'linewidth', 2)
hold off
xlabel('Time')
ylabel('\sigma_x')
set(gca, 'fontsize', 15)
grid on

% Mean p
figure(3)
plot(Tvector, MeanPvector, 'k-', 'linewidth', 2)
xlabel('Time')
ylabel('<p>')
set(gca, 'fontsize', 15)
grid on

% Width in p
figure(4)
plot(Tvector, WidthPvector, 'k-', 'linewidth', 2)
xlabel('Time')
ylabel('\sigma_p')
set(gca, 'fontsize', 15)
grid on

% Product of widths
figure(5)
plot(Tvector, WidthXvector.*WidthPvector, 'k-', 'linewidth', 2)
hold on
yline(0.5, 'r--', 'linewidth', 2);
xline(tau, 'b-.','linewidth', 2);
hold off
xlabel('Time')
legend('\sigma_x \cdot \sigma_p', 'h/(4\pi)','t=\tau', 'location', ...
    'southeast')
set(gca, 'fontsize', 15)
grid on
axis([0 Tfinal 0.45 0.6])