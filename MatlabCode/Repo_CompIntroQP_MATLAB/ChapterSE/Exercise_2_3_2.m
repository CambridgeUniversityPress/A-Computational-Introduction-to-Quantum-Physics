% This script simulates the evolution of a Gaussian wave packet moving
% freely in one dimension. It does so in four ways:
% Firstly, by plotting the analytically known wave function, secondly by
% estimating the numerical solution of the Schrödinger equation using three
% different approximation for the kinetic energy operator:
% 1) A three point finite difference formula,
% 2) A five point finite difference formula and
% 3) the Fast Fourier transform.
%
% Numerical inputs:
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%   Tfinal  - The duration of the simulation
%   dt      - The step size in time
%
% Physical inputs:
%   x0      - The mean position of the initial wave packet
%   p0      - The mean momentum of the initial wave packet
%   sigmaP  - The momentum width of the initial, Gaussian wave packet
%   tau     - The time at which the Gaussian is narrowest (spatially)
% 
% All inputs are hard coded initially.

% Numerical grid parameters
L = 100;
N = 256;                % Should be 2^k, k integer, for FFT's sake
h = L/(N-1);

% Numerical time parameters
Tfinal = 10;
dt = 0.05;

% Input parameters for the Gaussian
x0 = -20;
p0 = 3;
sigmaP = 0.2;
tau = 5;

% Set up grid
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Set up kinetic energy matrix by means of three point-formula
Tmat_FD3 = zeros(N, N);
Tmat_FD3(1, [1 2]) = [-2 1];
Tmat_FD3(N, [N-1 N]) = [1 -2];
for n=2:(N-1)
  Tmat_FD3(n, [n-1 n n+1]) = [1 -2 1]; 
end
Tmat_FD3 = Tmat_FD3/h^2;        % Correct duble derivative
Tmat_FD3 = -1/2*Tmat_FD3;       % Correct prefactor

% Set up kinetic energy matrix by means of five point-formula
Tmat_FD5 = zeros(N, N);
Tmat_FD5(1, [1:3]) = [-30 16 -1];
Tmat_FD5(2, [1:4]) = [16 -30 16 -1];
Tmat_FD5(N-1, [(N-3):N]) = [-1 16 -30 16];
Tmat_FD5(N, [(N-2):N]) = [-1 16 -30];
for n=3:(N-2)
  Tmat_FD5(n, [(n-2):(n+2)]) = [-1 16 -30 16 -1]; 
end
Tmat_FD5 = Tmat_FD5/(12*h^2);        % Correct duble derivative
Tmat_FD5 = -1/2*Tmat_FD5;            % Correct prefactor

% Set up kinetic energy matrix by means of the fast Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat_FFT = fft(eye(N));
% Multiply by (ik)^2
Tmat_FFT = diag(-k.^2)*Tmat_FFT;
% Transform back to x-representation
Tmat_FFT = ifft(Tmat_FFT);
Tmat_FFT = -1/2*Tmat_FFT;            % Correct prefactor

% Set up propagators for each approximation
U_FD3 = expm(-1i*Tmat_FD3*dt);
U_FD5 = expm(-1i*Tmat_FD5*dt);
U_FFT = expm(-1i*Tmat_FFT*dt);

% Set up initial Gaussian - analytically (numerical vector)
InitialNorm = nthroot(2/pi, 4) * sqrt(sigmaP/(1-2i*sigmaP^2*tau));
Psi0 = InitialNorm*exp(-sigmaP^2*(x-x0).^2/(1-2i*sigmaP^2*tau)+1i*p0*x);
% Absolute value squared of the time-dependent, analytical Gaussian
% (function varialbe)
PsiAbsDynamic = @(x, t) sqrt(2/pi)*sigmaP/sqrt(1+4*sigmaP^4*(t-tau)^2)...
    *exp(-2*sigmaP^2*(x-x0-p0*t).^2/(1+4*sigmaP^4*(t-tau)^2));

% Initiate plots
figure(1)
% Analytical, exact wave function
plAnalytic = plot(x,abs(Psi0).^2, 'k-', 'linewidth', 2);
hold on
% Three point finite difference approximation
plFD3 = plot(x,abs(Psi0).^2, 'r:', 'linewidth', 1.5);
% Five point finite difference approximation
plFD5 = plot(x,abs(Psi0).^2, 'b-.', 'linewidth', 1.5);
% Dark green plot of the FFT wave function
plFFT = plot(x,abs(Psi0).^2, '--', 'linewidth', 1.5, 'color', [0 0.7 0]);
legend('Analytical', 'FD3', 'FD5', 'FFT')
axis([-L/2 L/2 0 0.2])
set(gca, 'fontsize', 15)
grid on
hold off

% Initiate wave functons and time
Psi_FD3 = Psi0;
Psi_FD5 = Psi0;
Psi_FFT = Psi0;
t = 0;

% Loop which updates wave functions and plots in time
while t < Tfinal
  % Update numerical approximations
  Psi_FD3 = U_FD3*Psi_FD3;
  Psi_FD5 = U_FD5*Psi_FD5;
  Psi_FFT = U_FFT*Psi_FFT;
  
  % Update analytical wave function
  PsiAbsAnalytic = PsiAbsDynamic(x,t);
  
  % Update plots
  set(plAnalytic, 'ydata', PsiAbsAnalytic);
  set(plFD3, 'ydata', abs(Psi_FD3).^2);
  set(plFD5, 'ydata', abs(Psi_FD5).^2);
  set(plFFT, 'ydata', abs(Psi_FFT).^2);
  drawnow
  
  % Update time
  t = t+dt;             
end
