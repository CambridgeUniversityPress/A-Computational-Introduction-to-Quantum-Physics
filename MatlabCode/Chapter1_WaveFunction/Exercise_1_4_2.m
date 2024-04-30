% This script estimates expectation values for the position
% x for four given wave functions. It also plots them.
%
%
%
% Inputs:
%   PsiFunk - The expression for the unnormalized wave function.
%   L       - The extension of the spatial grid 
%   N       - The number of grid points
%
% All inputs, including the expressions for the elementary wave functions,
% are hard coded initially. The four different wave functions are commented
% in and out as desired.

% Unnormalzied wave functions 
% Psi_A:
%PsiFunk = @(x) 1./(1+(x-3).^2).^(3/2);
% Psi_B:
%PsiFunk = @(x) 1./(1+(x-3).^2).^(3/2).*exp(-4*i*x);
% Psi_C:
PsiFunk = @(x) exp(-x.^2);
% Psi_D:
%PsiFunk = @(x) (x+i).*exp(-(x-3*i-2).^2/10);

% Numerical grid parameters
L = 20;
N = 100;

% Set up grid
x = linspace(-L/2, L/2, N);
Psi = PsiFunk(x);                 % Vector with function values

% Normalization
Norm = sqrt(trapz(x, abs(Psi).^2));
Psi = Psi/Norm;

% Make plot
figure(1)
plot(x, abs(Psi).^2, 'k-', 'linewidth', 2)
grid on
set(gca,'fontsize',15)
xlabel('x');
% Check if Psi is complex and, if so, plot real and imaginary contributions
if max(abs(imag(Psi)))>1e-7     % The limit here is somewhat arbitrarily
  hold on
  plot(x, real(Psi).^2, 'b--', 'linewidth', 1.5)
  plot(x, imag(Psi).^2, 'r-.', 'linewidth', 1.5)
  hold off
  legend('|\Psi|^2','(Re \Psi)^2','(Im \Psi)^2')
end

% Calculate expectation value
MeanX = trapz(x, x.*abs(Psi).^2);             % Mean position
% Write result to screen:
disp(['Mean position: ', num2str(MeanX)])