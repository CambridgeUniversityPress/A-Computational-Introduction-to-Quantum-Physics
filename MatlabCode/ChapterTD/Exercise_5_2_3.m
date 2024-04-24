% This script solve the Schrödinger equation for a 1D model
% of an atom exposed to a linearly polarized laser pulse in the
% dipole approximation. It does so by directly propagating the 
% wave packet on a numerical grid by means of a Magnus propagator.
%
% The purely time-dependent laser pulse is modelled as a sin^2-type 
% envelope times a sine-carrier. 
%
% All parameters are given in atomic units.
%
% Inputs for the confining potential
% w     -   the width of the potential
% V0    -   the "height" of the potential (should be negative)
% s     -   "smoothness" parameter
%
% Inputs for the laser pulse 
% Ncycl  -   the number of optical cycles
% Tafter -   time propagation after pulse 
% omega  -   the central frequency of the laser
% E0     -   the strength of the pulse
%
% Numerical input parameters
% dt    - numerical time step
% N     - number of grid points, should be 2^n
% L     - the size of the numerical domain; it extends from -L/2 to L/2
% Xbeyond - the limit beyod which we assume the particle is liberated
% yMax  - the maximal y-value in the display of the wave packet
% Tafter - additional time of simulation - after the pulse is over
%
% All input parameters are hard coded initially.

% Inputs for the confining potential
w = 5;
V0 = -1;
s = 5;

% Inputs for the laser pulse
Ncycl  = 10;
omega = 1.0;
E0 = 0.5;

% Numerical input parameters
dt = .1;
N = 1024;
L = 400;
yMax = 3e-3;
Xbeyond = 7;
Tafter = 25;


% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      

% Confining potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Laser field
Tpulse=Ncycl*2*pi/omega;
Epulse = @(t) (t>0 & t<Tpulse)*E0.*sin(pi*t/Tpulse).^2.*...
    sin(omega*t);

% Set up kinetic energy matrix by means of the fast 
% Fourier transform
k=2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];        % Vector with k-values
% Fourier transform the identity matrix:
Tmat = fft(eye(N));
% Multiply by (ik)^2
Tmat = diag(-k.^2)*Tmat;
% Transform back to x-representation
Tmat = ifft(Tmat);
Tmat = -1/2*Tmat;                   % Correct prefactor
% H0 - the time-independent part of the Hamiltonian
H0 = Tmat + diag(Vpot(x));
% Half propagator corresponding to H0
UhalfH0 = expm(-1i*H0*dt/2);                

% Diagonalize time-independent Hamiltonian
[U E] = eig(H0);                         % Diagonalization
E = diag(E);                             % Extract diagonal energies
[E SortInd] = sort(E);                   % Sort energies
U = U(:,SortInd);                        % Sort eigenvectors and normalize
Psi = U(:,1)/sqrt(h);                    % Normalized initial wave function
disp(['Ground state energy: ', num2str(E(1))])         
Nbound = length(find(E<0));              % The number of bound states
disp(['The potential supports ',num2str(Nbound),' bound states'])
clear Tmat H0;                       % Clear obsolete matrices

% Create plots
% Pulse vector for plotting 
tVector = 0:dt:(Tpulse+Tafter);
PulseVector = Epulse(tVector);

figure(1)
% Plot laser field - and progress
subplot(2,1,1)
plot(tVector, PulseVector, 'k-', 'linewidth', 1.5)
xlabel('Time [a.u.]')
ylabel('E(t) [a.u.]')
axis([0 Tpulse+Tafter -1.1*E0 1.1*E0])
hold on
plPulse = plot(0, Epulse(0), 'r*', 'linewidth', 2);
hold off
xlabel('Time')
set(gca, 'fontsize', 15)
% Wave packet
subplot(2,1,2)
pl1 = plot(x, abs(Psi).^2, 'k-', 'linewidth', 1.5);
xlabel('x [a.u.]')
ylabel('|\Psi(x)|^2 [a.u.]')
set(gca, 'fontsize', 15)
axis([-L/2 L/2 0 yMax])
set(gca, 'fontsize', 15)

%
% Propagate
% First: For  t< Tpulse
t = 0;
while t < Tpulse
  % First half step, H_0
  Psi = UhalfH0*Psi;
  % Interaction
  Psi = diag(exp(-1i*Epulse(t+dt/2)*x*dt))*Psi;
  % Second half step, H0
  Psi = UhalfH0*Psi;
    
  % Plot propagation of wave function and pulse on the fly
  set(plPulse, 'xdata', t, 'ydata', Epulse(t));  
  set(pl1, 'ydata', abs(Psi).^2);  
  drawnow

  % Update time
  t=t+dt;  
end

% Second part of the propagation: t >= Tpluse
Ufull = UhalfH0^2;              % Full time-independent propagator
clear UhalfH0;
while t < Tpulse+Tafter
  % Full step, H_0
  Psi = Ufull*Psi;
  
  % Plot propagation of wave function and pulse on the fly
  set(plPulse, 'xdata', t, 'ydata', 0);  
  set(pl1, 'ydata', abs(Psi).^2);  
  drawnow
  
  % Update time
  t=t+dt;  
end

% Approximate ionization probability
Pbeyond = trapz(x, (abs(x) > Xbeyond).*abs(Psi).^2);
PionPercent = 100*Pbeyond;
disp(['Ionization probability: ', num2str(PionPercent), '%'])