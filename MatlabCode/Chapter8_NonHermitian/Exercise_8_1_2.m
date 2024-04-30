% This script solve the Schrödinger equation for a 1D model of an atom 
% exposed to a linearly polarized laser pulse in the dipole approximation. 
% It does so by directly propagating the wave packet on a numerical grid 
% by means of a Magnus propagator approximated as a spilt operator.
%
% The purely time-dependent laser pulse is modelled as a sin^2-type 
% envelope times a sine-carrier. 
%
% The numerical domaine is truncated by imposing an absorber close to the 
% boundaries. The ammount of absorption is monitored in time and used to 
% estimate the total ionization probability in the end. A squre monomial 
% is used for the absoprtion potential.
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
%
% Inputs for the absorbing potential
% eta   - the absoprtion strength
% Onset - for |x| beyond this value, the absorbin potential is supported
%
% Numerical input parameters
% dt    - numerical time step
% N     - number of grid points, should be 2^n
% L     - the size of the numerical domain; it extends from -L/2 to L/2
% yMax    - the maximal y-value in the display of the wave packet
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
N = 256;
L = 100;
yMax = 3e-3;
Tafter = 120;

% Inputs for the absorber
eta = 1e-2;
Onset = 30;
% Check that Onset is admissible
if Onset > L/2
  error('This value for Onset is too large.')
end


% Set up grid
h = L/(N-1);
x = transpose(linspace(-L/2, L/2, N));      % Column vector

% Confining potential
Vpot = @(x) V0./(exp(s*(abs(x)-w/2))+1);

% Laser field
Tpulse=Ncycl*2*pi/omega;
Epulse = @(t) (t>0 & t<Tpulse)*E0.*sin(pi*t/Tpulse).^2.*...
    sin(omega*t);

% Absorbing potential
Vabs = @(x) eta*(abs(x) > Onset).*(abs(x)-Onset).^2;

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
% H0 - the Hermitian, time-independent part of the Hamiltonian
H0 = Tmat + diag(Vpot(x));
% Heff - H0 augmented with the complex absorbing potential
Heff = H0 - 1i*diag(Vabs(x));
% Half propagator corresponding to H0
UhalfH0 = expm(-1i*Heff*dt/2);                

% Diagonalize time-independent Hamiltonian
[U E] = eig(H0);                         % Diagonalization
E = diag(E);                             % Extract diagonal energies
[E SortInd] = sort(E);                   % Sort energies
U = U(:,SortInd)/sqrt(h);                % Sort eigenvectors and normalize
Psi = U(:,1);                            % Initiate wave function
disp(['Ground state energy: ',num2str(E(1)),'.'])         
Nbound = length(find(E<0));              % The number of bound states
disp(['The potential supports ',num2str(Nbound),' bound states.'])
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
ylabel('Electric field [a.u.]')
axis([0 Tpulse+Tafter -1.1*E0 1.1*E0])
hold on
plPulse = plot(0, Epulse(0), 'r*', 'linewidth', 2);
hold off
xlabel('Time')
set(gca, 'fontsize', 15)

% Wave packet
subplot(2,1,2)
pl1 = plot(x, abs(Psi).^2, 'k-', 'linewidth', 1.5);
% Scaling factor for plot:
ScalingFactor = 0.7*yMax/max(Vabs(x));
hold on
plot(x, ScalingFactor*Vabs(x), 'r--', 'linewidth', 1.5)
hold off
xlabel('x [a.u.]')
ylabel('|\Psi(x)|^2 [a.u.]')
set(gca, 'fontsize', 15)
axis([-L/2 L/2 0 yMax])
set(gca, 'fontsize', 15)

% Allocate vector with the norm
NormVector = zeros(1, length(tVector));

%
% Propagate
% First: For  t< Tpulse
t = 0;
index = 1;
while t < Tpulse
  % First half step, H_0
  Psi = UhalfH0*Psi;
  % Interaction
  Psi = diag(exp(-1i*Epulse(t+dt/2)*x*dt))*Psi;
  % Second half step, H0
  Psi = UhalfH0*Psi;
    
  % Calculate norm
  Norm = trapz(x, abs(Psi).^2);
  NormVector(index) = Norm;
  
  % Plot propagation of wave function and pulse on the fly
  set(plPulse, 'xdata', t, 'ydata', Epulse(t));  
  set(pl1, 'ydata', abs(Psi).^2);  
  drawnow

  % Update time and index
  t=t+dt;  
  index = index + 1;
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
  
  % Calculate norm
  Norm = trapz(x, abs(Psi).^2);
  NormVector(index) = Norm;
  
  % Update time and index
  t=t+dt;  
  index = index + 1;
end

% Approximate ionization probability
Pabsorbed = 1 - Norm;
PionPercent = 100*Pabsorbed;
disp(['Ionization probability: ', num2str(PionPercent), '%.'])

% Plott time-dependent norm
figure(2)
plot(tVector, NormVector, 'k-', 'linewidth', 2)
hold on
xline(Tpulse, 'k--');
hold off
grid on
xlabel('Time [a.u.]')
ylabel('Probability')
set(gca, 'fontsize', 15)