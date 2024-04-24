% This script aims to find the minimum of a function by imposing this
% function as the potential of a Hamiltion for which the system
% undergoes adiabatic evolution. The system starts out 
% in the ground state of a harmonic oscillater. Then, the potential
% is gradually changed into another one - the one we wish to minimize. 
% If this is done sufficiently slowly, the system will remain in the 
% current ground state.
% 
% Next, in order to localize the minimum, we allow for the mass of the
% quantum particle to increas - rendering a ground state with a much
% smaller width.
%
% In both these two phases, the evolution is simulated using split 
% operator techniques. The potential and the kinetic energy change roles
% in the two phases.
%
% Time inputs
% Nstep   - the number time steps in the first phase
% Tphase1 - the duration of first phase
% Tphase2 - the duration of the second phase, the one with increasing mass
%
% Numerical input parameters
% N     - number of grid points, should be 2^n for FFT's sake
% L     - the size of the numerical domain; it extends from -L/2 to L/2
% 
% Function inputs
% Vinit     - intial, harmonic potential
% Vfinal    - potential to be minimized
% sFunk     - time-dependent function shiftling smootly from 1 to 0
% mFunk     - the time-dependent mass - for phase 2
%
% All input parameters are hard coded initially.
%

%
% Numerical input parameters
%
% Spatial inputs
N = 256;
L = 8;

% Time inputs
Nstep  =  500;
Tphase1 = 20;
Tphase2 = 150;

% Window for plotting
W = [-4 4 -.5 4];

% Initial potential
Vinit = @(x) 0.5*x.^2;

% Final potential - the one we want to minimize
Vfinal = @(x) (x.^2-1).^2 - x/5;

% Transition functions
% To change potential
sFunk = @(t) 0.5*(1 + cos(pi/Tphase1*t));
% To change kinetic energy
mFunk = @(t) 1 + 0.01*t^2;

% Time-dependent potential, phase 1
Pot = @(t,x) sFunk(t).*Vinit(x) + (1-sFunk(t)).*Vfinal(x);

% Set up the grid.
x = linspace(-L/2, L/2, N)';
h = L/(N-1);

% Set up kinetic energy matrix by means of the fast Fourier transform
k = 2*pi/(N*h)*[0:(N/2-1), (-N/2:-1)];          % Vector with k-values
% Fourier transform the identity matrix:
Tmat = fft(eye(N));
% Multiply by (ik)^2
Tmat = diag(-k.^2)*Tmat;
% Transform back to x-representation
Tmat = ifft(Tmat);
Tmat = -1/2*Tmat;            % Correct prefactor

% Time step
dt = Tphase1/Nstep;

% Initiate time
t=0;

% Initial state, ground state of the harmonic oscillator with k=1
Psi = pi^(-.25)*exp(-x.^2/2);                              

% Create plots
figure(1)
plWF = plot(x,abs(Psi).^2, 'k-', 'linewidth', 3);
hold on
plPot = plot(x, Vinit(x), 'b', 'linewidth', 2);
hold off
xlabel('Position')
set(gca, 'fontsize', 15)
grid on
axis(W)

% Half propagator for kinetic energy
UT_half = expm(-1i*Tmat*dt/2);    

% Propagate phase 1
t = 0;
while t < Tphase1
  % Half kinetic energy
  Psi = UT_half * Psi;              
  % Potential energy propagator
  Psi = diag(exp(-1i*Pot(t,x)*dt))*Psi;
  % Half kinetic energy
  Psi = UT_half * Psi;
  
  % Plot propagation of wave function and pulse on the fly
  set(plWF, 'ydata', abs(Psi).^2);  
  set(plPot, 'ydata', Pot(t,x));
  drawnow
  axis(W)
  
  % Update time
  t = t+dt;
end


% Entering pahse 2, where the potential stays constant but the mass,
% and thus also the kinetic energy, changes in time
%
% Half propagator for potential
Upot_half = diag(exp(-1i*Vfinal(x)*dt/2));

% Propagate
while t < Tphase1 + Tphase2
  % Half potential energy
  Psi = Upot_half * Psi;              
  % Kinetic energy propagator
  Mass = mFunk(t-Tphase1);
  UT = diag(exp(-1i*dt*(k.^2)/(2*Mass)));
  Phi = fft(Psi);
  Phi = UT*Phi;
  Psi = ifft(Phi);
  % Half potential energy
  Psi = Upot_half * Psi;              
  
  % Plot propagation of wave function and pulse on the fly
  set(plWF, 'ydata', abs(Psi).^2);  
  drawnow
  axis(W)
  
  % Update time
  t = t+dt;
end

