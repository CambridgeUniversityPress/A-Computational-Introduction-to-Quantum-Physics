% This script aims at finding the ground state of a Hamiltonian of a 
% certain potential by means of adiabatic evolution. The system starts out 
% in the ground state of a harmonic oscillator. Then, the potential
% is gradually changed into another one - the one we wish to minimize. If 
% this is done sufficiently slowly, the system will remain in the current 
% ground state.
%
% The evolution is simulated using the split operator technique.
%
% Time inputs
% Nstep  -   the number time steps
% Ttotal -   the duration of interaction
%
% Numerical input parameters
% N     - number of grid points, should be 2^n for FFT's sake
% L     - the size of the numerical domain; it extends from -L/2 to L/2
% 
% Function inputs
% Vinit     - intial, harmonic potential
% Vfinal    - potential to be minimized
% sFunk     - time-dependent function shiftling smootly from 1 to 0
%
% Vpot, not input: sFunk(t)*Vinit(x) + (1-sFunk(t))*Vfinal(x)
%
% All input parameters are hard coded initially

% Spatial inputs
N = 256;
L = 8;

% Time inputs
Nstep  =  500;
Ttotal =  20;

% Window for plotting
W = [-4 4 -.5 1.5];

% Initial potential
Vinit = @(x) 0.5*x.^2;

% Final potential - the one we want to minimize
Vfinal = @(x) (x.^2-1).^2 - x/5;

% Transition function
sFunk = @(t) 0.5*(1 + cos(pi/Ttotal*t));

% Time-dependent Potetial
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
dt = Ttotal/Nstep;

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

% Propagate
t = 0;
while t < Ttotal
  % Half kinetic energy
  Psi = UT_half * Psi;                
  % Potential energy propagator  
  Psi = diag(exp(-1i*Pot(t,x)*dt))*Psi;
  % Half kinetic energy - again
  Psi = UT_half * Psi;
  
  % Plot propagation of wave function and pulse on the fly
  set(plWF, 'ydata', abs(Psi).^2);  
  set(plPot, 'ydata', Pot(t,x));
  drawnow
  axis(W)
  
  % Update time
  t = t+dt;
end


% Determine overlap between final state and ground state
HamFinal = Tmat + diag(Vfinal(x));          % Final Hamiltonian
[B E] = eig(HamFinal);
% Sort and normalize eigenstates
E = diag(E); [E ind] = sort(E); B = B(:, ind); B = B/sqrt(h);
% Ground state
PsiGS = B(:,1);
% Calculate overlap and write result to screen
Overlap = trapz(x, conj(PsiGS).*Psi);
Overlap = abs(Overlap)^2;
disp(['Probability to remain in the ground state: ', ...
    num2str(Overlap*100), ' %.'])