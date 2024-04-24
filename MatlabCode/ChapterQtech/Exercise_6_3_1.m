% This script estimates the emitted radiation corresponding to the 
% x-component of a magnetic dipole of a spin 1/2-particle exposed to 
% a mangetic field. It is a model for an NMR setup. The magnetic field 
% has a static part oriented along the z-axis, and an oscillating part 
% point along the x-axis.
%
% The emitted radiation is assumed to be proportional to the time-average 
% of the the double time-derivative of the expectation value of the 
% magnetic dipole moment.
%
% Physical inputs - given in SI units
% Bz    - the strength of the static magnetic field
% B0    - the amplitude of the oscillating field
% Cycles- number of cycles of the driving field 
% fMin  - the minimal frequency of the oscillating field - in MHz
% fMax  - the maximal frequency of the oscillating field - in MHz
% df    - step size in frequency - in MHz
%
% Numerical inputs
% StepsPerCycle - Number of temporal steps  size used to estimate the 
% double derivative and numerical integral (fixed by fMax here).
%
% Natural constants
% g     - the proton's g-factor
% m     - the electron mass
% e     - the elementary charge
%
% All inputs are hard coded initially

% Inputs
% Magnetic field strengths - given in T
Bz = 5;
B0 = 0.01;
% The number of cycles - determines interaction time
Cycles = 100;
% Frequency parameters - given in MHz
fMin = 100;
fMax = 300;
df = 0.1;

% Constants
g = 5.5857;         % G-factor for the magnetic moment
m = 1.673e-27;      % Proton mass
hbar = 1.055e-34;   % Reduced Planck constant
e = 1.602e-19;      % Elementary charge

% Numercal time-increment
StepsPerCycle = 100;

% Derived parameters
E = g*e*hbar/(2*m)*Bz;
Omega = -g*e*hbar/(4*m)*B0;

% Vector with frequencies
fVector = fMin:df:fMax;
omegaVector = fVector*1e6*2*pi;     % Angular frequency - in Hz

% Initialize vector with intensities of emitted radiation
len = length(omegaVector);
IntensityVector = zeros(1, len);

% Loop over all input frequencies
index = 1;
for omega = omegaVector
  % Assign detuning
  delta = E/hbar - omega;
  
  % Call function to determine radiated intensity
  IntensityVector(index) = Intensity(omega, delta, Omega, ...
      Cycles, StepsPerCycle);
  
  % Update index
  index = index + 1;
end

% Plot response - with resonance frequency
figure(1)
plot(fVector, IntensityVector, 'k-', 'linewidth', 2)
% Resonance frequency - in MHz
fRes = E/(2*pi*hbar*1e6);
disp(['Resonance frequency: ', num2str(fRes),' MHz.'])
hold on
xline(fRes, 'r--', 'linewidth', 1.5);
hold off
xlabel('Frequency [MHz]')
ylabel('Intensity [Arbitrary units]')
yticks([])
grid on
set(gca, 'fontsize', 15)

% Function to determine Intensity
function Mean = Intensity(omega, delta, Omega, Cycles, StepsPerCycle)
  
  % Generalized rabi frequency
  OmegaG = sqrt(delta^2 + abs(Omega)^2);
  
  % Fix resolution in numerical estimates
  StepsPerCycle = 100;              % Time steps per optical cycle
  Tfinal = Cycles*2*pi/omega;       % Interaction duration
  t = linspace(0, Tfinal, StepsPerCycle*Cycles);
  dt = t(2)-t(1);
  
  % Calculate amplitudes for spin up and down as functions
  % of time
  a = exp( i*omega*t/2).*(cos(OmegaG*t/2) + ...
      i*delta/OmegaG*sin(OmegaG*t/2));
  b = exp(-i*omega*t/2)*Omega/OmegaG.*sin(OmegaG*t/2);
  
  % Dipole moment in x-direction
  DipoleX = real(a.*conj(b));
  
  % Determine double derivative by means of the three-point
  % finite differencd scheme
  DipoleX_DoubleDeriv = (DipoleX(1:(end-2))-2*DipoleX(2:(end-1))...
      +DipoleX(3:end))/dt^2;
  
  % Calculate average radiation itensity (proportional to
  % the square of the double derivative of the dipole moment)
  Mean = 1/Tfinal*dt*sum(DipoleX_DoubleDeriv.^2);
end