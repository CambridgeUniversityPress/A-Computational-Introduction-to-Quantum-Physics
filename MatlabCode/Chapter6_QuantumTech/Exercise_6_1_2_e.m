% This scripts uses current data from a (fictious) scanning tunneling 
% microscipe (STM) setup to determine the shape of a surface. It does so 
% by using the WKB-approximation to estimate the tunneling rate.
% 
% The current data is given by the file Current.dat, in which the first 
% column is the distance from the needle's startig point along one 
% particular direction along the surface. This quantity is given in 
% Ångström. The second column is the measured current - in arbitrary 
% units.
%
% Finally, the surface data is interpolated to render a more smooth 
% surface.
% 
% The input parameters are
% U     - The voltage between needle and surface - in Volt
% V0    - The work function; energy gap between metal and 
% needle - in eV
% E     - The energy of the conductance electrons - in eV

% Inputs
U = 1;      % Voltage - in V
E = 0.5;    % Energy - in eV
V0 = 4.5;   % Work function - in eV

% Constants
m = 9.109e-31;       % Electron mass - in kg
hbar = 1.055e-34;    % Reduced Planck constant - in Js
e = 1.602e-19;       % Elemetary charge - in C

% Convert inputs to SI
eU = e*U;
EinSI = E*e;
V0inSI = V0*e;

% Read the input
CurrentData = load("Current.dat");
Ydata = CurrentData(:,1).';
Idata = CurrentData(:,2).';

% Prefactor in formula
Prefactor = 3*eU*hbar/(4*sqrt(2*m)*((V0inSI-EinSI)^(3/2)- ...
    (V0inSI-EinSI-eU)^(3/2)));

% Vector with f(y)
Fdata = Prefactor*log(Idata);
% Set mean value to zero and convert from metres to Ångström
Fdata = Fdata-mean(Fdata);
Fdata = Fdata/1e-10;

% Interploate (with default number of points)
Ydense = linspace(min(Ydata), max(Ydata));
Fdense = spline(Ydata, Fdata, Ydense);

% Plot the shape of the surface
figure(1)
plot(Ydense, Fdense, 'k-', 'linewidth', 2)
xlabel('y [Å]')
ylabel('f(y) [Å]')
set(gca, 'fontsize', 15)
axis equal              % Ensure equal units on axes