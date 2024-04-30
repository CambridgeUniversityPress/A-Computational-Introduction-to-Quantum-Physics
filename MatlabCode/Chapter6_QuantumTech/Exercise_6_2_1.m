% This script determines which of the possible transitions between 
% eigenstates within a hydrogen atom produces photons with wavelengths 
% in the visible spectrum.
% Here we will take this to mean wavelengths in the interval between 385 
% an 765 nanometres.
% 
% The wavelenghs and transitions in question are written to screen. The 
% corresponding line spectrum is also plotted. To this end, a function 
% which converts wavelength to RGB code is introduced.
%
% The script does not really take inputs, save the maximum  quantum 
% number n involved.

% Maximum quantum number n
nMax = 20;

% Interval for visible light
LambdaMin = 385;
LambdaMax = 765;

% Natural constants
c = 3.00e8;             % The speed of light
B = 2.179e-18;          % The Bohr constant
h = 6.63e-34;           % The Planck constant (not the reduced one)


% Loop over all initial quantum numbers n1
counts = 0;
% Write heading to screen
disp('Transitions in the visible spectrum:')
for n1 = 2:nMax;
 % Loop over all final quantum numbers
 for n2 = 1:(n1-1);
   % Wavelength of emitted photon - in nanometres
   LambdaNM = h*c/(B*(1/n2^2-1/n1^2))/1e-9;
   if LambdaNM > LambdaMin & LambdaNM < LambdaMax
     counts = counts + 1;
     VisibleLambda(counts) = LambdaNM;
     Transition(counts, [1: 2]) = [n1 n2];
     % Write visible transition and wavelength to screen
     disp(['From n=', num2str(n1), ' to n=', num2str(n2)', ...
         ', wavelength: ', num2str(LambdaNM),' nm.'])
   end
 end
end

% Create line spectrum
figure(1)
% Set black background
set(gca,'color','k')
counter = 1;
for Wl = VisibleLambda
  [R G B] = Wl2RGB(Wl);
  xline(Wl, 'linewidth', 3, 'color', [R G B]);
  counter = counter + 1;
  if counter > 1
    hold on
  end
end
axis([380 780 0 1])
hold off
set(gca,'fontsize',20)
yticks([])
xlabel('\lambda [nm]')


% Function which converts wavelength to RGB (red-green-blue) code
function [R G B] = Wl2RGB(wl)

  % Wl must be given in nanometres. Moreover, it 
  % must reside between 380 nm and 780 nm.
  % This function is an adaption of source code found here:
  % http://www.physics.sfasu.edu/astro/color/spectra.html

  % Check if we are in the visible part
  if wl < 380 | wl > 780
    error('This wavelength is not visible.')
  end

  % Determine colour by piecewise linear interpolation
  if wl >= 380 & wl <= 440  
    R = -1.*(wl-440.)/(440.-380.);
    G = 0;
    B = 1;
  elseif wl > 440 & wl <= 490
    R = 0;
    G = (wl-440.)/(490.-440.);
    B = 1;
  elseif wl > 490 & wl <= 510
    R = 0;
    G = 1;
    B = -1.*(wl-510.)/(510.-490.);
  elseif wl > 510 & wl <= 580
    R = (wl-510.)/(580.-510.);
    G = 1;
    B = 0;
  elseif wl > 580 & wl <= 645
    R = 1;
    G = -1.*(wl-645.)/(645.-580.);
    B = 0;
  elseif wl > 645 & wl <= 780  
    R = 1;
    G = 0;
    B = 0;
  end

  % Reduce brightness near the edges
  if wl > 700
    SSS=.3+.7* (780.-wl)/(780.-700.);
    R = SSS*R;
    G = SSS*G;
    B = SSS*B;
  elseif wl < 420
    SSS=.3+.7*(wl-380.)/(420.-380.);
    R = SSS*R;
    G = SSS*G;
    B = SSS*B;
  end
end
