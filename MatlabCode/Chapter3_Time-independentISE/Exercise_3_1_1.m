% This script plots the determinant which is to be set to zero
% to ensure non-trivial solutions for the time-independent 
% Schrödinger equation for a particle of unit mass trapped inside a 
% rectangular well potential.
% 
% There are two input-parameters: The "height" of the well, V_0,
% and the width w of the well. As this is to be a well, not a barrier
% V_0 must be negative
%
% The zero points of the functions indicate admissible, quantized
% energies
%
% The input-parameters are hard-coded initially.

% Input parameters
V0 = -4;        % Negative debth
w = 5;          % Widht

% Energy vector
Npoints = 200;
Energies = linspace(V0, 0, Npoints);

% kappa vector (derived from energy vector)
kappaVector = sqrt(-2*Energies);

% k vector (derived from energy vector)
kVector = sqrt(2*(Energies-V0));

% Determinant corresponding to the symmetric wave function
DetSymm = kappaVector.*cos(kVector*w/2) - kVector.*sin(kVector*w/2);

% Determinant corresponding to the anti-symmetric wave function
DetAntiSymm = kappaVector.*sin(kVector*w/2) + kVector.*cos(kVector*w/2);

% Plot the functions which are to be zero
plot(Energies, DetSymm, 'b-', 'linewidth', 2)
hold on
plot(Energies, DetAntiSymm, 'r--', 'linewidth', 2)
yline(0, 'k-');
hold off
grid on
set(gca, 'fontsize', 15)
xlabel('Energy')

% Estimate roots by brute force and write them to screen
for n = 2:Npoints
  if DetSymm(n)*DetSymm(n-1)<0 | DetAntiSymm(n)*DetAntiSymm(n-1)<0
    disp(['Permissible energy: ', ...
        num2str((Energies(n)+Energies(n-1))/2)])
  end
end
