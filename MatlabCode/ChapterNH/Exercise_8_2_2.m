% This script plots the determinant of a coefficient matrix of a set 
% of equations which determines the resonance energies for a potential 
% consisting of two rectangular barriers. 
%
% Inputs are
% For the barriers:
% V0 - The height of each barier (it is the same for both)
% d - half the distance between the barriers
% w - the width of each barrier.
%
% For the energy ranges:
% ImEMax - the maximal absolute value of the negative imaginary energy
% dE - resolution in the enegy grid; it is same for the real and the 
% imaginary part.

% Physical input parameters
V0 = 4;     
d = 3;     
w = .5;     

% Energy grid
ReEMax = V0;
ImEMax = 0.25;          % Maximal value of negative imaginary energy
dE = .005;

% Vectors of real and imaginary energy
ReEvector = 0:dE:ReEMax;
ImEvector = -ImEMax:dE:0;

% Allocate
DetMatSymm = zeros(length(ReEvector), length(ImEvector));
DetMatAntiSymm = zeros(length(ReEvector), length(ImEvector));

% Define parmeters, onset and offset of the barriers
dPlus = d + w/2;
dMinus = d - w/2;


% Construct matrices with values of determinants
% Symmetric case
ReInd = 1;
for realE=ReEvector
  ImInd = 1;
  for imagE = ImEvector  
    % Energy-dependent parameters
    E = realE+i*imagE;            % Complex energy
    kappa = sqrt(2*(V0-E));       
    k = sqrt(2*E);
    
    % Coefficient matrix
    Mat = [cos(k*dMinus) -exp(-kappa*dMinus) -exp(kappa*dMinus) 0
    -k*sin(k*dMinus) kappa*exp(-kappa*dMinus) -kappa*exp(kappa*dMinus) 0
    0 exp(-kappa*dPlus) exp(kappa*dPlus) -exp(i*k*dPlus)
    0 -kappa*exp(-kappa*dPlus) kappa*exp(kappa*dPlus) -i*k*exp(i*k*dPlus)];
    
    % Calculate and assign determinant (in absolute value)
    DetMatSymm(ReInd,ImInd) = abs(det(Mat));
    ImInd = ImInd + 1;              % Update index
  end
  ReInd = ReInd + 1;                % Update index
end

% Plot logarithm of |det(A)| for the symmetric case
figure(1)
subplot(2, 1, 1)
% Plot the logarithm - to highlight the minima
pcolor(ReEvector, ImEvector, log(DetMatSymm).')
shading interp
set(gca,'fontsize',15)
xlabel('Re E')
ylabel('Im E')

% Anti-symmetric case - identical except for element 1,1 and 2,1 in the 
% coefficient matrix
ReInd = 1;
for realE = ReEvector
  ImInd = 1;
  for imagE = ImEvector  
    % Energy-dependent parameters
    E = realE+i*imagE;            % Complex energy
    kappa = sqrt(2*(V0-E));       
    k = sqrt(2*E);
    
    % Coefficient matrix
    Mat = [sin(k*dMinus) -exp(-kappa*dMinus) -exp(kappa*dMinus) 0
    k*cos(k*dMinus) kappa*exp(-kappa*dMinus) -kappa*exp(kappa*dMinus) 0
    0 exp(-kappa*dPlus) exp(kappa*dPlus) -exp(i*k*dPlus)
    0 -kappa*exp(-kappa*dPlus) kappa*exp(kappa*dPlus) -i*k*exp(i*k*dPlus)];
    
    % Calculate and assign determinant (in absolute value)
    DetMatAntiSymm(ReInd,ImInd) = abs(det(Mat));
    ImInd = ImInd + 1;              % Update index
  end
  ReInd = ReInd + 1;                % Update index
end

% Plot |det(A)| for the antisymmetric/odd case
subplot(2, 1, 2)
% Plot the logarithm - to highlight the minima
pcolor(ReEvector, ImEvector, log(DetMatAntiSymm).')
shading interp
set(gca,'fontsize',15)
xlabel('Re E')
ylabel('Im E')