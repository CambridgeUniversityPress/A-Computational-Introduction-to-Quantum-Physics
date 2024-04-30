% This script plots the relativistic and non-relativistic kinetic energy 
% functions of momentum and energy. The former is done in two ways - in 
% SI units for an object of 1 kg mass and more genrically with (p/mc)^2 
% as argument and with mc^2 as energy unit.
% 
% 500 points have been use for each of the plots. When the momentum is
% the argument, we have set 2mc as upper limit, while the velocity plot
% goes up to almost c, the speed of light.

% Constants (SI units)
c = 3.00e8;
m = 1;

%
% Plot against momentum - SI
%
% Momentum vector
p = linspace(0, 2*m*c, 500);
% Energy
T_nonrel = p.^2/(2*m);
T_rel = m*c^2*(sqrt(1+p.^2/(m*c)^2)-1);

figure(1)
plot(p, T_nonrel, 'b-', 'linewidth', 3)
hold on
plot(p, T_rel, 'r-', 'linewidth', 3)
hold off
grid on
xlabel('Momentum [kg m/s]')
ylabel('Energy [J]')
legend('Non-relativistic', 'Relativistic', 'location', 'northwest')
set(gca, 'fontsize', 15)

%
% Plot against momentum - generic units
%
% 'Momentum' vector
x = linspace(0, 2^2, 500);
% Energy
T_nonrel = x/2;
T_rel = (sqrt(1+x)-1);

figure(2)
plot(x, T_nonrel, 'b-', 'linewidth', 3)
hold on
plot(x, T_rel, 'r-', 'linewidth', 3)
hold off
grid on
xlabel('(p/mc)^2')
ylabel('Energy [mc^2]')
%legend('Non-relativistic', 'Relativistic', 'location', 'northwest')
set(gca, 'fontsize', 15)

% Plot against velocity
% Velocity vector
v = linspace(0, 0.99*c, 500);
% Energy
T_nonrel = 0.5*m*v.^2;
T_rel = m*c^2*(1./sqrt(1-(v/c).^2)-1);

figure(3)
plot(v, T_nonrel, 'b-', 'linewidth', 3)
hold on
plot(v, T_rel, 'r-', 'linewidth', 3)
xline(c/sqrt(5), 'k--');
hold off
grid on
xlabel('Velocity [m/s]')
ylabel('Energy [J]')
axis([0 c 0 3e17])
legend('Non-relativistic', 'Relativistic', 'location', 'northwest')
set(gca, 'fontsize', 15)