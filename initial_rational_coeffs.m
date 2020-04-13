% Compute best rational approximation of function of a given type to obtain
% initial coefficients for the rational layers

% Select the type of the rational: (degreeP, degreeQ)
degreeP = 3;
degreeQ = 2;

% Define the ReLU function in Chebfun
f = chebfun(@(x)x*(x>0),'splitting','on');

% Compute the best rational approximation to f
[p,q,r] = minimax(f, degreeP, degreeQ, 'silent');

% Plot the function f (in blue) and the rational approximation (in red)
xx = linspace(-1,1,3000);
LW = 'linewidth'; FS = 'fontsize';
plot(xx,f(xx),"Linewidth",2)
hold on
plot(xx,r(xx),"Linewidth",2)
hold off
set(gca,'FontSize',18)
shg

% Get the monomial coefficients of the numerator and denominator
P = poly(p);
Q = poly(q);
% Normalize to impose Q_0 = 1
P = P/Q(end);
Q = Q/Q(end);
% Padd the array with zeros if needed and print the initializer coefficients
alpha_initializer = [zeros(1,degreeP+1-length(P)), P]
beta_initializer = [zeros(1,degreeQ+1-length(Q)), Q]