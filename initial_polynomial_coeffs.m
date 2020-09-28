% Compute best polynomial approximation of function of a given type to obtain
% initial coefficients for the polynomial layers

% Select the type of the polynomial
degreeP = 3;

% Define the ReLU function in Chebfun
f = chebfun(@(x)x*(x>0),'splitting','on');

% Compute the best polynomial approximation to f
p = minimax(f, degreeP, 'silent');

% Plot the function f (in blue) and the polynomial approximation (in red)
xx = linspace(-1,1,3000);
LW = 'linewidth'; FS = 'fontsize';
plot(xx,f(xx),"Linewidth",2)
hold on
plot(xx,p(xx),"Linewidth",2)
hold off
set(gca,'FontSize',18)
shg

% Get the monomial coefficients of the polynomial
P = poly(p);

% Padd the array with zeros if needed and print the initializer coefficients
alpha_initializer = [zeros(1,degreeP+1-length(P)), P]
