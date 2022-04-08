clc
clear
close all

fprintf('Plotting the given data ...\n')
x = (0:1:13)';
y = ([202.36 239.03 280.71 309.12 323.15 332.78 328.45 306.40 287.36 247.97 202.89 161.11 93.68 20.78])';
m = length(y); % number of training examples in the training set

plot(x, y, 'rx', 'MarkerSize', 10);         
ylabel('y');                 
xlabel('x');       

fprintf('The Program is paused. Please press enter to continue.\n');
input('');

x = [ones(m, 1), x, ones(m, 1).*x.^2]; % Moodifying x so it is in the 2nd degree polynomial
theta = zeros(3, 1); % initializing the fitting parameters

iterations = 15000;
alpha = 0.0003; % alpha value

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost

J = Cost(x, y, theta);
fprintf('With theta = [0 ; 0 ; 0]\nCost computed = %f\n', J);
fprintf('Program paused. Press enter to continue.\n');
input('');

fprintf('\nRunning Gradient Descent ...\n') % Run gradient descent
[theta, J] = gradientDescent(x, y, theta, alpha, iterations);
% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

fprintf('\n');
fprintf('The cost function = %f\n', J(end));
fprintf('\n')

% Plot the linear fit
hold on; % keep previous plot visible
plot(x(:,2), x*theta, '-')
legend('The data set', 'Prediction with Gradient descent')
hold on % don't overlay any more plots on this figure

fprintf('The Program is paused. Please press enter to continue.\n');
input('');

fprintf('\nRunning NGD ...\n') % Run normal Equation
[theta, J] = NGD(x, y, theta, iterations);
% print theta to screen
fprintf('Theta found by NGD:\n');
fprintf('%f\n', theta);

fprintf('\n');
fprintf('The cost function = %f\n', J(end));
fprintf('\n')

plot(x(:,2), x*theta, '-', 'color', 'b', 'DisplayName','Prediction with Natural gradient')
hold off % don't overlay any more plots on this figure


function J = Cost(x, y, theta)
J = 0;
m = length(y);
hyp = x*theta;
J = J + (1/(2*m))*sum((hyp - y).^2);
end

function [theta, J_hist] = gradientDescent(X, y, theta, alpha, iterations)
J_hist = zeros(iterations, 1);
m = length(y);
 for iter = 1:iterations
     
    hyp = X*theta;   
    theta = theta - (alpha/m)*((hyp - y)'*X)' ;
    
 end
 % Save the cost funvtion J for the last iteration, Very Important! 
    J_hist(iter) = Cost(X, y, theta); 
end

function [theta, J_hist1] = NGD(X, y, theta, iterations)
J_hist1 = zeros(iterations, 1);
m = length(y);
alpha = 7.76;
 for iter = 1:iterations
     
    hyp = X*theta;   
    %F= 1 / ( 0.001 + abs((1/m)*((hyp - y)'*X)')^2);  
    g = (((hyp - y)'*X)')/m;
    gs = size(g);
    I = eye(gs(1));
    beta = 0.1;
    F = ( g *(g') + beta*I )';
    theta = theta - (alpha) * inv(F) * g;
    
 end
 % Save the cost funvtion J for the last iteration, Very Important! 
    J_hist1(iter) = Cost(X, y, theta); 
end
