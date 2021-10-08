%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 16;  % 4x4 Input Images of Digits
num_labels = 4;          % 4 labels, z, i, o, n   

%% =========== Part 1: Preparing Data =============
% we start the exercise by create our own input images

z = [1 1 1 1; 0 0 1 0; 0 1 0 0; 1 1 1 1];
i = [0 1 1 0; 0 1 1 0; 0 1 1 0; 0 1 1 0];
o = [1 1 1 1; 1 0 0 1; 1 0 0 1; 1 1 1 1];
n = [1 0 0 1; 1 1 0 1; 1 0 1 1; 1 0 0 1];
%z

z = reshape(z', 16, 1);
i = reshape(i', 16, 1);
o = reshape(o', 16, 1);
n = reshape(n', 16, 1);
%z

z_expand = z;
i_expand = i;
o_expand = o;
n_expand = n;
%z_expand

for x = 1:999
    z_expand = [z_expand, abs(z+rand(size(z))*0.1-0.05)];
    i_expand = [i_expand, abs(i+rand(size(i))*0.1-0.05)];
    o_expand = [o_expand, abs(o+rand(size(o))*0.1-0.05)];
    n_expand = [n_expand, abs(n+rand(size(n))*0.1-0.05)];
end
%display(z_expand)
%size(z_expand)

input = [z_expand, i_expand, o_expand, n_expand];
X = input';
%size(input)

y = zeros(1,4000);
y(1:1000) = 1;
y(1001:2000) = 2;
y(2001:3000) = 3;
y(3001:4000) = 4;
y = y';

%% ================ Part 2: One-vs-All Training ====================
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



