%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 16;  % 4x4 Input Images of Digits
hidden_layer_size = 4;   % 10 hidden units
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

z_label = [1, 0, 0, 0];
i_label = [0, 1, 0, 0];
o_label = [0, 0, 1, 0];
n_label = [0, 0, 0, 1];

z_expand = z;
i_expand = i;
o_expand = o;
n_expand = n;
%z_expand

for x = 1:99
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

y = zeros(1,400);
y(1:100) = 1;
y(101:200) = 2;
y(201:300) = 3;
y(301:400) = 4;
y = y';

%% ================ Part 2: Initializing Parameters ================
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 3: Training NN ===================
%  We have now implemented all the code necessary to train a neural 
%  network. To train our neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 3);

lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  us compute the training set accuracy.

z_test = [1 1 1 1; 0 0 1 0; 0 1 0 0; 1 1 1 1];
i_test = [0 1 1 0; 0 1 1 0; 0 1 1 0; 0 1 1 0];
o_test = [1 1 1 1; 1 0 0 1; 1 0 0 1; 1 1 1 1];
n_test = [1 0 0 1; 1 1 0 1; 1 0 1 1; 1 0 0 1];
%z

z_test = reshape(z_test', 16, 1);
i_test = reshape(i_test', 16, 1);
o_test = reshape(o_test', 16, 1);
n_test = reshape(n_test', 16, 1);
%z

z_test_expand = z_test;
i_test_expand = i_test;
o_test_expand = o_test;
n_test_expand = n_test;
%z_expand

for x = 1:19
    z_test_expand = [z_test_expand, abs(z_test+rand(size(z_test))*0.1-0.05)];
    i_test_expand = [i_test_expand, abs(i_test+rand(size(i_test))*0.1-0.05)];
    o_test_expand = [o_test_expand, abs(o_test+rand(size(o_test))*0.1-0.05)];
    n_test_expand = [n_test_expand, abs(n_test+rand(size(n_test))*0.1-0.05)];
end
%display(z_test_expand)
%size(z_test_expand)

input_test = [z_test_expand, i_test_expand, o_test_expand, n_test_expand];
X_test = input_test';
%size(input_test)
pred = predict(Theta1, Theta2, X_test);

y_test = zeros(1,80);
y_test(1:20) = 1;
y_test(21:40) = 2;
y_test(41:60) = 3;
y_test(61:80) = 4;
y_test = y_test';

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);








