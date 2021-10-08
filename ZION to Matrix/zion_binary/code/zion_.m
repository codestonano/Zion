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


y = zeros(4,400);
y(1,1:100) = 1;
y(2,101:200) = 1;
y(3,201:300) = 1;
y(4,301:400) = 1;
y = y';

y_label = zeros(400,1);
y_label(1: 100) = 1;
y_label(101: 200) = 2;
y_label(201: 300) = 3;
y_label(301: 400) = 4;

theta_goal = [1 -1 1 1; 1 1 1 -1; 
              1 1 1 -1; 1 -1 1 1; 
             -1 -1 1 1; -1 1 -1 1;
              1 1 -1 -1; -1 -1 1 1;
             -1 -1 1 1; 1 1 -1 -1;
             -1 1 -1 1; -1 -1 1 1;
              1 -1 1 1; 1 1 1 -1;
              1 1 1 -1; 1 -1 1 1];

theta = rand(16,4);
for a = 1:16
    for b = 1:4
        if theta(a,b) < 0.5 
           theta(a,b) = -1;
        else
           theta(a,b) = 1;
        end
    end
end

Accuracy_history = zeros(8,1);
J_history = zeros(8,1);

for i = 1:8
    theta(2*i-1,:) = theta_goal(2*i-1,:);
    theta(2*i,:) = theta_goal(2*i,:);
    h = softmax(X*theta);
    J = -sum(sum(y.*log(h)+(1-y).*log(1-h)))./length(y);
    J_history(i) = J;
    [maxx, pp] = max(h');
    pred = pp';
    Accuracy_history(i) = mean(double(pred == y_label)) * 100;
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_label)) * 100);
end
