%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 16;  % 4x4 Input Images of Digits
num_labels = 4;          % 4 labels, z, i, o, n   

%% =========== Part 1: Preparing Data =============

All_light = ones(4,4);
X1 = reshape(All_light', 16, 1);

Target_light = [0 0 0 1; 1 0 1 1; 1 1 1 0; 0 1 0 0];
X2 = reshape(Target_light', 16, 1);

input1 = X1';
input2 = X2';

theta = rand(16,1);

for i = 1:16
    if theta(i) > 0.5
       theta(i) = 1;
    else
       theta(i) = -1;
    end
end

goal_history = zeros(32,1);
goal = sum(sum(input2' .* theta)) - 1/2 * sum(sum(input1' .* theta));

for epoch = 1:2
    for i = 1:16
        theta(i) = -theta(i);
        temp = sum(sum(input2' .* theta)) - 1/2 * sum(sum(input1' .* theta));
        if temp > goal
           goal = temp;
        else
           theta(i) = -theta(i);
        end
        goal
        goal_history(i+16*(epoch-1)) = goal;
    end
end





