function g = softmax(z)
%SOFTMAX Compute softmax functoon
%   J = SOFTMAX(z) computes the softmax of z.

temp = exp(z);
g = temp./sum(temp);

end
