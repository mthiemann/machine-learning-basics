function o = computeOutput(Theta1, Theta2, X)
%PREDICT Predict the output of an input given the current weight matrixes of the network
%   p = computeOutput(Theta1, Theta2, X) outputs the predicted label of X given the
%   current weight matrixes of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
h1 = sigmoid([ones(m, 1) X] * Theta1');
o = sigmoid([ones(m, 1) h1] * Theta2');

% =========================================================================


end
