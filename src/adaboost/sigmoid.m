function [z] = sigmoid(s)
	z = 1 ./ (1+exp(-s));
