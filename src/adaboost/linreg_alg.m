% linear regression for adaboost
function [weight] = linreg_alg(X,y,u)
  N = size(X,1);
  d = size(X,2);
  X = [ones(N,1) X];
  weight = (X'*(X.*repmat(u,1,d+1)))^-1 * X' * (y .* u);
