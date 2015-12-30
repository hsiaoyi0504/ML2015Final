% logistic regression for adaboost
function [weight] = logreg_alg(X,y,u)
  n = 0.01;    % Learning rate
  T = 20;    % Iteration
  ws = [];
  N = size(X, 1);
  d = size(X,2);
  X = [ones(N,1) X];

  for i = 1:T
  	if i == 1
  		wi = zeros(d+1,1);
  	else
  		wi = ws(i-1,:)';
  	end
  	% ein = mean(sign(X*w)~=y)
  	dein = sum( repmat(sigmoid(-y.*(X*wi)).* u .*-y,1,d+1).*X, 1);
  	ws(i,:) = wi' - n*dein;
  end

  weight = ws(end,:)';
