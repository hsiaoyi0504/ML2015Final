clear
warning off;
data = csvread('../../data/ML_final_project/sample_train_x.csv',1,0);
y_data = csvread('../../data/ML_final_project/truth_train.csv');
data = [data(:,2:end) y_data(:,2)];

d = size(data,2)-1;
row_num = size(data,1);
[trainInd, valInd, testInd] = dividerand(row_num,0.6,0,0.4);
train_data = data(trainInd,:);
N = size(train_data,1);

test_data = data(testInd,:);
testX = test_data(:, 1:d);
testy = test_data(:, d+1);
testN = size(testX,1);

u1 = ones(N,1) ./ N;  % U for adaboost
us = [u1];          % U array
alpha = [];         % alpha for adaboost
ws = [];            % w array for logistic regression
eins = [];
eouts = [];

% run iteration
ITER_COUNT = 100;
h = waitbar(0, 'Training ><');
for T = 1:ITER_COUNT
  % initialize
  u = us(:,T);
  X = train_data(:,1:d);
  y = train_data(:,d+1);
  y(y==0)= -1;

  w = linreg_alg(X, y, u);
  pred = sign([ones(N,1) X]*w);
  et = sum(u .* (pred~=y)) / sum(u);
  diamond = sqrt((1-et)/et);
  u(y == pred) = u(y == pred) ./ diamond;
  u(y ~= pred) = u(y ~= pred) .* diamond;
  ws = [ws w];
  alpha = [alpha log(diamond)];
  us = [us u];

  G = zeros(N,1);
  G2 = zeros(testN,1);
  for i = 1:size(alpha,2)
    G = G + alpha(i).* [ones(N,1) X]*ws(:,i);
    G2 = G2 + alpha(i).*[ones(testN,1) testX]*ws(:,i);
  end
  eins = [eins mean(sign(G)~=y)];
  eouts = [eouts mean(sign(G2)~=testy)];
  waitbar(T / ITER_COUNT);
end
close(h)

% plot the ein and eout
figure
plot(1:ITER_COUNT, eins, 1:ITER_COUNT, eouts);
