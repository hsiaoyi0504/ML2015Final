load('../../data/feat1.mat');

len_val = round(0.1 * len_train);
perm = randperm(len_train);

eid = x1_int(len_train+1:end, 1);
x1 = zscore([double(x1_int), double(x1_float)]);
y = double(y);

x1_val   = x1(perm(1:len_val), :);
x1_train = x1(perm(len_val+1:len_train), :);
x1_test  = x1(len_train+1:end, :);
y_val    = y(perm(1:len_val));
y_train  = y(perm(len_val+1:len_train));
y_test   = y(len_train+1:end);

model = train(y_train, sparse(x1_train), '-s 2');
[~, accuracy, ~] = predict(y_train, sparse(x1_train), model);
[~, accuracy, ~] = predict(y_val, sparse(x1_val), model);
[y_test, accuracy, ~] = predict(y_test, sparse(x1_test), model);

dlmwrite('../../result/liblinear/result_1_track_2.csv', [eid, y_test], 'precision', '%d');
