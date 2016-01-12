mooc_load;

LAMBDA = 1;
EPSILON = 1e-7;

yp_val = [];
yp_test = [];

MODEL_NUM = 0;
while true
    name = ['/home/user/Desktop/ML_final_project/model/liblinear/model', num2str(MODEL_NUM), '.mat'];
    fprintf('Model %d (%s)\n', MODEL_NUM, name);
    if(exist(name, 'file'))
        MODEL_NUM = MODEL_NUM + 1;

        load(name);
        SUB_MODEL_NUM = length(model);

        yp_train_sub = zeros(len_train_train, SUB_MODEL_NUM);
        yp_val_sub = zeros(len_train_val, SUB_MODEL_NUM);
        yp_test_sub = zeros(len_test, SUB_MODEL_NUM);
        parfor n = 1:SUB_MODEL_NUM
            fprintf('Submodel %d/%d\n', n, SUB_MODEL_NUM);
            [~, ~, prob] = liblinear.predict(y_train, sparse(x1_train), model(n), '-b 1'); yp_train_sub(:, n) = prob(:, 1);
            [~, ~, prob] = liblinear.predict(y_val, sparse(x1_val), model(n), '-b 1'); yp_val_sub(:, n) = prob(:, 1);
            [~, ~, prob] = liblinear.predict(y_test, sparse(x1_test), model(n), '-b 1'); yp_test_sub(:, n) = prob(:, 1);
        end
        fprintf('Metamodel\n');
        MODEL = liblinear.train(y_train, sparse(yp_train_sub), '-s 0 -q');
        [~, acc, prob] = liblinear.predict(y_val, sparse(yp_val_sub), MODEL, '-b 1'); 
        yp_val(:, end + 1) = prob(:, 1);
        [~, ~, prob] = liblinear.predict(y_test, sparse(yp_test_sub), MODEL, '-b 1'); 
        yp_test(:, end + 1) = prob(:, 1);
    else
        break;
    end
end

save('../../result/liblinear/result_val.mat', 'eid_val', 'yp_val');
save('../../result/liblinear/result_test.mat', 'eid_test', 'yp_test');
