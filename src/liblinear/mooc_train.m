mooc_load;

MODEL_NUM = 128;
SUB_MODEL_NUM = 32;
SEL_RATIO = 0.5;

sel_num = round(SEL_RATIO * len_train_train);
file = 0;

for m = 1:MODEL_NUM
    while true
        name = ['/home/user/Desktop/ML_final_project/model/liblinear/model', num2str(file), '.mat'];
        if(~exist(name, 'file'))
            break;
        else
            file = file + 1;
        end
    end
    fprintf('Model %d/%d (%s)\n', m, MODEL_NUM, name);

    parfor n = 1:SUB_MODEL_NUM
        fprintf('Submodel %d/%d\n', n, SUB_MODEL_NUM);

        sel = randi(len_train_train, sel_num, 1);
        model(n) = liblinear.train(y_train(sel, :), sparse(x1_train(sel, :)), '-s 0 -q');
    end

    save(name, 'model');
end
