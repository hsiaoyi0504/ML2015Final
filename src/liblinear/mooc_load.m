if(~exist('len', 'var'))
    load('../../data/feat1.mat');

    for attr = {'x1', 'x2', 'x3'}
        eval([attr{:}, ' = zscore(cat(length(size(', attr{:}, '_int)), double(', attr{:}, '_int), double(', attr{:}, '_float)), 0, 1);']);
        eval(['clear ', attr{:}, '_int ', attr{:}, '_float;']);
    end
    y = double(y);

    for attr = {'eid', 'w', 'x1', 'x2', 'x3', 'y'}
        eval([attr{:}, '_size = size(', attr{:}, ');']);

        eval([attr{:}, '_train = zeros([len_train_train, ', attr{:}, '_size(2:end)]);']);
        eval([attr{:}, '_val = zeros([len_train_val, ', attr{:}, '_size(2:end)]);']);
        eval([attr{:}, '_test = zeros([len_test, ', attr{:}, '_size(2:end)]);']);

        eval([attr{:}, '_train(:) = ', attr{:}, '(perm(1:len_train_train), :);']);
        eval([attr{:}, '_val(:) = ', attr{:}, '(perm(len_train_train+1:len_train), :);']);
        eval([attr{:}, '_test(:) = ', attr{:}, '(perm(len_train+1:len), :);']);
        eval(['clear ', attr{:}, ' ', attr{:}, '_size;']); 
    end
end

