SLOT = 30;

for file = {'train', 'test'}
    eval(['log = log_', file{1}, ';']);
    eval(['enrollment = enrollment_', file{1}, ';']);

    log_interval = [0; find(diff(log{1})); length(log{1})];
    object_map = containers.Map(object{2}, 1:length(object{1}));
    object_course_sel = find(object{3} == 4);

    e_max = find(enrollment{1} >= log{1}(end), 1);
    xa = zeros(e_max, SLOT, length(SOURCE_ORDER) + length(CATEGORY));
    xb = zeros(e_max, 6 + length(SOURCE_ORDER) + length(CATEGORY));
    xb(:, 1:6) = sample(1:e_max, 1:6);
    if(strcmp(file{1}, 'train'))
        y = drop{2}(1:e_max);
    end
    for e = 1:e_max
        eid = enrollment{1}(e);
        log_range = log_interval(e) + 1:log_interval(e + 1);
        log_range_size = range(log_range) + 1;
        if(mod(e, 10) == 0)
            fprintf('On eid = %d / %d\n', eid, enrollment{1}(e_max));
        end
        
        course_time = object{5}(object_course_sel(find(strcmp(object{1}(object_course_sel), enrollment{3}{e}))));
        slot_num = max(ceil((log{2}(log_range) - course_time) / (30 / SLOT)), 1);
        slot_interval = [0; find(diff(slot_num)); length(slot_num)];

        %%
        source_sequential = bsxfun(@eq, log{3}(log_range), 1:length(SOURCE));
        event_sequential = bsxfun(@eq, log{4}(log_range), 1:length(EVENT));
        activity_sequential = source_sequential(:, SOURCE_ORDER) & event_sequential(:, EVENT_ORDER);

        %
        object_exist = cellfun(@(x) object_map.isKey(x), log{5}(log_range));
        category_sequential = zeros(log_range_size, 1);
        category_sequential(object_exist) = object{3}(cellfun(@(x) object_map(x), log{5}(log_range(object_exist))));
        category_sequential = bsxfun(@eq, category_sequential, 1:length(CATEGORY));

        %%
        activity_feature = zeros(SLOT, length(SOURCE_ORDER));
        for s = 1:length(slot_interval) - 1
            activity_feature(slot_num(slot_interval(s) + 1), :) = sum(activity_sequential(slot_interval(s) + 1:slot_interval(s + 1), :));
        end

        %
        category_feature = zeros(SLOT, length(CATEGORY));
        for s = 1:length(slot_interval) - 1
            category_feature(slot_num(slot_interval(s) + 1), :) = sum(category_sequential(slot_interval(s) + 1:slot_interval(s + 1), :));
        end

        xa(e, :, :) = permute([activity_feature, category_feature], [3, 1, 2]);
        xb(e, 7:end) = [sum(activity_feature, 1), sum(category_feature, 1)];
        
        %{
        image = log10([activity_feature, category_feature] + 1) / 2;
        image = repelem([drop{2}(e) * ones(size(image, 1), 1), image, (1 - drop{2}(e) * ones(size(image, 1), 1))], 10, 10);
        imwrite(image, [dataDir, 'feat/', num2str(eid), '_', num2str(drop{2}(e)), '.png']);
        %}
    end

    eval(['xa_', file{1}, '= xa;']);
    eval(['xb_', file{1}, '= xb;']);
    if(strcmp(file{1}, 'train'))
        eval(['y_', file{1}, '= y;']);
    end
end

save('../data.mat', 'xa_train', 'xb_train', 'y_train', 'xa_test', 'xb_test');
