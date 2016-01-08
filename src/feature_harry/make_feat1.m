% enrollment_attr / course_attr
% enrollment_attr / user_attr

if(~exist('object', 'var'))
    load('../../data/data.mat');
end

HOUR = 1 / 24;
MINUTE = HOUR / 60;
SECOND = MINUTE / 60;

TIME_BIN = 30;

SESS_BIN = 30;
SESS_IDLE_THRESH = 30 * MINUTE;
SESS_TIME_HIST_BIN = linspace(-2, 3, SESS_BIN);
SESS_DURATION_HIST_BIN = linspace(-5, 0, SESS_BIN);
SESS_INTERVAL_HIST_BIN = linspace(-2, 2, SESS_BIN);

ACTIVE_PADDING = 1;
ACTIVE_WINDOW = 2 * ACTIVE_PADDING + 1;

%
is = find(object.category_id == find(strcmp(CATEGORY, 'course')));
[~, order] = sort(object.course_id(is));
object_course_index = is(order);
object_course_module_num = hist(object.course_id, 1:length(COURSE_NAME))';

for file = {'train', 'test'}
    eval(['lg = lg_', file{1}, ';']);
    eval(['enrollment = enrollment_', file{1}, ';']);
    eval(['sample = sample_', file{1}, ';']);

    %% Meta
    % lg
    lg_module_exist = (lg.module_id ~= 0);
    lg_module_exist_index = MODULE_INDEX_LOOKUP(lg.module_id(lg_module_exist));

    lg_source_one_hot = bsxfun(@eq, lg.source, 1:length(SOURCE));
    lg_event_one_hot = bsxfun(@eq, lg.event, 1:length(EVENT));
    lg_activity_one_hot = lg_source_one_hot(:, SOURCE_ORDER) & lg_event_one_hot(:, EVENT_ORDER);
    
    lg_module_category = zeros(lg.length, 1);
    lg_module_category(lg_module_exist) = object.category_id(lg_module_exist_index);
    lg_object_category_one_hot = bsxfun(@eq, lg_module_category, 1:length(CATEGORY));
    
    lg_module_time = zeros(lg.length, 1);
    lg_module_time(lg_module_exist) = object.start_time(lg_module_exist_index);

    % enrollment
    enrollment_index_max = 100; %find(enrollment.enrollment_id >= lg.enrollment_id(end), 1);
    enrollment_index_range = [0; find(diff(lg.enrollment_id)); lg.length];
    enrollment_index_range = enrollment_index_range(1:enrollment_index_max + 1);
    enrollment_course_id = enrollment.course_id(1:enrollment_index_max);
    enrollment_course_time = object.start_time(object_course_index(enrollment_course_id));
    enrollment_user_id = enrollment.user_id(1:enrollment_index_max);
    enrollment_kernel = bsxfun(@eq, enrollment_user_id, enrollment_user_id') & ~eye(enrollment_index_max);

    % truth
    truth_dropout = truth.dropout(1:enrollment_index_max);

    %% Assignment
    % enrollment
    enrollment_id = enrollment.enrollment_id(1:enrollment_index_max);
    enrollment_time = max(lg.time(enrollment_index_range(1:end - 1) + 1) - enrollment_course_time, 0);
    enrollment_log_num = diff(enrollment_index_range);
    enrollment_log_activity_num = zeros(enrollment_index_max, length(SOURCE_ORDER));
    enrollment_log_object_category_num = zeros(enrollment_index_max, length(CATEGORY));

    enrollment_log_num_time = zeros(TIME_BIN, 1, enrollment_index_max);
    enrollment_log_activity_num_time = zeros(TIME_BIN, length(SOURCE_ORDER), enrollment_index_max);
    enrollment_log_object_category_num_time = zeros(TIME_BIN, length(CATEGORY), enrollment_index_max);

    % session
    sess_time_hist = zeros(enrollment_index_max, SESS_BIN);
    sess_duration_hist = zeros(enrollment_index_max, SESS_BIN);
    sess_interval_hist = zeros(enrollment_index_max, SESS_BIN);

    % user
    user_course_active_num = zeros(enrollment_index_max, ACTIVE_WINDOW + 1);
    user_course_active_log_num = zeros(enrollment_index_max, ACTIVE_WINDOW + 1);
    user_course_active_num_time = zeros(enrollment_index_max, ACTIVE_WINDOW * TIME_BIN);
    user_course_active_log_num_time = zeros(enrollment_index_max, ACTIVE_WINDOW * TIME_BIN);

    for i = 1:enrollment_index_max
        id = enrollment_id(i);
        sel = enrollment_index_range(i) + 1:enrollment_index_range(i + 1);
        sel_size = enrollment_index_range(i + 1) - enrollment_index_range(i);

        if(mod(i, 1) == 0)
            fprintf('On enrollment id = %d / %d\n', id, enrollment_id(enrollment_index_max));
        end
        
        %% Meta
        time = lg.time(sel);
        time_diff = diff(time);
        time_bin = max(ceil((time - enrollment_course_time(i)) / (30 / TIME_BIN)), 1);
        time_bin_kernel = double(bsxfun(@eq, time_bin, 1:TIME_BIN)');

        module_time = lg_module_time(sel);

        %% Assignment
        % enrollment
        activity = lg_activity_one_hot(sel, :);
        object_category = lg_object_category_one_hot(sel, :);

        enrollment_log_activity_num(i, :) = sum(activity);
        enrollment_log_object_category_num(i, :) = sum(object_category);
        enrollment_log_num_time(:, :, i) = sum(time_bin_kernel, 2);
        enrollment_log_activity_num_time(:, :, i) = time_bin_kernel * activity;
        enrollment_log_object_category_num_time(:, :, i) = time_bin_kernel * object_category;  

        % session
        sess_time = time - module_time;
        sess_time = max(sess_time(module_time ~= 0), 1 * SECOND);
        sess_range = find(time_diff > SESS_IDLE_THRESH);
        sess_duration = max(time([sess_range; sel_size]) - time([0; sess_range] + 1), 1 * SECOND);
        sess_interval = time_diff(sess_range);

        sess_time_hist(i, :) = hist(log10(sess_time), SESS_TIME_HIST_BIN);
        sess_duration_hist(i, :) = hist(log10(sess_duration), SESS_DURATION_HIST_BIN);
        sess_interval_hist(i, :) = hist(log10(sess_interval), SESS_INTERVAL_HIST_BIN);

        % user
        active_bin = ceil(bsxfun(@minus, time, enrollment_course_time(enrollment_kernel(i, :))') / (30 / TIME_BIN)) + ACTIVE_PADDING * TIME_BIN;
        active_hist_small = histc(active_bin, 1:TIME_BIN:ACTIVE_WINDOW * TIME_BIN + 1, 1)';
        active_hist_large = histc(active_bin, 1:ACTIVE_WINDOW * TIME_BIN, 1)';

        user_course_active_num(enrollment_kernel(i, :), :) = user_course_active_num(enrollment_kernel(i, :), :) + (active_hist_small > 0); 
        user_course_active_num_time(enrollment_kernel(i, :), :) = user_course_active_num_time(enrollment_kernel(i, :), :) + (active_hist_large > 0);
        user_course_active_log_num(enrollment_kernel(i, :), :) = user_course_active_log_num(enrollment_kernel(i, :), :) + active_hist_small;
        user_course_active_log_num_time(enrollment_kernel(i, :), :) = user_course_active_log_num_time(enrollment_kernel(i, :), :) + active_hist_large;
    end

    % enrollment
    enrollment_log_num_time = permute(enrollment_log_num_time, [3, 1, 2]);
    enrollment_log_activity_num_time = permute(enrollment_log_activity_num_time, [3, 1, 2]);
    enrollment_log_object_category_num_time = permute(enrollment_log_object_category_num_time, [3, 1, 2]);

    % user
    user_kernel = double(bsxfun(@eq, enrollment_id, enrollment_id'));

    user_id = enrollment_user_id;
    user_log_num = user_kernel * enrollment_log_num;
    user_log_activity_num = user_kernel * enrollment_log_activity_num;
    user_log_object_category_num = user_kernel * enrollment_log_object_category_num;
    user_course_num = sum(user_kernel, 2);
    user_course_active_num = user_course_active_num(:, 1:end - 1);
    user_course_active_log_num = user_course_active_log_num(:, 1:end - 1);
    user_course_drop_num = user_kernel * truth_dropout; 

    % course
    course_kernel = double(bsxfun(@eq, enrollment_course_id, 1:length(COURSE_NAME))');
    course_id_log_num = course_kernel * enrollment_log_num;
    course_id_log_activity_num = course_kernel * enrollment_log_activity_num;
    course_id_log_object_category_num = course_kernel * enrollment_log_object_category_num;
    course_id_user_num = hist(enrollment_course_id, 1:length(COURSE_NAME))';
    course_id_user_drop_num = hist(enrollment_course_id(truth_dropout), 1:length(COURSE_NAME))';

    course_id = enrollment_course_id;
    course_module_num = object_course_module_num(course_id);
    course_log_num = course_id_log_num(course_id, :);
    course_log_activity_num = course_id_log_activity_num(course_id, :);
    course_log_object_category_num = course_id_log_object_category_num(course_id, :);
    course_user_num = course_id_user_num(course_id);
    course_user_drop_num = course_id_user_drop_num(course_id);

    % finalize
    x1 = [...
        enrollment_id, ...
        enrollment_time, ...
        enrollment_log_num, ...
        enrollment_log_num ./ user_log_num, ...
        enrollment_log_num ./ course_log_num, ...
        enrollment_log_activity_num, ...
        bsxfun(@rdivide, enrollment_log_activity_num, user_log_num), ...
        bsxfun(@rdivide, enrollment_log_activity_num, course_log_num), ...
        enrollment_log_object_category_num, ...
        bsxfun(@rdivide, enrollment_log_object_category_num, user_log_num), ...
        bsxfun(@rdivide, enrollment_log_object_category_num, course_log_num), ...
        ...
        sess_time_hist, ...
        sess_duration_hist, ...
        sess_interval_hist, ...
        ...
        user_id, ...
        user_log_num, ...
        user_log_activity_num, ...
        bsxfun(@rdivide, user_log_activity_num, user_log_num), ...
        user_log_object_category_num, ...
        bsxfun(@rdivide, user_log_object_category_num, user_log_num), ...
        user_course_num, ...
        user_course_active_num, ...
        bsxfun(@rdivide, user_course_active_num, user_course_num), ...
        user_course_active_log_num, ...
        bsxfun(@rdivide, user_course_active_log_num, user_log_num), ...
        user_course_drop_num, ...
        user_course_drop_num ./ user_course_num, ...
        ...
        course_id, ...
        course_module_num, ...
        course_log_num, ...
        course_log_activity_num, ...
        bsxfun(@rdivide, course_log_activity_num, course_log_num), ...
        course_log_object_category_num, ...
        bsxfun(@rdivide, course_log_object_category_num, course_log_num), ...
        course_user_num, ...
        course_user_drop_num, ...
        course_user_drop_num ./ course_user_num, ...
    ];
    
    x2 = cat(3, ...
        enrollment_log_num_time, ...
        bsxfun(@rdivide, enrollment_log_num_time, user_log_num), ...
        bsxfun(@rdivide, enrollment_log_num_time, course_log_num), ...
        enrollment_log_activity_num_time, ...
        bsxfun(@rdivide, enrollment_log_activity_num_time, user_log_num), ...
        bsxfun(@rdivide, enrollment_log_activity_num_time, course_log_num), ...
        enrollment_log_object_category_num_time, ...
        bsxfun(@rdivide, enrollment_log_object_category_num_time, user_log_num), ...
        bsxfun(@rdivide, enrollment_log_object_category_num_time, course_log_num) ...
    );

    x3 = cat(3, ...
        user_course_active_num_time, ...
        bsxfun(@rdivide, user_course_active_num_time, user_course_num), ...
        user_course_active_log_num_time, ...
        bsxfun(@rdivide, user_course_active_num_time, user_log_num) ...
    );
    
    for i = 1:3
        eval(['x', num2str(i), '_', file{1}, '= x', num2str(i), ';']);
    end
end

y_train = truth_dropout;

save(['../../data/feat1_.mat'], 'x1_train', 'x2_train', 'x3_train', 'y_train', 'x1_test', 'x2_test', 'x3_test');
