if(~exist('object', 'var'))
    load('../../data/data.mat');
end

HOUR = 1 / 24;
MINUTE = HOUR / 60;
SECOND = MINUTE / 60;

TIME_BIN = 30;

SESS_BIN = 30;
SESS_IDLE_THRESH = 30 * MINUTE;
SESS_TIME_HIST_BIN = linspace(-2, 2.5, SESS_BIN);
SESS_DURATION_HIST_BIN = linspace(-5, -0.5, SESS_BIN);
SESS_INTERVAL_HIST_BIN = linspace(-1.5, 1.5, SESS_BIN);

ACTIVE_PADDING = 1;
ACTIVE_WINDOW = 2 * ACTIVE_PADDING + 1;

% object
is = int32(find(object.category_id == find(strcmp(CATEGORY, 'course'))));
[~, order] = sort(object.course_id(is));
object_course_index = is(order);
object_course_module_num = int32(hist(object.course_id, 1:length(COURSE_NAME))');
clear is order;
    
% lg
lg_module_exist = (lg.module_id ~= 0);
lg_module_exist_index = MODULE_INDEX_LOOKUP(lg.module_id(lg_module_exist));

lg_source_one_hot = bsxfun(@eq, lg.source, 1:length(SOURCE));
lg_event_one_hot = bsxfun(@eq, lg.event, 1:length(EVENT));
lg_activity_one_hot = lg_source_one_hot(:, SOURCE_ORDER) & lg_event_one_hot(:, EVENT_ORDER);
clear lg_source_one_hot lg_event_one_hot;
    
lg_module_category = zeros(lg.length, 1);
lg_module_exist_category = object.category_id(lg_module_exist_index);
[category, ~, lg_module_category(lg_module_exist)] = unique(lg_module_exist_category);
lg_object_category_one_hot = bsxfun(@eq, lg_module_category, 1:length(category));
clear lg_module_category lg_module_exist_category;

lg_module_time = zeros(lg.length, 1);
lg_module_time(lg_module_exist) = object.start_time(lg_module_exist_index);
clear lg_module_exist lg_module_exist_index;
    
% enrollment
enrollment_index_max = enrollment.length;
enrollment_index_range = int32([0; find(diff(lg.enrollment_id)); lg.length]);
enrollment_index_range = enrollment_index_range(1:enrollment_index_max + 1);
enrollment_course_id = enrollment.course_id(1:enrollment_index_max);
enrollment_course_time = object.start_time(object_course_index(enrollment_course_id));
enrollment_user_id = enrollment.user_id(1:enrollment_index_max);

% truth
truth_dropout = [truth.dropout; false(enrollment.length_test, 1)];
truth_dropout = int32(truth_dropout(1:enrollment_index_max)); 

%% FEATURE
% enrollment
enrollment_id = enrollment.enrollment_id(1:enrollment_index_max);
enrollment_time = int32(max(lg.time(enrollment_index_range(1:end - 1) + 1) - enrollment_course_time, 0));
enrollment_log_num = int32(diff(enrollment_index_range));
enrollment_log_activity_num = zeros(enrollment_index_max, length(SOURCE_ORDER), 'int32');
enrollment_log_object_category_num = zeros(enrollment_index_max, length(category), 'int32');

enrollment_log_num_time = zeros(TIME_BIN, 1, enrollment_index_max, 'int32');
enrollment_log_activity_num_time = zeros(TIME_BIN, length(SOURCE_ORDER), enrollment_index_max, 'int32');
enrollment_log_object_category_num_time = zeros(TIME_BIN, length(category), enrollment_index_max, 'int32');

% session
sess_time_hist = zeros(enrollment_index_max, SESS_BIN, 'int32');
sess_duration_hist = zeros(enrollment_index_max, SESS_BIN, 'int32');
sess_interval_hist = zeros(enrollment_index_max, SESS_BIN, 'int32');

% user
user_id_log_num = zeros(length(USER_NAME), 1, 'int32');
user_id_log_activity_num = zeros(length(USER_NAME), length(SOURCE_ORDER), 'int32');
user_id_log_object_category_num = zeros(length(USER_NAME), length(category), 'int32');
user_id_drop_num = zeros(length(USER_NAME), 1, 'int32');
user_id_course_num = zeros(length(USER_NAME), 1, 'int32');

user_course_active_num = zeros(enrollment_index_max, ACTIVE_WINDOW + 1, 'int32');
user_course_active_log_num = zeros(enrollment_index_max, ACTIVE_WINDOW + 1, 'int32');
user_course_active_num_time = zeros(enrollment_index_max, ACTIVE_WINDOW * TIME_BIN, 'int32');
user_course_active_log_num_time = zeros(enrollment_index_max, ACTIVE_WINDOW * TIME_BIN, 'int32');

for i = 1:enrollment_index_max
    eid = enrollment_id(i);
    uid = enrollment_user_id(i);
    sel = enrollment_index_range(i) + 1:enrollment_index_range(i + 1);
    sel_size = enrollment_index_range(i + 1) - enrollment_index_range(i);

    if(mod(i, 100) == 0)
        if(i < enrollment.length_train)
            fprintf('On enrollment id = %d / %d (train)\n', eid, enrollment_id(enrollment_index_max));
        else
            fprintf('On enrollment id = %d / %d (test)\n', eid, enrollment_id(enrollment_index_max));
        end
    end
        
    time = lg.time(sel);
    time_diff = diff(time);
    time_bin = max(ceil((time - enrollment_course_time(i)) / (30 / TIME_BIN)), 1);
    time_bin_kernel = double(bsxfun(@eq, time_bin, 1:TIME_BIN)');

    module_time = lg_module_time(sel);
    
    % enrollment
    activity = lg_activity_one_hot(sel, :);
    object_category = lg_object_category_one_hot(sel, :);

    enrollment_log_activity_num(i, :) = sum(activity, 1);
    enrollment_log_object_category_num(i, :) = sum(object_category, 1);
    enrollment_log_num_time(:, :, i) = sum(time_bin_kernel, 2);
    enrollment_log_activity_num_time(:, :, i) = int32(time_bin_kernel * activity);
    enrollment_log_object_category_num_time(:, :, i) = int32(time_bin_kernel * object_category);
    
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
    user_same = (uid == enrollment_user_id);
    user_same(i) = false;
    active_bin = ceil(bsxfun(@minus, time, enrollment_course_time(user_same)') / (30 / TIME_BIN)) + ACTIVE_PADDING * TIME_BIN;
    active_hist_small = int32(histc(active_bin, 1:TIME_BIN:ACTIVE_WINDOW * TIME_BIN + 1, 1)');
    active_hist_large = int32(histc(active_bin, 1:ACTIVE_WINDOW * TIME_BIN, 1)');
    
    user_id_log_num(uid) = user_id_log_num(uid) + enrollment_log_num(i);
    user_id_log_activity_num(uid, :) = user_id_log_activity_num(uid, :) + enrollment_log_activity_num(i, :);
    user_id_log_object_category_num(uid, :) = user_id_log_object_category_num(uid, :) + enrollment_log_object_category_num(i, :);
    user_id_course_num(uid) = user_id_course_num(uid) + 1;
    user_id_drop_num(uid) = user_id_drop_num(uid) + truth_dropout(i);

    user_course_active_num(user_same, :) = user_course_active_num(user_same, :) + int32(active_hist_small > 0); 
    user_course_active_num_time(user_same, :) = user_course_active_num_time(user_same, :) + int32(active_hist_large > 0);
    user_course_active_log_num(user_same, :) = user_course_active_log_num(user_same, :) + active_hist_small;
    user_course_active_log_num_time(user_same, :) = user_course_active_log_num_time(user_same, :) + active_hist_large;
end

clear i eid uid sel sel_size;
clear time time_diff time_bin time_bin_kernel module_time;
clear activity object_category;
clear sess_time sess_range sess_duration sess_interval;
clear user_same active_bin active_hist_small active_hist_large;

% enrollment
enrollment_log_num_time = permute(enrollment_log_num_time, [3, 1, 2]);
enrollment_log_activity_num_time = permute(enrollment_log_activity_num_time, [3, 1, 2]);
enrollment_log_object_category_num_time = permute(enrollment_log_object_category_num_time, [3, 1, 2]);

% user
user_id = enrollment_user_id;
user_log_num = user_id_log_num(user_id); 
user_log_activity_num = user_id_log_activity_num(user_id, :);
user_log_object_category_num = user_id_log_object_category_num(user_id, :);
user_course_num = user_id_course_num(user_id); 
user_course_active_num = user_course_active_num(:, 1:end - 1);
user_course_active_log_num = user_course_active_log_num(:, 1:end - 1);
user_course_drop_num = user_id_drop_num(user_id) - truth_dropout;
clear user_id_*;
    
% course
course_kernel = double(bsxfun(@eq, enrollment_course_id, 1:length(COURSE_NAME))');
course_id_log_num = int32(course_kernel * double(enrollment_log_num));
course_id_log_activity_num = int32(course_kernel * double(enrollment_log_activity_num));
course_id_log_object_category_num = int32(course_kernel * double(enrollment_log_object_category_num));
course_id_user_num = int32(hist(enrollment_course_id, 1:length(COURSE_NAME))');
course_id_user_drop_num = int32(course_kernel * double(truth_dropout)); 

course_id = enrollment_course_id;
course_module_num = object_course_module_num(course_id);
course_log_num = course_id_log_num(course_id, :);
course_log_activity_num = course_id_log_activity_num(course_id, :);
course_log_object_category_num = course_id_log_object_category_num(course_id, :);
course_user_num = course_id_user_num(course_id);
course_user_drop_num = course_id_user_drop_num(course_id);
clear course_kernel course_id_*;

clear object_course_index object_course_module_num;
clear lg_activity_one_hot lg_object_category_one_hot lg_module_time category;
clear enrollment_index_max enrollment_index_range enrollment_course_id enrollment_course_time enrollment_user_id;

clear COURSE_NAME COURSE_INDEX_LOOKUP MODULE_NAME MODULE_INDEX_LOOKUP CATEGORY USER_NAME SOURCE EVENT SOURCE_ORDER EVENT_ORDER;
clear HOUR MINUTE SECOND TIME_BIN SESS_BIN SESS_IDLE_THRESH SESS_TIME_HIST_BIN SESS_DURATION_HIST_BIN SESS_INTERVAL_HIST_BIN ACTIVE_PADDING ACTIVE_WINDOW;

normalize = @(x, y) bsxfun(@rdivide, single(x), single(y) + 1e-9);
feat1.len_train = enrollment.length_train;
feat1.len_test = enrollment.length_test;
feat1.len = feat1.len_train + feat1.len_test;
clear object lg enrollment truth;

% Finalize
feat1.x1_int = [...
    enrollment_id, ... % 1
    enrollment_time, ... % 1
    enrollment_log_num, ... % 1
    enrollment_log_activity_num, ... % 9
    enrollment_log_object_category_num, ... % 5
    ... % 17
    sess_time_hist, ... % 30
    sess_duration_hist, ... % 30
    sess_interval_hist, ... % 30
    ... % 107
    user_id, ... % 1
    user_log_num, ... % 1
    user_log_activity_num, ... % 9
    user_log_object_category_num, ... % 5
    user_course_num, ... % 1
    user_course_active_num, ... % 3
    user_course_active_log_num, ... % 3
    user_course_drop_num, ... % 1
    ... % 131
    course_id, ... % 1
    course_module_num, ... % 1
    course_log_num, ... % 1
    course_log_activity_num, ... % 9
    course_log_object_category_num, ... % 5
    course_user_num, ... % 1
    course_user_drop_num ... % 1
]; % 150
feat1.x1_float = [
    normalize(enrollment_log_num, user_log_num), ... % 1
    normalize(enrollment_log_num, course_log_num), ... % 1
    normalize(enrollment_log_activity_num, user_log_num), ... % 9
    normalize(enrollment_log_activity_num, course_log_num), ... % 9
    normalize(enrollment_log_object_category_num, user_log_num), ... % 5
    normalize(enrollment_log_object_category_num, course_log_num), ... % 5
    ... % 30
    normalize(user_log_activity_num, user_log_num), ... % 9
    normalize(user_log_object_category_num, user_log_num), ... % 5
    normalize(user_course_active_num, user_course_num), ... % 3
    normalize(user_course_active_log_num, user_log_num), ... % 3
    normalize(user_course_drop_num, user_course_num) ... % 1
    ... % 51
    normalize(course_log_activity_num, course_log_num), ... % 9
    normalize(course_log_object_category_num, course_log_num), ... % 5
    normalize(course_user_drop_num, course_user_num) ... % 1
]; % 66
clear enrollment_time enrollment_log_num enrollment_log_activity_num enrollment_log_object_category_num;
clear sess_time_hist sess_duration_hist sess_interval_hist;
clear user_id user_log_activity_num user_log_object_category_num user_course_active_num user_course_active_log_num user_course_drop_num;
clear course_id course_module_num course_log_activity_num course_log_object_category_num course_user_num course_user_drop_num;

feat1.x2_int = cat(3, ...
    enrollment_log_num_time, ...
    enrollment_log_activity_num_time, ...
    enrollment_log_object_category_num_time ...
);
feat1.x2_float = cat(3, ...
    normalize(enrollment_log_num_time, user_log_num), ...
    normalize(enrollment_log_num_time, course_log_num), ...
    normalize(enrollment_log_activity_num_time, user_log_num), ...
    normalize(enrollment_log_activity_num_time, course_log_num), ...
    normalize(enrollment_log_object_category_num_time, user_log_num), ...
    normalize(enrollment_log_object_category_num_time, course_log_num) ...
);
clear enrollment_log_num_time enrollment_log_activity_num_time enrollment_log_object_category_num_time course_log_num course_log_num_rel;

feat1.x3_int = cat(3, ...
    user_course_active_num_time, ...
    user_course_active_log_num_time ...
);
feat1.x3_float = cat(3, ...
    normalize(user_course_active_num_time, user_course_num), ...
    normalize(user_course_active_num_time, user_log_num) ...
);
clear user_course_active_num_time user_course_active_log_num_time user_log_num;

feat1.eid = enrollment_id;
feat1.w = 1 ./ single(user_course_num);
feat1.y = truth_dropout;
clear enrollment_id user_course_num truth_dropout;

save(['../../data/feat1.mat'], '-struct', 'feat1'); 
