dataDir = '~/Desktop/ML_final_project/';

% object.course_id, object.module_id, object.category_id, object.start_time 
fid = fopen([dataDir, 'object.csv']);
object_cell = textscan(fid, '%s%s%s%s%{uuuu-MM-dd''T''HH:mm:ss}D', 'HeaderLines', 1, 'Delimiter', ',');

[COURSE_NAME, COURSE_LOOKUP, object.course_id] = unique(object_cell{1}); object.course_id = int8(object.course_id);
[MODULE_NAME, MODULE_LOOKUP, object.module_id] = unique(object_cell{2}); object.module_id = int16(object.module_id);
CATEGORY = {'about', 'chapter', 'combinedopenended', 'course', 'course_info', 'dictation', 'discussion', 'html', 'outlink', 'peergrading', 'problem', 'sequential', 'static_tab', 'vertical', 'video'};
[~, object.category_id] = ismember(object_cell{3}, CATEGORY); object.category_id = int8(object.category_id);
object.start_time = datenum(object_cell{5});

fclose(fid);
clear object_cell;

% enrollment.enrollment_id, enrollment.user_id, enrollment.course_id
for file = {'train', 'test'}
    fid = fopen([dataDir, 'enrollment_', file{1}, '.csv']);
    enrollment_cell = textscan(fid, '%d%s%s', 'HeaderLines', 1, 'Delimiter', ',');

    enrollment.enrollment_id = enrollment_cell{1};
    enrollment.user_id = enrollment_cell{2};
    [~, enrollment.course_id] = ismember(enrollment_cell{3}, COURSE_NAME); enrollment.course_id = int8(enrollment.course_id);

    fclose(fid);
    eval(['enrollment_', file{1}, '= enrollment;']);
end

USER_NAME = unique([enrollment_train.user_id; enrollment_test.user_id]);
[~, enrollment_train.user_id] = ismember(enrollment_train.user_id, USER_NAME); enrollment_train.user_id = int32(enrollment_train.user_id);
[~, enrollment_test.user_id] = ismember(enrollment_test.user_id, USER_NAME); enrollment_test.user_id = int32(enrollment_test.user_id);

clear enrollment_cell enrollment;

% lg.enrollment_id, lg.time, lg.source, lg.event, lg.module_id
SOURCE = {'browser', 'server'};
EVENT = {'access', 'discussion', 'nagivate', 'page_close', 'problem', 'video', 'wiki'};
SOURCE_ORDER = [1, 1, 1, 1, 2, 2, 2, 2, 2];
EVENT_ORDER = [1, 4, 5, 6, 1, 2, 3, 5, 7];

for file = {'train', 'test'}
    fid = fopen([dataDir, 'log_', file{1}, '.csv']);
    log_cell = textscan(fid, '%d%{uuuu-MM-dd''T''HH:mm:ss}D%s%s%s', 'HeaderLines', 1, 'Delimiter', ',');

    lg.enrollment_id = log_cell{1};
    lg.time = datenum(log_cell{2});
    [~, lg.source] = ismember(log_cell{3}, SOURCE); lg.source = int8(lg.source);
    [~, lg.event] = ismember(log_cell{4}, EVENT); lg.event = int8(lg.event);
    [~, lg.module_id] = ismember(log_cell{5}, MODULE_NAME); lg.module_id = int16(lg.module_id);

    fclose(fid);
    eval(['lg_', file{1}, '= lg;']);
end

clear log_cell lg;

% truth.enrollment_id, truth.dropout
fid = fopen([dataDir, 'truth_train.csv']);

truth_cell = textscan(fid, '%d%d', 'Delimiter', ',');
truth.enrollment_id = truth_cell{1};
truth.dropout = logical(truth_cell{2});

fclose(fid);
clear truth_cell;

%
sample_train = int32(csvread([dataDir, 'sample_train_x.csv'], 1, 0));
sample_test = int32(csvread([dataDir, 'sample_test_x.csv'], 1, 0));

clear file fid;
