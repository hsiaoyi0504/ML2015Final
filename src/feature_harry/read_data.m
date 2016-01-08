dataDir = '~/Desktop/ML_final_project/';

% object.course_id, object.module_id, object.category_id, object.start_time 
fid = fopen([dataDir, 'object.csv']);
object_cell = textscan(fid, '%s%s%s%s%{uuuu-MM-dd''T''HH:mm:ss}D', 'HeaderLines', 1, 'Delimiter', ',');

object.length = length(object_cell{1});
[COURSE_NAME, COURSE_INDEX_LOOKUP, object.course_id] = unique(object_cell{1}); 
COURSE_INDEX_LOOKUP = int16(COURSE_INDEX_LOOKUP);
object.course_id = int8(object.course_id);
[MODULE_NAME, MODULE_INDEX_LOOKUP, object.module_id] = unique(object_cell{2}); 
MODULE_INDEX_LOOKUP = int16(MODULE_INDEX_LOOKUP);
object.module_id = int16(object.module_id);
CATEGORY = {'about', 'chapter', 'combinedopenended', 'course', 'course_info', 'dictation', 'discussion', 'html', 'outlink', 'peergrading', 'problem', 'sequential', 'static_tab', 'vertical', 'video'};
[~, object.category_id] = ismember(object_cell{3}, CATEGORY); object.category_id = int8(object.category_id);
object.start_time = datenum(object_cell{5});

fclose(fid);
clear object_cell;

% enrollment
enrollment.length = 0;
for field = {'enrollment_id', 'user_id', 'course_id'}
    enrollment.(field{1}) = [];
end
for file = {'train', 'test'}
    fid = fopen([dataDir, 'enrollment_', file{1}, '.csv']);
    enrollment_cell = textscan(fid, '%d%s%s', 'HeaderLines', 1, 'Delimiter', ',');

    enrollment.(['length_', file{1}]) = length(enrollment_cell{1});
    enrollment.length = enrollment.length + length(enrollment_cell{1});
    enrollment.enrollment_id = [enrollment.enrollment_id; enrollment_cell{1}];
    enrollment.user_id = [enrollment.user_id; enrollment_cell{2}];
    [~, temp] = ismember(enrollment_cell{3}, COURSE_NAME); 
    enrollment.course_id = [enrollment.course_id; int8(temp)];
    
    fclose(fid);
end

[USER_NAME, ~, enrollment.user_id] = unique(enrollment.user_id); enrollment.user_id = int32(enrollment.user_id);

clear enrollment_cell;

% lg
SOURCE = {'browser', 'server'};
EVENT = {'access', 'discussion', 'nagivate', 'page_close', 'problem', 'video', 'wiki'};
SOURCE_ORDER = [1, 1, 1, 1, 2, 2, 2, 2, 2];
EVENT_ORDER = [1, 4, 5, 6, 1, 2, 3, 5, 7];

lg.length = 0;
for field = {'enrollment_id', 'time', 'source', 'event', 'module_id'}
    lg.(field{1}) = [];
end
for file = {'train', 'test'}
    fid = fopen([dataDir, 'log_', file{1}, '.csv']);
    log_cell = textscan(fid, '%d%{uuuu-MM-dd''T''HH:mm:ss}D%s%s%s', 'HeaderLines', 1, 'Delimiter', ',');

    lg.(['length_', file{1}]) = length(log_cell{1});
    lg.length = lg.length + length(log_cell{1});
    lg.enrollment_id = [lg.enrollment_id; log_cell{1}];
    lg.time = [lg.time; datenum(log_cell{2})];
    [~, temp] = ismember(log_cell{3}, SOURCE);
    lg.source = [lg.source; int8(temp)];
    [~, temp] = ismember(log_cell{4}, EVENT);
    lg.event = [lg.event; int8(temp)];
    [~, temp] = ismember(log_cell{5}, MODULE_NAME);
    lg.module_id = [lg.module_id; int16(temp)];

    fclose(fid);
end

clear log_cell;

% truth.enrollment_id, truth.dropout
fid = fopen([dataDir, 'truth_train.csv']);

truth_cell = textscan(fid, '%d%d', 'Delimiter', ',');
truth.length = length(truth_cell{1});
truth.enrollment_id = truth_cell{1};
truth.dropout = logical(truth_cell{2});

fclose(fid);
clear truth_cell;

clear dataDir file field fid ans temp;

save('../../data/data.mat');
