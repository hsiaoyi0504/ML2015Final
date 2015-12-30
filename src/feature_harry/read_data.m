%dataDir = '/Users/harry/Desktop/ML_final_project/';
dataDir = input('Raw data directory? ');

% enrollment_id,username,course_id
for file = {'train', 'test'}
    fenrollment = fopen([dataDir, 'enrollment_', file{1}, '.csv']);
    enrollment = textscan(fenrollment, '%d%s%s', 'HeaderLines', 1, 'Delimiter', ',');
    fclose(fenrollment);
    eval(['enrollment_', file{1}, '= enrollment;']);
end

% course_id,module_id,category,children,start
CATEGORY = {'about', 'chapter', 'combinedopenended', 'course', 'course_info', 'dictation', 'discussion', 'html', 'outlink', 'peergrading', 'problem', 'sequential', 'static_tab', 'vertical', 'video'};

fobject = fopen([dataDir, 'object.csv']);
object = textscan(fobject, '%s%s%s%s%{uuuu-MM-dd''T''HH:mm:ss}D', 'HeaderLines', 1, 'Delimiter', ',');
[~, object{3}] = ismember(object{3}, CATEGORY);
object{5} = datenum(object{5});
fclose(fobject);

% enrollment_id,time,source,event,objectect
SOURCE = {'browser', 'server'};
EVENT = {'access', 'discussion', 'navigate', 'page_close', 'problem', 'video', 'wiki'};
SOURCE_ORDER = [1, 1, 1, 1, 2, 2, 2, 2, 2];
EVENT_ORDER = [1, 4, 5, 6, 1, 2, 3, 5, 7];

for file = {'train', 'test'}
    flog = fopen([dataDir, 'log_', file{1}, '.csv']);
    log = textscan(flog, '%d%{uuuu-MM-dd''T''HH:mm:ss}D%s%s%s', 'HeaderLines', 1, 'Delimiter', ',');
    log{2} = datenum(log{2});
    [~, log{3}] = ismember(log{3}, SOURCE);
    [~, log{4}] = ismember(log{4}, EVENT);
    fclose(flog);
    eval(['log_', file{1}, '= log;']);
end

% enrollment_id,dropout
fdrop = fopen([dataDir, 'truth_train.csv']);
drop = textscan(fdrop, '%d%d', 'Delimiter', ',');
drop{2} = double(drop{2});
fclose(fdrop);

%
sample_train = csvread([dataDir, 'sample_train_x.csv'], 1, 0);
sample_test = csvread([dataDir, 'sample_test_x.csv'], 1, 0);
