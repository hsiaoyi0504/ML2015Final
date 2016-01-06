% enrollment_attr / course_attr
% enrollment_attr / user_attr

if(~exist('object', 'var'))
    load('../../data/data.mat');
end
TIME_BIN = 30;
SESSION_IDLE_THRESH = 30;
SESSION_HIST_BIN = [0:0.1:5];

object.category_is_course = find(object.category_id == find(strcmp(CATEGORY, 'course')));
[~, object.course_id_lookup] = ismember(1:length(object.category_is_course), object.course_id(object.category_is_course));

%for file = {'train', 'test'}
for file = {'train'}
    eval(['lg = lg_', file{1}, ';']);
    eval(['enrollment = enrollment_', file{1}, ';']);
    eval(['sample = sample_', file{1}, ';']);

    lg.enrollment_id_interval = [0; find(diff(lg.enrollment_id)); length(lg.enrollment_id)];
    %index_max = find(enrollment.enrollment_id >= lg.enrollment_id(end), 1);
    index_max = 100;
    
    enrollment.course_start_time = object.start_time(object.category_is_course(object.course_id_lookup(enrollment.course_id)));

    c = cell(index_max, 1);
    x = zeros(index_max, length(SESSION_HIST_BIN));
    for i = 1:index_max
        enrollment_id = enrollment.enrollment_id(i);
        lg_enrollment_id_range = lg.enrollment_id_interval(i) + 1:lg.enrollment_id_interval(i + 1);
        lg_enrollment_id_range_size = lg.enrollment_id_interval(i + 1) - lg.enrollment_id_interval(i);
        if(mod(i, 1000) == 0)
            fprintf('On enrollment id = %d / %d\n', enrollment_id, enrollment.enrollment_id(index_max));
        end

        lg_time = lg.time(lg_enrollment_id_range) * 24 * 60;
        lg_time_boundary = diff(lg_time) > SESSION_IDLE_THRESH;
        session_duration = lg_time([lg_time_boundary; true]) - lg_time([true; lg_time_boundary]);
        session_duration = max(session_duration, 1);
        session_hist = hist(log(session_duration), SESSION_HIST_BIN);
        
        c{i} = session_duration;
        x(i, :) = session_hist;
    end
end

