% object.course_id, object.module_id, object.category_id, object.start_time 
% enrollment.enrollment_id, enrollment.user_id, enrollment.course_id
% lg.enrollment_id, lg.time, lg.source, lg.event, lg.module_id
% truth.enrollment_id, truth.dropout

if(~exist('object', 'var'))
    load('../../data/data.mat');
end
SLOT = 30;

object.category_is_course = find(object.category_id == find(strcmp(CATEGORY, 'course')));
[~, object.course_id_lookup] = ismember(1:length(object.category_is_course), object.course_id(object.category_is_course));

for file = {'train', 'test'}
    eval(['lg = lg_', file{1}, ';']);
    eval(['enrollment = enrollment_', file{1}, ';']);
    eval(['sample = sample_', file{1}, ';']);

    lg.enrollment_id_interval = [0; find(diff(lg.enrollment_id)); length(lg.enrollment_id)];
    index_max = find(enrollment.enrollment_id >= lg.enrollment_id(end), 1);
    xa = zeros(index_max, SLOT, length(SOURCE_ORDER) + length(CATEGORY), 'int16');
    xb = zeros(index_max, length(SOURCE_ORDER) + length(CATEGORY), 'int32');
    
    enrollment.course_start_time = object.start_time(object.category_is_course(object.course_id_lookup(enrollment.course_id)));

    for i = 1:index_max
        enrollment_id = enrollment.enrollment_id(i);
        lg_enrollment_id_range = lg.enrollment_id_interval(i) + 1:lg.enrollment_id_interval(i + 1);
        lg_enrollment_id_range_size = lg.enrollment_id_interval(i + 1) - lg.enrollment_id_interval(i);
        if(mod(i, 1000) == 0)
            fprintf('On enrollment id = %d / %d\n', enrollment_id, enrollment.enrollment_id(index_max));
        end

        lg_time = lg.time(lg_enrollment_id_range);
        lg_time_slot = max(ceil((lg_time - enrollment.course_start_time(i)) / (30 / SLOT)), 1);
        lg_time_slot_interval = [0; find(diff(lg_time_slot)); length(lg_time_slot)];
        %{
        lg_time_slot_kernel = zeros(length(lg_time_slot), SLOT);
        for j = 1:length(lg_time_slot_interval) - 1
            lg_time_slot_kernel(lg_time_slot_interval(j) + 1:lg_time_slot_interval(j + 1), lg_time_slot(lg_time_slot_interval(j) + 1)) = 1;
        end
        %}
        lg_time_slot_kernel = sparse(1:length(lg_time_slot), lg_time_slot, 1, length(lg_time_slot), SLOT);

        %
        lg_source = lg.source(lg_enrollment_id_range);
        lg_event = lg.event(lg_enrollment_id_range);
        lg_module_id = lg.module_id(lg_enrollment_id_range);
        
        source_sequential = bsxfun(@eq, lg_source, 1:length(SOURCE));
        event_sequential = bsxfun(@eq, lg_event, 1:length(EVENT));
        activity_sequential = source_sequential(:, SOURCE_ORDER) & event_sequential(:, EVENT_ORDER);

        lg_module_exist = lg_module_id ~= 0;
        lg_module_category = zeros(lg_enrollment_id_range_size, 1);
        lg_module_category(lg_module_exist) = object.category_id(MODULE_LOOKUP(lg_module_id(lg_module_exist)));
        category_sequential = bsxfun(@eq, lg_module_category, 1:length(CATEGORY));

        %%
        activity_feature = lg_time_slot_kernel' * activity_sequential;
        category_feature = lg_time_slot_kernel' * category_sequential;
        
        %{
        activity_feature = zeros(SLOT, length(SOURCE_ORDER));
        category_feature = zeros(SLOT, length(CATEGORY));
       
        for j = 1:length(lg_time_slot_interval) - 1
            lg_time_slot_range = lg_time_slot_interval(j) + 1:lg_time_slot_interval(j + 1);
            activity_feature(lg_time_slot(lg_time_slot_interval(j) + 1), :) = sum(activity_sequential(lg_time_slot_range, :));
            category_feature(lg_time_slot(lg_time_slot_interval(j) + 1), :) = sum(category_sequential(lg_time_slot_range, :));
        end
        %}

        xa(i, :, :) = permute([activity_feature, category_feature], [3, 1, 2]);
        xb(i, :) = [sum(activity_feature, 1), sum(category_feature, 1)];
    end
    
    xb = [sample(1:index_max, 1:6), xb];

    eval(['xa_', file{1}, '= xa;']);
    eval(['xb_', file{1}, '= xb;']);
end

y_train = truth.dropout(1:index_max);

save(['../../data/feat0_', num2str(SLOT), '.mat'], 'xa_train', 'xb_train', 'y_train', 'xa_test', 'xb_test');

