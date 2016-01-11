# `feat0` specifications
-
I am too lazy to write this...

# `feat1` specifications
-

* `len`: `120542`
* `len_train`: `96434`
* `len_train_train`: `91613`, suggested training size
* `len_train_val`: `4821`, suggested validation size
* `len_test`: `24108`
* `permutation`: `120542`, suggested permutation
    * `permutation[0:len_train_train]`: suggested training set
    * `permutation[len_train_train:len_train]`: suggested validation set
    * `permutation[len_train:len]`: testing set
* `w`: `120542`, weight in `track 2` or the inverse of number of courses taken by the user
* `x1_int`(`120542` × `150`): Non-time-expanded
* `x1_float`(`120542` × `66`): Non-time-expanded
* `x2_int`(`120542` × `30` × `15`): Time-expanded / Enrollment
* `x2_float`(`120542` × `30` × `30`): Time-expanded / Enrollment
* `x3_int`(`120542` × `90` × `2`): Time-expanded / User
* `x3_float`(`120542` × `90` × `2`): Time-expanded / User
* `y`(`120542` × `1`): Dropout

Resulting in `1926` dimensions in total.

```Matlab
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

feat1.x3_int = cat(3, ...
    user_course_active_num_time, ...
    user_course_active_log_num_time ...
);
feat1.x3_float = cat(3, ...
    normalize(user_course_active_num_time, user_course_num), ...
    normalize(user_course_active_num_time, user_log_num) ...
);

feat1.y = truth_dropout;
```

The items are specified below:

* Non-time-expanded
    * Enrollment 
        * `enrollment_id`(`1`): trivial as shit
        * `enrollment_time`(`1`): the time of first log since the course released

        -
        * `enrollment_log_num{norm_enrollment}`(`3`): number of logs in this enrollment, `norm_enrollment` specified below
            * ` `: no normalization
            * `/user`: normalized by user
            * `/course`: normalized by course
        * `enrollment_log_{activity}_num{norm_enrollment}`(`3` × `9`): `activity` specified below
            * `browser_access`
            * `browser_page_close`
            * `browser_problem`
            * `browser_video`
            * `server_access`
            * `server_discussion`
            * `server_nagivate`
            * `server_problem`
            * `server_wiki`
        * `enrollment_log_{object_category}_num{norm_enrollment}`(`3` × `5`): `object_category` specified below
            * `about`
            * `chapter`
            * `combinedopenended`
            * `course`
            * `course_info`
            * `dictation`
            * `discussion`
            * `html`
            * `outlink`
            * `peergrading`
            * `problem`
            * `sequential`
            * `static_tab`
            * `vertical`
            * `video`
    * Session (`3` × `SESS_BIN`)
        * `sess_time_hist`(`SESS_BIN`): the histogram of the time for an object to be interacted since its release
        * `sess_duration_hist`(`SESS_BIN`): the histogram of the duration of log sessions from this enrollment
        * `sess_interval_hist`(`SESS_BIN`): the histogram of the interval of log sessions from this enrollment

    * User 
        * `user_id`(`1`): trivial as fuck

        -
        * `user_log_num`(`1`): number of logs summed over all enrollment related to this user
        * `user_log_{activity}_num{norm_user}`(`2` × `9`): number of specified `activity` of logs summed over all enrollment related to this user, `norm_user` specified below
            * ` `: no normalization
            * `/user`: normalized by user
        * `user_log_{object_category}_num{norm_user}`(`2` × `5`): number of specified `object_category` of logs summed over all enrollment related to this user
        
        -
        * `user_course_num`(`1`): number of courses taken by this user
        * `user_course_active_num{norm_user}@period`(`ACTIVE_WINDOW` × `2` × `1`): number of other active courses before, during, and after this enrollment
        * `user_course_active_log_num{norm_user}@period`(`ACTIVE_WINDOW` × `2` × `1`)
        * `user_course_drop_num{norm_user}`(`2` × `1`): number of OTHER courses dropped by the user
    * Course 
        * `course_id`(`1`): trivial, right?
        * `course_module_num`(`1`): number of modules in this course

        -
        * `course_log_num`(`1`): number of logs summed over all enrollment related to this course
        * `course_log_{activity}_num{norm_course}`(`2` × `9`): number of specified `activity` of logs summed over all enrollment related to this course, `norm_course` specified below
            * ` `: no normalization
            * `/course`: normalized by course

        * `course_log_{object_category}_num{norm_course}`(`2` × `5`): number of specified `object_category` of logs summed over all enrollment related to this course

        -
        * `course_user_num`(`1`): number of users taking this course
        * `course_user_drop_num{norm_course}`(`2` × `1`): number of users dropping this course
        
* Time-expanded 
    * Enrollment (`75` × `TIME_BIN`)
        * `enrollment_log_num{norm_enrollment}@time`(`TIME_BIN` × `3` × `1`): `enrollment_log_num{norm_enrollment}` expanded to `TIME_BIN` uniform time bins
        * `enrollment_log_{activity}_num{norm_enrollment}@time`(`TIME_BIN` × `3` × `9`)
        * `enrollment_log_{object_category}_num{norm_enrollment}@time`(`TIME_BIN` × `3` × `5`)
    * User (`4` × `ACTIVE_WINDOW` × `TIME_BIN`)
        * `user_course_active_num{norm_user}@period@time`(`TIME_BIN` × `ACTIVE_WINDOW` × `2` × `1`)
        * `user_course_active_log_num{norm_user}@period@time`(`TIME_BIN` × `ACTIVE_WINDOW` × `2` × `1`)
        