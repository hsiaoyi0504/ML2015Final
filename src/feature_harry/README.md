# `feat0` specifications
-
I am too lazy to write this...

# `feat1` specifications
-
Features are collected in:

* `x1`(`#enrollment` × `286`): Non-time-expanded
* `x2`(`#enrollment` × `30` × `75`): Time-expanded / Enrollment
* `x3`(`#enrollment` × `90` × `4`): Time-expanded / User

Resulting in `2896` dimensions in total.

The items are specified below:

* Non-time-expanded
    * Enrollment (`77`)
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
        * `enrollment_log_{object_category}_num{norm_enrollment}`(`3` × `15`): `object_category` specified below
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

    * User (`65`)
        * `user_id`(`1`): trivial as fuck

        -
        * `user_log_num`(`1`): number of logs summed over all enrollment related to this user
        * `user_log_{activity}_num{norm_user}`(`2` × `9`): number of specified `activity` of logs summed over all enrollment related to this user, `norm_user` specified below
            * ` `: no normalization
            * `/user`: normalized by user
        * `user_log_{object_category}_num{norm_user}`(`2` × `15`): number of specified `object_category` of logs summed over all enrollment related to this user
        
        -
        * `user_course_num`(`1`): number of courses taken by this user
        * `user_course_active_num{norm_user}@period`(`ACTIVE_WINDOW` × `2` × `1`): number of other active courses before, during, and after this enrollment
        * `user_course_active_log_num{norm_user}@period`(`ACTIVE_WINDOW` × `2` × `1`)
        * `user_course_drop_num{norm_user}`(`2` × `1`): number of courses dropped by the user
    * Course (`54`)
        * `course_id`(`1`): trivial, right?
        * `course_module_num`(`1`): number of modules in this course

        -
        * `course_log_num`(`1`): number of logs summed over all enrollment related to this course
        * `course_log_{activity}_num{norm_course}`(`2` × `9`): number of specified `activity` of logs summed over all enrollment related to this course, `norm_course` specified below
            * ` `: no normalization
            * `/course`: normalized by course

        * `course_log_{object_category}_num{norm_course}`(`2` × `15`): number of specified `object_category` of logs summed over all enrollment related to this course

        -
        * `course_user_num`(`1`): number of users taking this course
        * `course_user_drop_num{norm_course}`(`2` × `1`): number of users dropping this course
        
* Time-expanded 
    * Enrollment (`75` × `TIME_BIN`)
        * `enrollment_log_num{norm_enrollment}@time`(`TIME_BIN` × `3` × `1`): `enrollment_log_num{norm_enrollment}` expanded to `TIME_BIN` uniform time bins
        * `enrollment_log_{activity}_num{norm_enrollment}@time`(`TIME_BIN` × `3` × `9`)
        * `enrollment_log_{object_category}_num{norm_enrollment}@time`(`TIME_BIN` × `3` × `15`)
    * User (`4` × `ACTIVE_WINDOW` × `TIME_BIN`)
        * `user_course_active_num{norm_user}@period@time`(`TIME_BIN` × `ACTIVE_WINDOW` × `2` × `1`)
        * `user_course_active_log_num{norm_user}@period@time`(`TIME_BIN` × `ACTIVE_WINDOW` × `2` × `1`)
        