# `Feat1` specifications
---

* Time-irrelevant attributes
  * Enrollment-related
    * `enrollment_id`
    * `enrollment_log_num`: log number in this enrollment
    * `enrollment_log_num_to_user_log_num`: `enrollment_log_num` / `user_log_num`
    * `enrollment_log_num_to_course_log_num`: `enrollment_log_num` / `user_course_num`
  * User-related
    * `user_id`
    * `user_log_num`: log number summed over all enrollment related to this user
    * `user_course_num`: number of courses taken by this user
  * Course-related
    * `course_id`
    * `course_log_num`: log number summed over all enrollment related to this course
    * `course_user_num`: number of users taking this course


