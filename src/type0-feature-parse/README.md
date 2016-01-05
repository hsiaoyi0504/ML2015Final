# Featuring Parsing Spec

total 128 features

### 按照順序排列，week1 代表開始時間至七天後，week2 代表七天後時間至十四天後，week3 代表十四天後時間至二十一天後， week4 為二十一天之後的時間
* `ID`: 不解釋
* `total_enroll_time`: enrollment總時間(單位秒)，最後一次操作時間減掉開始時間。
* `enrollment_log_count`: enrollment 總 log 數。
* `user_active_rate_on_course`: 此 User 在修這門課活躍程度的比例，算法： `enrollment_log_count / course_log_count`。
* `course_active_rate_for_user`: 此 User 在所有他修過的課中，在這門課活躍程度的比例，算法：`enrollment_log_count / user_log_count`
* `browser_count`: enrollment 中 event_source 是 broser 的總數。
* `server_count`: enrollment 中 event_source 是 server 的總數。
* `event_problem_count`: enrollment 中 event 是 problem 的總數。
* `event_video_count`: enrollment 中 event 是 video 的總數。
* `event_access_count`: enrollment 中 event 是 access 的總數。
* `event_wiki_count`: enrollment 中 event 是 wiki 的總數。
* `event_discussion_count`: enrollment 中 event 是 discussion 的總數。
* `event_nagivate_count`: enrollment 中 event 是 nagivate 的總數。
* `event_page_close_count`: enrollment 中 event 是 page_close 的總數。
* `object_vertical_count`: enrollment 使用 vertical object 的總數。
* `object_problem_count`: enrollment 使用 problem object 的總數。
* `object_video_count`: enrollment 使用 video object 的總數。
* `object_sequential_count`: enrollment 使用 sequential object 的總數。
* `object_discussion_count`: enrollment 使用 discussion object 的總數。
* `object_html_count`: enrollment 使用 html object 的總數。
* `object_chapter_count`: enrollment 使用 chapter object 的總數。
* `object_about_count`: enrollment 使用 about object 的總數。
* `object_outlink_count`: enrollment 使用 outlink object 的總數。
* `object_course_info_count`: enrollment 使用 course info object 的總數。
* `object_static_tab_count`: enrollment 使用 static tab object 的總數。
* `object_course_count`: enrollment 使用 course object 的總數。
* `object_combinedopenended_count`: enrollment 使用 combinedopenended object 的總數。
* `object_peergrading_count`: enrollment 使用 peergrading object 的總數。
* `object_dictation_count`: enrollment 使用 dictation object 的總數。
* `weekX_enrollment_log_count`: week X 的 enrollment 總 log 數。
* `weekX_browser_count`: week X 的 enrollment 中 event_source 是 broser 的總數。
* `weekX_server_count`: week X 的 enrollment 中 event_source 是 server 的總數。
* `weekX_event_problem_count`: week X 的 enrollment 中 event 是 problem 的總數。
* `weekX_event_video_count`: week X 的 enrollment 中 event 是 video 的總數。
* `weekX_event_access_count`: week X 的 enrollment 中 event 是 access 的總數。
* `weekX_event_wiki_count`: week X 的 enrollment 中 event 是 wiki 的總數。
* `weekX_event_discussion_count`: week X 的 enrollment 中 event 是 discussion 的總數。
* `weekX_event_nagivate_count`: week X 的 enrollment 中 event 是 nagivate 的總數。
* `weekX_event_page_close_count`: week X 的 enrollment 中 event 是 page_close 的總數。
* `weekX_object_vertical_count`: week X 的 enrollment 使用 vertical object 的總數。
* `weekX_object_problem_count`: week X 的 enrollment 使用 problem object 的總數。
* `weekX_object_video_count`: week X 的 enrollment 使用 video object 的總數。
* `weekX_object_sequential_count`: week X 的 enrollment 使用 sequential object 的總數。
* `weekX_object_discussion_count`: week X 的 enrollment 使用 discussion object 的總數。
* `weekX_object_html_count`: week X 的 enrollment 使用 html object 的總數。
* `weekX_object_chapter_count`: week X 的 enrollment 使用 chapter object 的總數。
* `weekX_object_about_count`: week X 的 enrollment 使用 about object 的總數。
* `weekX_object_outlink_count`: week X 的 enrollment 使用 outlink object 的總數。
* `weekX_object_course_info_count`: week X 的 enrollment 使用 course info object 的總數。
* `weekX_object_static_tab_count`: week X 的 enrollment 使用 static tab object 的總數。
* `weekX_object_course_count`: week X 的 enrollment 使用 course object 的總數。
* `weekX_object_combinedopenended_count`: week X 的 enrollment 使用 combinedopenended object 的總數。
* `weekX_object_peergrading_count`: week X 的 enrollment 使用 peergrading object 的總數。
* `weekX_object_dictation_count`: week X 的 enrollment 使用 dictation object 的總數。
