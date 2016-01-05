import os
import dateutil.parser
import time
import datetime
import csv

if not os.path.isdir("../../data/ML_final_project"):
  print("can't find data directory, please extract 'ML_final_release.tar.gz' ><")
  exit(1)


data_path = "../../data/ML_final_project/"
train_log_file_name = "log_train.csv"
test_log_file_name = "log_test.csv"
train_enrollment_file_name = "enrollment_train.csv"
test_enrollment_file_name = "enrollment_test.csv"
object_file_name = "object.csv"
# handle four kind of data
event_list = ["problem","video","access","wiki","discussion","nagivate","page_close"]
module_list = ["vertical","problem","video","sequential","discussion",
    "html_num","chapter","about","outlink","course_info","static_tab","course",
    "combinedopenended","peergrading","dictation"]
enrollment_data = {}
object_data = {}
course_data = {}
user_data = {}

# load enrollment_id
print "load init data..."
enroll_fp = open(data_path+train_enrollment_file_name,"r")
csv_reader = csv.reader(enroll_fp)
csv_reader.next()   #skip header
train_index = []
for row in csv_reader:
  # enrollment data initialize
  train_index.append(int(row[0]))
  enrollment_data[row[0]] = {
    "user_id": row[1],
    "course_id": row[2],
    "start_time": 0,
    "end_time": 0,
    # below are the save data
    "log_num": 0,
    "server_count": 0,
    "browser_count": 0,
    "week_data":[{},{},{},{}]
  }
  #record course_cnt
  if row[2] in course_data:
    course_data[row[2]]['user_cnt'] += 1
  else:
    course_data[row[2]] = {
      "user_cnt": 1,
      "log_cnt": 0
    }
  #record user cnt
  if row[1] in user_data:
    user_data[row[1]]['course_cnt'] += 1
  else:
    user_data[row[1]] = {
      "course_cnt": 1,
      "log_cnt": 0
    }
  for name in module_list:
    enrollment_data[row[0]]["object_"+name+"_count"] = 0
  for ev in event_list:
    enrollment_data[row[0]]["event_"+ev+"_count"] = 0
  for i in xrange(0,4):
    enrollment_data[row[0]]["week_data"][i]["log_count"] = 0
    enrollment_data[row[0]]["week_data"][i]["browser_count"] = 0
    enrollment_data[row[0]]["week_data"][i]["server_count"] = 0
    for name in module_list:
      enrollment_data[row[0]]["week_data"][i][name] = 0
    for ev in event_list:
      enrollment_data[row[0]]["week_data"][i]["event_"+ev] = 0

enroll_fp.close()

enroll_fp = open(data_path+test_enrollment_file_name,"r")
csv_reader = csv.reader(enroll_fp)
csv_reader.next()   #skip header
test_index = []
for row in csv_reader:
  # enrollment data initialize
  test_index.append(row[0])
  enrollment_data[row[0]] = {
    "user_id": row[1],
    "course_id": row[2],
    "start_time": 0,
    "end_time": 0,
    # below are the save data
    "log_num": 0,
    "server_count": 0,
    "browser_count": 0,
    "week_data":[{},{},{},{}]
  }
  #record course_cnt
  if row[2] in course_data:
    course_data[row[2]]['user_cnt'] += 1
  else:
    course_data[row[2]] = {
      "user_cnt": 1,
      "log_cnt": 0
    }
  #record user cnt
  if row[1] in user_data:
    user_data[row[1]]['course_cnt'] += 1
  else:
    user_data[row[1]] = {
      "course_cnt": 1,
      "log_cnt": 0
    }
  for name in module_list:
    enrollment_data[row[0]]["object_"+name+"_count"] = 0
  for ev in event_list:
    enrollment_data[row[0]]["event_"+ev+"_count"] = 0
  for i in xrange(0,4):
    enrollment_data[row[0]]["week_data"][i]["log_count"] = 0
    enrollment_data[row[0]]["week_data"][i]["browser_count"] = 0
    enrollment_data[row[0]]["week_data"][i]["server_count"] = 0
    for name in module_list:
      enrollment_data[row[0]]["week_data"][i][name] = 0
    for ev in event_list:
      enrollment_data[row[0]]["week_data"][i]["event_"+ev] = 0

enroll_fp.close()

# load object_data for log mapping
object_fp = open(data_path+object_file_name,"r")
csv_reader = csv.reader(object_fp)
csv_reader.next()   #skip header
for row in csv_reader:
  object_data[row[1]] = row[2]
object_fp.close()

#handle log data and save to enrollment_data
print "handle log data..."

for file_name in [train_log_file_name,test_log_file_name]:
  log_fp = open(data_path+file_name,"r")
  csv_reader = csv.reader(log_fp)
  csv_reader.next() #skip header
  for row in csv_reader:
    enroll = enrollment_data[row[0]]
    enroll["log_num"] += 1
    cid = enroll["course_id"]
    uid = enroll["user_id"]
    course_data[cid]["log_cnt"] += 1
    user_data[uid]["log_cnt"] += 1

    _time = time.mktime(dateutil.parser.parse(row[1]).timetuple())
    if enroll["start_time"] == 0:
      enroll["start_time"] = _time
    enroll["end_time"] = _time
    source =row[2]
    enroll[source+"_count"] += 1
    event = row[3]
    enroll["event_"+event+"_count"] += 1
    # week data handle
    time_diff = _time - enroll["start_time"]
    if time_diff <= 86400 * 7:
      week = 0
    elif time_diff <= 86400 * 14:
      week = 1
    elif time_diff <= 86400 * 21:
      week = 2
    else:
      week = 3
    enroll["week_data"][week]["log_count"] += 1
    enroll["week_data"][week][source+"_count"] += 1
    enroll["week_data"][week]["event_"+event] += 1
    obj = ""
    if row[4] in object_data:
      obj = object_data[row[4]]
    if obj:
      enroll["object_"+obj+"_count"] += 1
      enroll["week_data"][week][obj] += 1

  log_fp.close()

#record course count and user count
for ind in enrollment_data:
  enroll = enrollment_data[ind]
  cid = enroll["course_id"]
  uid = enroll["user_id"]
  enroll["total_enroll_time"] = enroll["end_time"] - enroll["start_time"]
  enroll["enrollment_log_count"] = enroll["log_num"]
  # enroll["user_active_rate_on_course"] = 0
  # enroll["course_active_rate_for_user"] = 0
  enroll["user_active_rate_on_course"] = float(enroll["log_num"]) / course_data[cid]["log_cnt"]
  enroll["course_active_rate_for_user"] = float(enroll["log_num"]) / user_data[uid]["log_cnt"]
  for i in xrange(0,4):
    enroll["week"+str(i+1)+"_enrollment_log_count"] = enroll["week_data"][i]["log_count"]
    enroll["week"+str(i+1)+"_browser_count"] = enroll["week_data"][i]["browser_count"]
    enroll["week"+str(i+1)+"_server_count"] = enroll["week_data"][i]["server_count"]
    for name in module_list:
      enroll["week"+str(i+1)+"_object_"+name+"_count"] = enroll["week_data"][i][name]
    for ev in event_list:
      enroll["week"+str(i+1)+"_event_"+name+"_count"] = enroll["week_data"][i]["event_"+ev]
  del enroll["week_data"]

print "Complete calculating and output..."

# save to train sample csv
f = open("sample_train.csv", "w")
w = csv.writer(f)
all_header = ["ID","total_enroll_time","enrollment_log_count","user_active_rate_on_course",
    "course_active_rate_for_user","browser_count","server_count"]
for ev in event_list:
  all_header.append("event_"+ev+"_count")
for name in module_list:
  all_header.append("object_"+name+"_count")
for i in xrange(1,5):
  all_header.append("week"+str(i)+"_enrollment_log_count")
  all_header.append("week"+str(i)+"_browser_count")
  all_header.append("week"+str(i)+"_server_count")
  for ev in event_list:
    all_header.append("week"+str(i)+"_event_"+ev+"_count")
  for name in module_list:
    all_header.append("week"+str(i)+"_object_"+name+"_count")

w.writerow(all_header) #write header
for ind in train_index:
  write_row = []
  enroll = enrollment_data[str(ind)]
  for head in all_header:
    if head == "ID":
      write_row.append(int(ind))
    elif head in enroll:
      write_row.append(enroll[head])
    else:
      write_row.append(0)
  w.writerow(write_row)
f.close()
#save to test.csv
f = open("sample_test.csv", "w")
w = csv.writer(f)
w.writerow(all_header) #write header
for ind in test_index:
  write_row = []
  enroll = enrollment_data[str(ind)]
  for head in all_header:
    if head == "ID":
      write_row.append(int(ind))
    elif head in enroll:
      write_row.append(enroll[head])
    else:
      write_row.append(0)
  w.writerow(write_row)
f.close()
print "Done ^^"
