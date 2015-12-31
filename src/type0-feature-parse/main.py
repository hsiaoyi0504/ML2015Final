import os
import sys
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
enrollment_data = {}
object_data = {}
course_data = {}
user_data = {}
# list for csv output
output_csv_header= ['ID', 'user_log_num', 'course_log_num', 'take_course_num',
  'take_user_num', 'log_num']
event_header = []
category_header = ['chapter_count', 'sequential_count', 'video_count']

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
    "start_time": None,
    # below are the save data
    "log_num": 0,
    "take_course_num": 0,
    "take_user_num": 0,
    "log_num": 0,
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
    "start_time": None,
    # below are the save data
    "log_num": 0,
    "take_course_num": 0,
    "take_user_num": 0,
    "log_num": 0,
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

enroll_fp.close()

# load object_data for log mapping
object_fp = open(data_path+object_file_name,"r")
csv_reader = csv.reader(object_fp)
csv_reader.next()   #skip header
for row in csv_reader:
  object_data[row[1]] = {
    "category": row[2],
    "start_time": row[4]
  }
object_fp.close()

#handle log data and save to enrollment_data
print "handle log data..."
log_fp = open(data_path+train_log_file_name,"r")
csv_reader = csv.reader(log_fp)
csv_reader.next() #skip header
for row in csv_reader:
  enroll = enrollment_data[row[0]]
  time = row[1]
  source =row[2]
  event = row[3]
  obj = {}
  if row[4] in object_data:
    obj = object_data[row[4]]
  #record data
  if enroll["start_time"] is None:
    enroll["start_time"] = time
  enroll["log_num"] += 1
  course_data[enroll["course_id"]]["log_cnt"] += 1
  user_data[enroll["user_id"]]["log_cnt"] += 1
  if source+"_"+event in enroll:
    enroll[source+"_"+event] += 1
  else:
    enroll[source+"_"+event] = 1
  if source+"_"+event not in event_header:
    event_header.append(source+"_"+event)
  if 'category' in obj:
    if obj["category"]+"_count" in enroll:
      enroll[obj["category"]+"_count"] += 1
    else:
      enroll[obj["category"]+"_count"] = 1

log_fp.close()
log_fp = open(data_path+test_log_file_name,"r")
csv_reader = csv.reader(log_fp)
csv_reader.next() #skip header
for row in csv_reader:
  enroll = enrollment_data[row[0]]
  time = row[1]
  source =row[2]
  event = row[3]
  obj = {}
  if row[4] in object_data:
    obj = object_data[row[4]]
  #record data
  if enroll["start_time"] is None:
    enroll["start_time"] = time
  enroll["log_num"] += 1
  course_data[enroll["course_id"]]["log_cnt"] += 1
  user_data[enroll["user_id"]]["log_cnt"] += 1
  if source+"_"+event in enroll:
    enroll[source+"_"+event] += 1
  else:
    enroll[source+"_"+event] = 1
  if source+"_"+event not in event_header:
    event_header.append(source+"_"+event)
  if 'category' in obj:
    if obj["category"]+"_count" in enroll:
      enroll[obj["category"]+"_count"] += 1
    else:
      enroll[obj["category"]+"_count"] = 1

log_fp.close()
#record course count and user count
for ind in enrollment_data:
  enroll = enrollment_data[ind]
  cid = enroll["course_id"]
  uid = enroll["user_id"]
  enroll["user_log_num"] = user_data[uid]["log_cnt"]
  enroll["course_log_num"] = course_data[cid]["log_cnt"]
  enroll["take_user_num"] = course_data[cid]["user_cnt"]
  enroll["take_course_num"] = user_data[uid]["course_cnt"]

print "Complete calculating and output..."

# save to train sample csv
f = open("sample_train.csv", "w")
w = csv.writer(f)
all_header = output_csv_header+event_header+category_header
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
all_header = output_csv_header+event_header+category_header
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
