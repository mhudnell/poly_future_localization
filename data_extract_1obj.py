#
# Run from "F:\Car data\kitti\data_tracking\training\label_02"
# $ cd "F:\Car data\kitti\data_tracking\training\label_02"
# $ python "C:\Users\Max\Research\maxGAN\data_extract_1obj.py"
#

import os
from collections import deque

def getLineCounts_1obj():
  """Check that all 'past' data files have 10 lines, and 'future' data files have 11 lines."""
  total_past_count = 0
  badpast_count = 0
  for subdir, dirs, files in os.walk('F:\\Car data\\label_02_extracted\\past_1obj_LTWH'):
    for file in files:
      filename = os.path.join(subdir, file)

      with open(filename) as f:
        for i, l in enumerate(f):
          pass
      # print(str(i + 1), end='')
      if i + 1 != 10:
        print(str(i + 1) + " : " + filename)
        badpast_count += 1

      total_past_count += 1
  if badpast_count == 0:
    print("All {} past files are of correct length".format(total_past_count))

  total_future_count = 0
  badfuture_count = 0
  for subdir, dirs, files in os.walk('F:\\Car data\\label_02_extracted\\future_1obj_LTWH'):
    for file in files:
      filename = os.path.join(subdir, file)

      with open(filename) as f:
        for i, l in enumerate(f):
          pass
      # print(str(i + 1), end='')
      if i + 1 != 11:
        print(str(i + 1) + " : " + filename)
        badfuture_count += 1

      total_future_count += 1
  if badfuture_count == 0:
    print("All {} future files are of correct length".format(total_future_count))


def generateDataFiles_1obj(fpath):
  """
  Read each kitti data file; identify objects with 15-frame sequences and create 'past' samples using the first 10 frames,
  and create 'future' samples using the first 10 frames and the 15th (target) frame. Write each 'past' and 'future' sample to its own file.
  """
  f = open(fpath,'r')
  fname = os.path.splitext(fpath)[0]

  path_past = 'F:\\Car data\\label_02_extracted\\past_1obj_LTWH\\' + fname
  path_future = 'F:\\Car data\\label_02_extracted\\future_1obj_LTWH\\' + fname

  if not os.path.exists(path_past):
    os.makedirs(path_past)
  if not os.path.exists(path_future):
    os.makedirs(path_future)

  data_num = 0
  obj_bb_dict = {}
  obj_lastFrame_dict = {}

  while True:
    line = f.readline()
    if not line:
      f.close()
      return data_num

    words = line.split()
    frame_num = words[0]
    obj_id = words[1]

    if words[2] != "DontCare":
      L = float(words[6])
      T = float(words[7])
      R = float(words[8])
      B = float(words[9])
      bb = [str(L), str(T), str(R - L), str(B - T)]

      if obj_id in obj_bb_dict and obj_lastFrame_dict[obj_id] == int(frame_num) - 1:
        obj_queue = obj_bb_dict[obj_id]
        #if len(obj_queue) > 1: print(len(obj_queue))
        if len(obj_queue) >= 15:
          obj_queue.popleft()

        obj_queue.append(bb)
        obj_lastFrame_dict[obj_id] = int(frame_num)

        if len(obj_queue) == 15:
          fpast = open(path_past + '\\past' + str(data_num) +'_objId'+obj_id+'_frameNum'+frame_num+'.txt', 'w')
          ffuture = open(path_future + '\\future' + str(data_num) +'_objId'+obj_id+'_frameNum'+frame_num+'.txt', 'w')

          for i in range(10):
            fpast.write(" ".join(obj_queue[i]) + "\n")
            ffuture.write(" ".join(obj_queue[i]) + "\n")  # include past data along with future position
          ffuture.write(" ".join(obj_queue[14]) + "\n")

          fpast.close()
          ffuture.close()
          data_num += 1
      else:
        obj_bb_dict[obj_id] = deque([bb]) # create a new queue for this obj_id
        obj_lastFrame_dict[obj_id] = int(frame_num)




getLineCounts_1obj()

# file_list = os.listdir()
# for fpath in file_list:
#   print(fpath, ' . . . ...')
#   print(generateDataFiles_1obj(fpath))
