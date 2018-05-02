#
# Run from "F:\Car data\kitti\data_tracking\training\label_02"
# $	python "C:\Users\Max\Research\maxGAN\data_extract_1obj.py"
#

import os
from collections import deque

def getLineCounts_1obj():

	total_past_count = 0
	badpast_count = 0
	for subdir, dirs, files in os.walk('F:\\Car data\\label_02_extracted\\past_1obj'):
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
	for subdir, dirs, files in os.walk('F:\\Car data\\label_02_extracted\\future_1obj'):
		for file in files:
			filename = os.path.join(subdir, file)

			with open(filename) as f:
				for i, l in enumerate(f):
					pass
			# print(str(i + 1), end='')
			if i + 1 != 1:
				print(str(i + 1) + " : " + filename)
				badfuture_count += 1

			total_future_count += 1
	if badfuture_count == 0:
		print("All {} future files are of correct length".format(total_future_count))

def generateDataFiles_1obj(fpath):
	f = open(fpath,'r')
	fname = os.path.splitext(fpath)[0]

	path_past = 'F:\\Car data\\label_02_extracted\\past_1obj_test\\' + fname
	path_future = 'F:\\Car data\\label_02_extracted\\future_1obj_test\\' + fname

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
			bb = [words[6], words[7], words[8], words[9]]

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
					ffuture.write(" ".join(obj_queue[14]) + "\n")

					fpast.close()
					ffuture.close()
					data_num += 1
			else:
				obj_bb_dict[obj_id] = deque([bb])
				obj_lastFrame_dict[obj_id] = int(frame_num)

getLineCounts_1obj()

# file_list = os.listdir()
# for fpath in file_list:
# 	print(fpath, ' . . . ...')
# 	print(generateDataFiles_1obj(fpath))
