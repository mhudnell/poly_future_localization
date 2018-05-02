#
# Run from "F:\Car data\kitti\data_tracking\training\label_02"
# $	python "C:\Users\Max\Research\maxGAN\data_extract.py"
#

import os

def getLineCounts():

	total_past_count = 0
	badpast_count = 0
	for subdir, dirs, files in os.walk('F:\\Car data\\label_02_extracted\\past'):
		for file in files:
			filename = os.path.join(subdir, file)

			with open(filename) as f:
				for i, l in enumerate(f):
					pass
			# print(str(i + 1), end='')
			if i + 1 != 50:
				print(str(i + 1) + " : " + filename)
				badpast_count += 1

			total_past_count += 1
	if badpast_count == 0:
		print("All {} past files are of correct length".format(total_past_count))

	total_future_count = 0
	badfuture_count = 0
	for subdir, dirs, files in os.walk('F:\\Car data\\label_02_extracted\\future'):
		for file in files:
			filename = os.path.join(subdir, file)

			with open(filename) as f:
				for i, l in enumerate(f):
					pass
			# print(str(i + 1), end='')
			if i + 1 != 25:
				print(str(i + 1) + " : " + filename)
				badfuture_count += 1

			total_future_count += 1
	if badfuture_count == 0:
		print("All {} future files are of correct length".format(total_future_count))

def generateDataFiles(fpath):
	f = open(fpath,'r')
	fname = os.path.splitext(fpath)[0]
	# path_past = 'F:\\Car data\\label_02_extracted\\past\\' + fname
	# path_future = 'F:\\Car data\\label_02_extracted\\future\\' + fname

	path_past = 'F:\\Car data\\label_02_extracted\\past_test\\' + fname
	path_future = 'F:\\Car data\\label_02_extracted\\future_test\\' + fname

	if not os.path.exists(path_past):
		os.makedirs(path_past)
	if not os.path.exists(path_future):
		os.makedirs(path_future)
	
	frame = 0
	objs_this_frame = 0
	data_num = 0
	next_pos = 0

	while True:
		fpast = open(path_past + '\\past' + str(data_num) +'.txt', 'w')
		ffuture = open(path_future + '\\future' + str(data_num) +'.txt', 'w')
		frame_count = 0
		first_time = True

		while frame_count < 15:
			# fpast.write("f.tell(): " + str(f.tell()) +"\n")
			next_pos_candidate = f.tell()
			line = f.readline()
			if not line:
				# fpast.write("frame:"+str(frame)+"\n") if frame_count < 10 else ffuture.write("frame:"+str(frame)+"\n")
				while objs_this_frame < 5:
					fpast.write("0 0 0 0\n") if frame_count < 10 else ffuture.write("0 0 0 0\n")
					objs_this_frame += 1
				f.close()
				fpast.close()
				ffuture.close()
				return

			words = line.split()

			# we hit the next frame in kitti dataset
			if frame != int(words[0]):
				# fpast.write("frame:"+str(frame)+"\n") if frame_count < 10 else ffuture.write("frame:"+str(frame)+"\n")

				# record position of second frame for next iteration
				if first_time:
					next_pos = next_pos_candidate
					first_time = False

				# zero pad if there were less than 5 objects
				while objs_this_frame < 5:
					fpast.write("0 0 0 0\n") if frame_count < 10 else ffuture.write("0 0 0 0\n")
					objs_this_frame += 1

				frame = int(words[0])
				frame_count += 1
				objs_this_frame = 0

				if frame_count == 15: break

			# record first 5 objects which are important (not "DontCare")
			if objs_this_frame < 5 and words[2] != "DontCare":
					bb = [words[6], words[7], words[8], words[9]]
					obj_line = " ".join(bb)
					fpast.write(obj_line + "\n") if frame_count < 10 else ffuture.write(obj_line + "\n")
					objs_this_frame += 1

		fpast.close()
		ffuture.close()
		f.seek(next_pos)
		data_num += 1
		frame = data_num


getLineCounts()
# generateDataFiles('0002.txt')

# file_list = os.listdir()
# for fpath in file_list:
# 	print(fpath, ' . . . ...')
# 	generateDataFiles(fpath)