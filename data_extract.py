#
# Run from "F:\Car data\kitti\data_tracking\training\label_02"
# $	python "C:\Users\Max\Research\data_extract.py"
#

import os

def generateDataFiles(fpath):
	f = open(fpath,'r')
	fname = os.path.splitext(fpath)[0]
	path_past = 'F:\\Car data\\label_02_extracted\\past\\' + fname
	path_future = 'F:\\Car data\\label_02_extracted\\future\\' + fname

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

			if first_time and frame_count == 1:
				next_pos = f.tell()
				frame = int(words[0])
				first_time = False

			if frame != int(words[0]):
				# fpast.write("frame:"+str(frame)+"\n") if frame_count < 10 else ffuture.write("frame:"+str(frame)+"\n")
				while objs_this_frame < 5:
					fpast.write("0 0 0 0\n") if frame_count < 10 else ffuture.write("0 0 0 0\n")
					objs_this_frame += 1
				frame = int(words[0])
				frame_count += 1
				objs_this_frame = 0

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


# generateDataFiles('0000.txt')

file_list = os.listdir()
for fpath in file_list:
	print(fpath, ' . . . ...')
	generateDataFiles(fpath)