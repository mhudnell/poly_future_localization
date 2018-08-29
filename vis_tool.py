import cv2
import numpy as np

# img = cv2.imread('F:\\Car data\\kitti\\data_tracking\\training\\image_02\\0000\\000000.png')

# img = cv2.rectangle(img, (-10,-10), (100,100), (0,255,0), 5)

# cv2.imwrite( "test.png", img );

''' drawFrameRects
Draws bounding boxes onto specified frame
params: 
		sample_set: the set to get the specified frame from.
		frame: the frame to draw boxes on.
		bb: a nx4 array of strings, each row being LTWH vals for a bounding box.
'''
def drawFrameRects(sample_set, frame, objId, bb, isGen, folder_dir='C:\\Users\\Max\\Research\\maxGAN\\bb images\\'):
	img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\'+ sample_set +'\\'+ frame + '.png'

	# load img to draw on
	img = cv2.imread(img_file)

	# convert bb (ltwh values) to int
	bb_int = np.zeros((len(bb),4), dtype='int32')
	for i in range(len(bb_int)):
		nums = bb[i]
		bb_int[i][0] = int(float(nums[0]))
		bb_int[i][1] = int(float(nums[1]))
		bb_int[i][2] = int(float(nums[2]))
		bb_int[i][3] = int(float(nums[3]))

	box_color = (0,255,0) if isGen else (255, 0, 0)

	# draw each bounding box on the image
	for i in range(len(bb_int)):
		if i < 10:
			img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][0] + bb_int[i][2], bb_int[i][1] + bb_int[i][3]), (61,165,244,0.5), 5)	# draw past frames in orange
		else:
			img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][0] + bb_int[i][2], bb_int[i][1] + bb_int[i][3]), box_color, 5)			# draw future frame in green
		# elif bb_int[i][0] <= bb_int[i][2] and bb_int[i][1] <= bb_int[i][3]:
		# 	# img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][2], bb_int[i][3]), box_color, 5)			# draw future frame in green if ltrb vals are valid
		# else:	# model has ltrb wrong
		# 	# img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][2], bb_int[i][3]), box_color, 5)			# draw future frame in red if ltrb vals are invalid

	# output image file to 'bb_images' folder
	suffix = "gen" if isGen else "real"
	cv2.imwrite(folder_dir + sample_set + '_' + frame + '_' + objId + '_' + suffix + '.png', img);

	return

# ''' drawFrameRects_1obj
# Draws bounding boxes onto specified frame
# params: 
# 		sample_set: the set to get the specified frame from.
# 		frame: the frame to draw boxes on.
# 		bb: a 5x4 array of strings, each row being ltrb vals for a bounding box.
# '''
# def drawFrameRects_1obj(sample_set, frame, bb, isGen, folder_dir='C:\\Users\\Max\\Research\\maxGAN\\bb images\\'):
# 	img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\'+ sample_set +'\\'+ frame
# 	img = cv2.imread(img_file)

# 	# convert bb (ltrb values) to int
# 	bb_int = np.zeros((5,4), dtype='int32')
# 	for i in range(5):
# 		nums = bb[i]
# 		bb_int[i][0] = int(float(nums[0]))
# 		bb_int[i][1] = int(float(nums[1]))
# 		bb_int[i][2] = int(float(nums[2]))
# 		bb_int[i][3] = int(float(nums[3]))

# 	# draw each bounding box on the image
# 	for i in range(5):
# 		if bb_int[i][0] <= bb_int[i][2] and bb_int[i][1] <= bb_int[i][3]:
# 			img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][2], bb_int[i][3]), (0,255,0), 5)	# draw in green if ltrb vals are valid
# 		else:	# model has ltrb wrong
# 			img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][2], bb_int[i][3]), (0,0,255), 5)	# draw in red if ltrb vals are invalid

# 	# output image file to 'bb_images' folder
# 	prefix = "gen" if isGen else "real"
# 	cv2.imwrite(folder_dir + prefix + frame, img);

# 	return

'''
draws bounding boxes onto multiple specified frames
	TODO: implement, it is the same as drawFrameRects right now
'''
def drawFramesRects(frame, bbs, isGen, folder_dir='C:\\Users\\Max\\Research\\maxGAN\\bb images\\'):
	img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\0000\\' + frame
	img = cv2.imread(img_file)

	rects = np.zeros((5,4), dtype='int32')
	for i in range(5):
		nums = bb[i]
		rects[i][0] = int(float(nums[0]))
		rects[i][1] = int(float(nums[1]))
		rects[i][2] = int(float(nums[2]))
		rects[i][3] = int(float(nums[3]))

	for i in range(5):
		if rects[i][0] <= rects[i][2] and rects[i][1] <= rects[i][3]:
			img = cv2.rectangle(img, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]), (0,255,0), 5)
		else:	# model has ltrb wrong
			img = cv2.rectangle(img, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]), (0,0,255), 5)

	prefix = "gen" if isGen else "real"
	cv2.imwrite(folder_dir + prefix + frame, img);

	return


# img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\0000\\000000.png'
# bb_file = 'F:\\Car data\\label_02_extracted\\future\\0000\\future0.txt'
# drawObjectRects(img_file, bb_file)
