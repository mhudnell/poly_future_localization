import os
import numpy as np

# file_list = os.listdir('F:\\Car data\\label_02_extracted\\past')
# print(file_list)

past_all = np.empty([7703, 50, 4])

file_num1 = 0
for path, subdirs, files in os.walk('F:\\Car data\\label_02_extracted\\past'):
	for name in files:
		fpath = os.path.join(path, name)
		f = open(fpath,'r')
		past_one = np.empty([50, 4])
		for i in range(50):
			line = f.readline()
			if not line:
				break
			words = line.split()
			past_one[i] = [float(word) for word in words]
		past_all[file_num1] = past_one
		file_num1 += 1

future_all = np.empty([7703, 25, 4])

file_num2 = 0
for path, subdirs, files in os.walk('F:\\Car data\\label_02_extracted\\future'):
	for name in files:
		fpath = os.path.join(path, name)
		f = open(fpath,'r')
		future_one = np.empty([25, 4])
		for i in range(25):
			line = f.readline()
			if not line:
				break
			words = line.split()
			future_one[i] = [float(word) for word in words]
		future_all[file_num2] = future_one
		file_num2 += 1

def max_get_data_batch(data_x, data_y, batch_size, seed=0):
    start_i = (batch_size * seed) % len(data_x)
    stop_i = start_i + batch_size
    shuffle_seed = (batch_size * seed) // len(data_x)
    np.random.seed(shuffle_seed)
    indices = np.random.choice(len(data_x), size=len(data_x), replace=False ) # wasteful to shuffle every time
    # print(indices)
    indices = list(indices) + list(indices) # duplicate to cover ranges past the end of the set
    print(indices[ start_i: stop_i ])
    x = data_x[ indices[ start_i: stop_i ] ]
    y = data_y[ indices[ start_i: stop_i ] ]

    print('x.shape: ', x.shape)
    print('y.shape: ', y.shape)
    
    # return np.reshape(x, (batch_size, -1) )
    return x, y

print('file_num1: ', file_num1)
print('file_num2: ', file_num2)
print('past_all.shape: ', past_all.shape)
print('future_all.shape: ', future_all.shape)


x, y = max_get_data_batch(past_all, future_all, 10, seed=7)
# print(past_all[7702])

# f = open('F:\\Car data\\label_02_extracted\\past\\0000\\past0.txt','r')
# past_one = np.empty([50, 4])
# for i in range(50):
# 	line = f.readline()
# 	if not line:
# 		break
# 	words = line.split()
# 	past_one[i] = [float(word) for word in words]

# print(past_one)