#
# (for generateDataFiles_1obj) Run from "F:\Car data\kitti\data_tracking\training\label_02"
# $ cd "F:\Car data\kitti\data_tracking\training\label_02"
# $ python "C:\Users\Max\Research\maxGAN\data_extract_1obj.py"
#

import numpy as np
import os
from collections import deque
from math import log, exp

set_dimensions = {
        '0000': (375, 1242, 3),
        '0001': (375, 1242, 3),
        '0002': (375, 1242, 3),
        '0003': (375, 1242, 3),
        '0004': (375, 1242, 3),
        '0005': (375, 1242, 3),
        '0006': (375, 1242, 3),
        '0007': (375, 1242, 3),
        '0008': (375, 1242, 3),
        '0009': (375, 1242, 3),
        '0010': (375, 1242, 3),
        '0011': (375, 1242, 3),
        '0012': (375, 1242, 3),
        '0013': (375, 1242, 3),
        '0014': (370, 1224, 3),
        '0015': (370, 1224, 3),
        '0016': (370, 1224, 3),
        '0017': (370, 1224, 3),
        '0018': (374, 1238, 3),
        '0019': (374, 1238, 3),
        '0020': (376, 1241, 3),
}

def getLineCounts_1obj():
    """Check that all 'past' data files have 10 lines, and 'future' data files have 11 lines."""
    total_past_count = 0
    badpast_count = 0
    for subdir, _, files in os.walk('F:\\Car data\\label_02_extracted\\past_1obj_LTWH'):
        for file in files:
            filename = os.path.join(subdir, file)

            with open(filename) as f:
                for i, _ in enumerate(f):
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
    for subdir, _, files in os.walk('F:\\Car data\\label_02_extracted\\future_1obj_LTWH'):
        for file in files:
            filename = os.path.join(subdir, file)

            with open(filename) as f:
                for i, _ in enumerate(f):
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

def get_kitti_data(normalize=True, transform=True):
    """
    Parse kitti label files and construct a set of samples. Each sample has 11 'bounding boxes', where each
    bounding box stores 4 floats (left, top, width, height). The 11th bounding box is the 'target'

    Returns:
        samples (ndarray (ndim: 3)):
    """
    samples = []
    samples_info = []  # (sample set, frame number)

    for root, _, file_names in os.walk("F:\\Car data\\kitti\\data_tracking\\training\\label_02\\"):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            sample_set_name = os.path.splitext(file_name)[0]
            object_queues = {}  # Contains a queue (of bounding boxes) with max size 15 for each object id in the file
            object_last_seen = {}  # Key: Object ids; Value: Last seen frame for the given object

            with open(file_path) as f:
                for line in f:
                    words = line.split()
                    frame_number = int(words[0])
                    object_id = words[1]

                    if words[2] != "DontCare":
                        L = float(words[6])
                        T = float(words[7])
                        R = float(words[8])
                        B = float(words[9])
                        bb = normalize_bb([L, T, R - L, B - T], sample_set_name) if normalize else [L, T, R - L, B - T]

                        if object_id in object_last_seen and object_last_seen[object_id] == frame_number - 1:  # Object has been seen previously AND the object was seen in the previous frame
                            object_queue = object_queues[object_id]
                            object_queue.append(bb)

                            if len(object_queue) >= 15:  # => Can create a sample
                                if len(object_queue) == 16:
                                    object_queue.popleft()
                                sample = []
                                for i in range(10):
                                    sample.append(object_queue[i])
                                if transform:
                                    target = get_transformation(object_queue[9], object_queue[14])
                                else:
                                    target = object_queue[14]
                                sample.append(target)  # Append the target box (or transformation)
                                samples.append(sample)
                                samples_info.append((sample_set_name, str(frame_number).zfill(6), object_id))

                        else:
                            object_queues[object_id] = deque([bb])  # Reset / create a new queue of bounding boxes for this object

                    object_last_seen[object_id] = frame_number  # Update last seen frame for this object

        return np.asarray(samples), samples_info

def get_batch(samples, batch_size, seed=0):
    """
    Retreive (batch_size) number of samples. Each sample is vectorized.

    Returns:
        ndarray (ndim: 2):
    """

    if seed:  # For Testing
        print("Getting seeded batch")
        np.random.seed(seed)

    indices = np.random.choice(len(samples), size=batch_size, replace=False)
    batch = samples[indices]

    return np.reshape(batch, (batch_size, -1))

def normalize_bb(bb, sample_set):
    """Normalizes the sample passed in. Since python is pass-by-object, the return value is actually unnecessary."""
    dimensions = set_dimensions[sample_set]
    h = dimensions[0]
    w = dimensions[1]
    bb[0] = bb[0] / w
    bb[1] = bb[1] / h
    bb[2] = bb[2] / w
    bb[3] = bb[3] / h
    return bb

def unnormalize_sample(sample, sample_set):
    """Unnormalizes the sample passed in. Since python is pass-by-object, the return value is actually unnecessary."""
    dimensions = set_dimensions[sample_set]
    h = dimensions[0]
    w = dimensions[1]
    sample[:, 0] = sample[:, 0] * w
    sample[:, 1] = sample[:, 1] * h
    sample[:, 2] = sample[:, 2] * w
    sample[:, 3] = sample[:, 3] * h
    return sample


def get_transformation(anchor, target):
    """Calculate the transformation (t), which goes from the anchor box, to the target box."""
    t = np.empty(4)
    t[0] = (target[0] - anchor[0]) / anchor[2]
    t[1] = (target[1] - anchor[1]) / anchor[3]
    t[2] = log(target[2] / anchor[2])
    t[3] = log(target[3] / anchor[3])
    return t

def transform(anchor, t):
    """Apply transformation t to anchor box to get a proposal box."""
    proposal = np.empty(4)
    proposal[0] = anchor[2]*t[0] + anchor[0]
    proposal[1] = anchor[3]*t[1] + anchor[1]
    proposal[2] = anchor[2]*exp(t[2])
    proposal[3] = anchor[3]*exp(t[3])
    return proposal

if __name__ == '__main__':
    # samples, _ = get_kitti_data(normalize=True)
    # batch = get_batch(samples, 3, seed=7)
    # print(batch)
    # print(batch[:, :4*10])

    samples, _ = get_kitti_data(normalize=True)
    print(samples)
    print("shape: ", samples.shape)
    print("ndim: ", samples.ndim)
    print("dtype: ", samples.dtype)
