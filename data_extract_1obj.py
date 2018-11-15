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

kitti_raw_map = [
    '2011_09_26_drive_0001_sync',
    '2011_09_26_drive_0002_sync',
    '2011_09_26_drive_0005_sync',
    '2011_09_26_drive_0009_sync',
    '2011_09_26_drive_0011_sync',
    '2011_09_26_drive_0013_sync',
    '2011_09_26_drive_0014_sync',
    '2011_09_26_drive_0015_sync',
    '2011_09_26_drive_0017_sync',
    '2011_09_26_drive_0018_sync',
    '2011_09_26_drive_0019_sync',
    '2011_09_26_drive_0020_sync',
    '2011_09_26_drive_0022_sync',
    '2011_09_26_drive_0023_sync',
    '2011_09_26_drive_0027_sync',
    '2011_09_26_drive_0028_sync',
    '2011_09_26_drive_0029_sync',
    '2011_09_26_drive_0032_sync',
    '2011_09_26_drive_0035_sync',
    '2011_09_26_drive_0036_sync',
    '2011_09_26_drive_0039_sync',
    '2011_09_26_drive_0046_sync',
    '2011_09_26_drive_0048_sync',
    '2011_09_26_drive_0051_sync',
    '2011_09_26_drive_0052_sync',
    '2011_09_26_drive_0056_sync',
    '2011_09_26_drive_0057_sync',
    '2011_09_26_drive_0059_sync',
    '2011_09_26_drive_0060_sync',
    '2011_09_26_drive_0061_sync',
    '2011_09_26_drive_0064_sync',
    '2011_09_26_drive_0070_sync',
    '2011_09_26_drive_0079_sync',
    '2011_09_26_drive_0084_sync',
    '2011_09_26_drive_0086_sync',
    '2011_09_26_drive_0087_sync',
    '2011_09_26_drive_0091_sync',
    '2011_09_26_drive_0093_sync'
]

def get_kitti_training(normalize=True, transform=True):
    # training_sets = ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007",
    #                  "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016"]
    training_sets2 = ["0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011",
                     "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]
    return get_kitti_data(training_sets2, normalize=normalize, transform=transform)

def get_kitti_testing(normalize=True, transform=True):
    # testing_sets = ["0017", "0018", "0019", "0020"]
    testing_sets2 = ["0000", "0001", "0002", "0003"]
    # three-fold_1 = ["0000", "0001", "0002"]
    # three-fold_1 = ["0003", "0001", "0002"]
    return get_kitti_data(testing_sets2, normalize=normalize, transform=transform)

def get_kitti_data(sets, normalize=True, transform=True):
    """
    Parse kitti label files and construct a set of samples. Each sample has 11 'bounding boxes', where each
    bounding box stores 4 floats (left, top, width, height). The 11th bounding box is the 'target'

    Returns:
        samples (ndarray (ndim: 3)): (m, 11, 4)
    """
    samples = []
    samples_info = []  # (sample set, frame number)

    for root, _, file_names in os.walk("F:\\Car data\\kitti\\data_tracking\\training\\label_02\\"):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            sample_set_name = os.path.splitext(file_name)[0]
            if sets is not None and int(sample_set_name) not in sets:  # Skip over sets not requested.
                # print("RESERVING " + sample_set_name + " for testing")
                # print(int(sample_set_name), "not in ", sets)
                continue
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
                        cx = (L + R)/2
                        cy = (T + B)/2
                        # bb = normalize_bb([L, T, R - L, B - T], sample_set_name) if normalize else [L, T, R - L, B - T]
                        bb = normalize_bb([cx, cy, R - L, B - T], sample_set_name) if normalize else [cx, cy, R - L, B - T]

                        if object_id in object_last_seen and object_last_seen[object_id] == frame_number - 1:  # Object has been seen previously AND the object was seen in the previous frame
                            object_queue = object_queues[object_id]
                            object_queue.append(bb)

                            # if len(object_queue) >= 15:  # have 1.5sec of continuous info=> Can create a sample
                            #     if len(object_queue) == 16:
                            if len(object_queue) >= 20:  # have 2.0sec of continuous info=> Can create a sample
                                if len(object_queue) == 21:
                                    object_queue.popleft()
                                sample = []
                                for i in range(10):
                                    sample.append(object_queue[i])
                                if transform:
                                    # target = get_transformation(object_queue[9], object_queue[14])
                                    target = get_transformation(object_queue[9], object_queue[19])
                                else:
                                    raise Exception('should not happen.')
                                    # target = object_queue[14]
                                sample.append(target)  # Append the target box (or transformation)
                                samples.append(sample)
                                samples_info.append((sample_set_name, str(frame_number).zfill(6), object_id))

                        else:
                            object_queues[object_id] = deque([bb])  # Reset / create a new queue of bounding boxes for this object

                    object_last_seen[object_id] = frame_number  # Update last seen frame for this object

        return np.asarray(samples), samples_info

def get_kitti_raw_tracklets(timepoints, sets=None, normalize=True, use_occluded=True, class_types=['Car', 'Van', 'Truck'], past_frames=10):
    """
    Parse kitti tracklet files and construct a set of samples [input X and targets Y].

    Arguments:
        timepoints: array of timepoints, should be rounded to tenths of a second.
        sets: [set of integers; range: 0-37] if specified, only retrieve these sets
    Returns:
        x: [N x 10 x 4] 10 past bounding boxes. Normalized [cx, cy, w, h] values.
        y: [N x 4 x (# of future frames)] transformation parameters. Each column corresponds to a transformation for a timestep.
        samples_info: [N x 5] [<sequence dir>, <anchor frame filename>, <final frame filename>, <object id>, <object class>]
    """
    x = []
    y = []
    samples_info = []  # (seq_dir, anchor frame, final frame, object id, object class)

    future_frames = (timepoints*10).astype(int)
    num_required_frames = future_frames[-1]
    print('future_frames:', future_frames)
    print('num_required_frames:', num_required_frames)

    root = "/data/b/mhudnell_cvpr_2019/2011_09_26_tracklets_only"
    for i, seq in enumerate(os.listdir(root)):
        # print(i, seq)
        seq_dir = os.path.join(root, seq)
        if not os.path.isdir(seq_dir) or (sets is not None and i not in sets):
            continue
        # print(seq_dir)

        tracklet_path = os.path.join(seq_dir, "2d_tracklet_filtered.txt")
        object_queues = {}  # Contains a queue (of bounding boxes) with max size 15 for each object id in the file
        object_last_seen = {}  # Key: Object ids; Value: Last seen frame for the given object

        with open(tracklet_path) as f:
            for line in f:
                words = line.split()
                frame_number = int(words[0])
                object_id = words[1]
                object_class = words[2]

                if (not use_occluded) and (words[4] not in ['0', '1']):
                    continue

                # Add classes to include all vehicles, or pedestrians.
                if object_class not in class_types:     #        # 'Pedestrian', 'Cyclist'
                    continue

                L = float(words[6])
                T = float(words[7])
                R = float(words[8])
                B = float(words[9])
                w = R - L
                h = B - T
                cx = (L + R)/2
                cy = (T + B)/2
                bb = [cx/1242, cy/375, w/1242, h/375] if normalize else [cx, cy, w, h]

                # Object was seen in the previous frame?
                if object_id in object_last_seen and object_last_seen[object_id] == frame_number - 1:  
                    object_queue = object_queues[object_id]
                    object_queue.append(bb)

                    # Check for required period of continuous info => Can create a sample
                    if len(object_queue) >= 10 + num_required_frames:
                        if len(object_queue) == 10 + num_required_frames + 1:
                            object_queue.popleft()

                        x_sample = np.empty((past_frames, 4))
                        for i in range(10-past_frames, 10):
                            x_sample[i-(10-past_frames)] = object_queue[i]

                        y_sample = np.empty((4, len(future_frames)))
                        for i, j in enumerate(future_frames):
                            t = get_transformation(object_queue[9], object_queue[9+j])
                            y_sample[:, i] = t

                        x.append(x_sample)
                        y.append(y_sample)

                        anchor_frame = str(frame_number - num_required_frames).zfill(10)
                        final_frame = str(frame_number).zfill(10)
                        samples_info.append((seq_dir, anchor_frame, final_frame, object_id, object_class))

                else:
                    object_queues[object_id] = deque([bb])  # Reset / create a new queue of bounding boxes for this object

                object_last_seen[object_id] = frame_number  # Update last seen frame for this object

    return np.asarray(x), np.asarray(y), samples_info

def get_epoch(samples, batch_size, seed=0):
    """
    Retrieve n batches of size batch_size, where n is the number of batches needed to fit an entire epoch.
    Leaves out some samples if `samples` is not divisible by `batch_size`

    Returns a list of lists, where each internal list is a batch
    """
    if seed:  # For Testing
        print("Getting seeded epoch")
        np.random.seed(seed)

    num_samples = len(samples)
    num_batches = num_samples // batch_size
    indices = np.random.choice(num_samples, size=(num_batches, batch_size), replace=False)

    vectorized_samples = samples.reshape((num_samples, 44))
    batches = vectorized_samples[indices]

    return batches



def get_batch(samples, batch_size, seed=None):
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

def get_batch_ids(num_samples, batch_size, seed=None):
    """
    Retreive (batch_size) number of indices from [0-num_samples).

    Returns:
        ndarray (ndim: 2):
    """

    if seed:  # For Testing
        print("Getting seeded batch")
        np.random.seed(seed)

    indices = np.random.choice(num_samples, size=batch_size, replace=False)

    return indices

def random_flip_batch(x_batch, y_batch):
    """Randomly flips cx of bounding box, negates transformation, for all samps"""
    assert (x_batch.shape[-2:] == (10, 4)), "x_batch must be shape (?, 10, 4)"
    assert (y_batch.shape[-2:] == (4, 10)), "y_batch must be shape (?, 4, 10)"

    if np.random.randint(0, 1):
        x_batch[:, :, 0] = 1 - x_batch[:, :, 0]
        y_batch[:, 0, :] = -y_batch[:, 0, :]


def scale_bb(bb, sample_set, desired_res):
    """
    Return the scaled version of the specified bounding box, such that it is the desired resolution.
    Necessary because kitti sample sets come in slightly different dimensions, so one is chosen as
    the 'best', and all other sets are scaled to the 'best' resolution.
    """
    sample_dimensions = set_dimensions[sample_set]
    scale_w = desired_res[1] / sample_dimensions[1]
    scale_h = desired_res[0] / sample_dimensions[0]
    scaled = np.empty(4)
    scaled[0] = bb[0] * scale_w
    scaled[1] = bb[1] * scale_h
    scaled[2] = bb[2] * scale_w
    scaled[3] = bb[3] * scale_h
    return scaled

def descale_bb(bb, sample_set, desired_res):
    """
    Return the de-scaled version of the specified bounding box, such that it is the sample set's resolution.
    The input bb is assumed to currently be in the resolution of `desired_res`.
    """
    sample_dimensions = set_dimensions[sample_set]
    scale_w = sample_dimensions[1] / desired_res[1]
    scale_h = sample_dimensions[0] / desired_res[0]
    descaled = np.empty(4)
    descaled[0] = bb[0] * scale_w
    descaled[1] = bb[1] * scale_h
    descaled[2] = bb[2] * scale_w
    descaled[3] = bb[3] * scale_h
    return descaled

def normalize_bb(bb, sample_set, desired_res=(375, 1242)):
    """
    Return the normalized values for the bounding box passed in. Works for
    both [CX, CY, W, H] and [L, T, W, H] bounding boxes.
    """
    # Scale to desired dimensions before normalizing
    scaled = scale_bb(bb, sample_set, desired_res)
    # dimensions = set_dimensions[sample_set]
    # h = dimensions[0]
    # w = dimensions[1]
    h = desired_res[0]
    w = desired_res[1]
    normalized = np.empty(4)
    normalized[0] = scaled[0] / w
    normalized[1] = scaled[1] / h
    normalized[2] = scaled[2] / w
    normalized[3] = scaled[3] / h
    return normalized

def unnormalize_bb(bb, sample_set=None, desired_res=(375, 1242), top_left=False):
    """Return the denormalized values (in LTWH format) for the bounding box passed in. Assumes normalized format is [CX, CY, W, H]."""
    h = desired_res[0]
    w = desired_res[1]
    bb_w = bb[2] * w
    bb_h = bb[3] * h
    denormalized = np.empty(4)
    denormalized[0] = bb[0] * w
    denormalized[1] = bb[1] * h
    denormalized[2] = bb_w
    denormalized[3] = bb_h
    # Scale back to original dimensions after denormalizing
    if sample_set is not None:
        denormalized = descale_bb(denormalized, sample_set, desired_res)
    return denormalized

def unnormalize_sample(sample, sample_set, top_left=False):
    """Return the denormalized bbs for the sample passed in."""
    denormalized = np.empty((len(sample), 4))
    for i, bb in enumerate(sample):
        unnormal = unnormalize_bb(bb, sample_set)
        denormalized[i] = center_to_topleft_bb(unnormal)
    return denormalized


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
    try:
        proposal[0] = anchor[2]*t[0] + anchor[0]
        proposal[1] = anchor[3]*t[1] + anchor[1]
        proposal[2] = anchor[2]*exp(t[2])
        proposal[3] = anchor[3]*exp(t[3])
    except OverflowError as err:
        print('Overflowed after ', t, err)
        proposal = np.zeros(4)
    return proposal

def center_to_topleft_bb(bb):
    """Convert bounding box from format [CX, CY, W, H] to [L, T, W, H]."""
    topleft = np.empty(4)
    topleft[0] = bb[0] - bb[2] / 2
    topleft[1] = bb[1] - bb[3] / 2
    topleft[2] = bb[2]
    topleft[3] = bb[3]
    return topleft

if __name__ == '__main__':
    # samples, _ = get_kitti_data(normalize=True)
    # batch = get_batch(samples, 3, seed=7)
    # print(batch)
    # print(batch[:, :4*10])

    # samples_train, _ = get_kitti_training(normalize=True)
    
    # print(samples)
    # print("shape: ", samples_train.shape)
    # print("ndim: ", samples.ndim)
    # print("dtype: ", samples.dtype)

    # samples_test, _ = get_kitti_testing(normalize=True)
    

    # epoch = get_epoch(samples, 128)
    # print("epoch shape:", epoch.shape)

    # samples, samples_info = get_kitti_raw_data()
    timepoints = np.linspace(0.1, 1.0, 10)
    x, y, samples_info = get_kitti_raw_tracklets(timepoints, use_occluded=True)

    print('x shape:', x.shape)
    print('y shape:', y.shape)
    print('info length:', len(samples_info))

    print(x[0])
    print(y[0])
