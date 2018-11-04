import cv2
import numpy as np
import data_extract_1obj

def drawFrameRects(sample_set, frame, objId, bb, isGen, folder_dir, dataset='kitti_tracking', display=False):
    '''
    Draws the provided bounding boxes onto the specified frame and outputs the drawn image to 'folder_dir'.

    Args:
        sample_set (string): The set to get the specified frame from.
        frame (string): The frame to draw boxes on.
        objId (string): The id of the object being tracked.
        bb ((n,4) array): An array with each row being LTWH values describing a bounding box. bb[-1] is a transform, bb[-2] is anchor box.
        isGen (boolean): Indicates whether the concluding bounding box in 'bb' was generated, or if it is the target.
        folder_dir: The folder to output the drawn images to.
    '''
    print(sample_set)
    if dataset == 'kitti_raw':
        img_file = sample_set + '\\image_02\\data\\' + frame + '.png'
    elif dataset == 'kitti_raw_tracklets':
        img_file = sample_set + '\\image_02\\data\\' + frame + '.png'
    else:
        img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\' + sample_set + '\\' + frame + '.png'

    print(img_file)
    # Load img to draw on.
    img = cv2.imread(img_file)

    proposal = data_extract_1obj.transform(bb[-2], bb[-1])
    bb[-1] = proposal

    if dataset in ['kitti_raw', 'kitti_raw_tracklets']:
        unnormal = np.empty(bb.shape)
        unnormal[:, 0] = bb[:, 0] * 1242
        unnormal[:, 1] = bb[:, 1] * 375
        unnormal[:, 2] = bb[:, 2] * 1242
        unnormal[:, 3] = bb[:, 3] * 375
        # print("unnormal:", unnormal)
        for i, bb in enumerate(unnormal):
            unnormal[i] = data_extract_1obj.center_to_topleft_bb(bb)
        # print("top-left pix:", unnormal)
    else:
        unnormal = data_extract_1obj.unnormalize_sample(bb, sample_set)
    # topleft = data_extract_1obj.center_to_topleft_bb(unnormal)

    # Convert bb (LTWH values) to ints.
    bb_int = np.zeros((len(unnormal), 4), dtype='int32')
    for i in range(len(bb_int)):
        nums = unnormal[i]
        bb_int[i][0] = int(float(nums[0]))
        bb_int[i][1] = int(float(nums[1]))
        bb_int[i][2] = int(float(nums[2]))
        bb_int[i][3] = int(float(nums[3]))

    # Draw target box in green if ground truth, blue if generated (color is specified in (b,g,r) format)
    target_color = (255, 0, 0) if isGen else (0, 255, 0)

    # Draw each bounding box on the image.
    for i in range(len(bb_int)):
        if i < 10:
            img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][0] + bb_int[i][2], bb_int[i][1] + bb_int[i][3]), (61, 165, 244, 0.5), 5)     # draw past frames in orange
        else:
            img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][0] + bb_int[i][2], bb_int[i][1] + bb_int[i][3]), target_color, 5)            # draw future frame in color: 'target_color'

    # Write image to file.
    suffix = "generated" if isGen else "target"

    if display:
        cv2.imshow('ImageWindow', img)
        cv2.waitKey()
    else:
        cv2.imwrite(folder_dir + sample_set + '_' + frame + '_' + objId + '_' + suffix + '.png', img)

    return

class Rect:
    def __init__(self, l, t, r, b):
        assert (l < r), "l should be less than r"
        assert (t < b), "t should be less than b"

        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.area = (r - l) * (b - t)

    def __str__(self):
        return "l:" + str(self.l) + " t:" + str(self.t) + " r:" + str(self.r) + " b:" + str(self.b) + " area:" + str(self.area)

    @classmethod
    def make_cXcYWH(cls, cx, cy, w, h):
        l = cx - w/2
        t = cy - h/2
        r = cx + w/2
        b = cy + h/2
        return cls(l, t, r, b)

    @classmethod
    def make_LTWH(cls, l, t, w, h):
        r = l + w
        b = t + h
        return cls(l, t, r, b)

    @classmethod
    def get_intersection(cls, rect1, rect2):
        l = max(rect1.l, rect2.l)
        t = max(rect1.t, rect2.t)
        r = min(rect1.r, rect2.r)
        b = min(rect1.b, rect2.b)

        if l < r and t < b:
            return cls(l, t, r, b)
        else:
            return False

    @staticmethod
    def get_IoU(rect1, rect2):
        intersect = Rect.get_intersection(target, generated)
        if intersect:
            iou = intersect.area / (target.area + generated.area - intersect.area)
            assert (iou > 0), "Non-positive IoU!"
            return iou
        return 0
    
    @staticmethod
    def get_DE(rect1, rect2):
        c1 = np.array([(rect1.l + rect1.r)/2, (rect1.t + rect1.b)/2])
        c2 = np.array([(rect2.l + rect2.r)/2, (rect2.t + rect2.b)/2])
        return np.linalg.norm(c1 - c2)


def get_IoU(anchor, target_transform, generated_transform, sample_set=None):
    t_bb = data_extract_1obj.transform(anchor, target_transform)
    g_bb = data_extract_1obj.transform(anchor, generated_transform)
    t_bb = data_extract_1obj.unnormalize_bb(t_bb, sample_set=sample_set)
    g_bb = data_extract_1obj.unnormalize_bb(g_bb, sample_set=sample_set)

    target = Rect.make_cXcYWH(t_bb[0], t_bb[1], t_bb[2], t_bb[3])
    generated = Rect.make_cXcYWH(g_bb[0], g_bb[1], g_bb[2], g_bb[3])
    intersect = Rect.get_intersection(target, generated)

    if intersect:
        iou = intersect.area / (target.area + generated.area - intersect.area)
        assert (iou > 0), "Non-positive IoU!"
        return iou

    return 0

def calc_metrics(anchor, target_transform, generated_transform, sample_set=None):
    """ Calculates displacement error and IoU metrics """
    t_bb = data_extract_1obj.transform(anchor, target_transform)
    g_bb = data_extract_1obj.transform(anchor, generated_transform)
    t_bb = data_extract_1obj.unnormalize_bb(t_bb, sample_set=sample_set)
    g_bb = data_extract_1obj.unnormalize_bb(g_bb, sample_set=sample_set)

    target = Rect.make_cXcYWH(t_bb[0], t_bb[1], t_bb[2], t_bb[3])
    generated = Rect.make_cXcYWH(g_bb[0], g_bb[1], g_bb[2], g_bb[3])

    iou = Rect.get_IoU(target, generated)
    de = Rect.get_DE(target, generated)

    return [iou, de]


if __name__ == '__main__':

    # samples, samples_info = data_extract_1obj.get_kitti_data(sets=None)
    # samples, samples_info = data_extract_1obj.get_kitti_raw_data()
    samples, samples_info = data_extract_1obj.get_kitti_raw_tracklets()

    # print(samples.shape)
    # print(len(samples_info))

    # x = samples[0]
    # x_info = samples_info[0]
    # print(x)
    # print(x_info)

    # drawFrameRects(x_info[0], x_info[1], x_info[2], x, False, None, dataset='kitti_raw', display=True)
    # drawFrameRects(x_info[0], x_info[1], x_info[2], x, False, None, dataset='kitti_raw_tracklets', display=True)

    # View multiple
    # for i in range(5):
    #     x = samples[i]
    #     x_info = samples_info[i]

    #     print(x)
    #     print(x_info)

    #     drawFrameRects(x_info[0], x_info[1], x_info[2], x, False, None, dataset='kitti_raw_tracklets', display=True)
