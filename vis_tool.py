import cv2
import numpy as np
import data_extract_1obj

def drawFrameRects(sample_set, frame, objId, bb, isGen, folder_dir):
    '''
    Draws the provided bounding boxes onto the specified frame and outputs the drawn image to 'folder_dir'.

    Args:
        sample_set (string): The set to get the specified frame from.
        frame (string): The frame to draw boxes on.
        objId (string): The id of the object being tracked.
        bb ((n,4) array): An array with each row being LTWH values describing a bounding box.
        isGen (boolean): Indicates whether the concluding bounding box in 'bb' was generated, or if it is the target.
        folder_dir: The folder to output the drawn images to.
    '''
    img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\' + sample_set + '\\' + frame + '.png'

    # Load img to draw on.
    img = cv2.imread(img_file)

    proposal = data_extract_1obj.transform(bb[-2], bb[-1])
    bb[-1] = proposal
    unnormal = data_extract_1obj.unnormalize_sample(bb, sample_set)

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
    cv2.imwrite(folder_dir + sample_set + '_' + frame + '_' + objId + '_' + suffix + '.png', img)

    return

class Rect:
    def __init__(self, l, t, r, b):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.area = (r - l) * (b - t)

    def __str__(self):
        return "l:" + str(self.l) + " t:" + str(self.t) + " r:" + str(self.r) + " b:" + str(self.b) + " area:" + str(self.area)

    @classmethod
    def make_center_XYWH(cls, cx, cy, w, h):
        l = cx - w/2
        t = cy - h/2
        r = cx + w/2
        b = cy + h/2
        return cls(l, t, r, b)

def get_IoU(anchor, target_transform, generated_transform, sample_set):
    t_bb = data_extract_1obj.transform(anchor, target_transform)
    g_bb = data_extract_1obj.transform(anchor, generated_transform)
    t_bb = data_extract_1obj.unnormalize_bb(t_bb, sample_set)
    g_bb = data_extract_1obj.unnormalize_bb(g_bb, sample_set)

    target = Rect(t_bb[0], t_bb[1], t_bb[0] + t_bb[2], t_bb[1] + t_bb[3])
    generated = Rect(g_bb[0], g_bb[1], g_bb[0] + g_bb[2], g_bb[1] + g_bb[3])
    intersect = Rect(max(target.l, generated.l), max(target.t, generated.t),
                     min(target.r, generated.r), min(target.b, generated.b))

    return intersect.area / (target.area + generated.area - intersect.area)
