import cv2
import numpy as np
import data_extract_1obj
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
import matplotlib.patches as patches

def drawFrameRects(sample_set, frame, objId, bb_orig, isGen, folder_dir, dataset='kitti_tracking', display=False, anchor_frame=False):
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
    bb = np.copy(bb_orig)
    # print(sample_set)

    if anchor_frame:
        frame_num = int(frame)
        anchor_frame_num = frame_num - 10
        frame = str(anchor_frame_num).zfill(10)

    if dataset == 'kitti_raw':
        # img_file = sample_set + '\\image_02\\data\\' + frame + '.png'
        img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\' + sample_set + '\\' + frame + '.png'
    elif dataset == 'kitti_raw_tracklets':
        img_file = sample_set + '\\image_02\\data\\' + frame + '.png'
    else:
        raise Exception("`dataset` parameter must be one of: ['kitti_tracking', 'kitti_raw_tracklets']")

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
    # target_color = (255, 0, 0) if isGen else (0, 255, 0)

    # Draw each bounding box on the image.
    for i in range(len(bb_int)):
        if i < 9:       # Color for prior bbs
            if i % 3 != 0:
                continue
            color = (255, 102, 153)
        elif i == 9:   # Color for anchor box
            color = (102, 224, 225)
        else:           # Color for prediction / ground truth
            color = (255, 0, 0) if isGen else (0, 255, 0)
        
        img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][0] + bb_int[i][2], bb_int[i][1] + bb_int[i][3]), color, 2)

    # Write image to file.
    suffix = "generated" if isGen else "target"

    if display:
        cv2.imshow('ImageWindow', img)
        cv2.waitKey()
    else:
        # cv2.imwrite(folder_dir + sample_set + '_' + frame + '_' + objId + '_' + suffix + '.png', img)
        cv2.imwrite(os.path.join(folder_dir, frame + '_' + objId + '_' + suffix + '.png'), img)

    return

def draw_p_and_gt(set_info, prior_bbs, t_p, t_gt, output_dir, unnormalized=True, display=False, heatmap=None, sigma=None, draw_past=False):
    '''
    Draws past, proposed, and ground truth future frames.

    Arguments:
        set_info: [<sequence dir>, <anchor frame filename>, <final frame filename>, <object id>, <object class>]
    '''
    bbs = np.copy(prior_bbs)
    img_file = os.path.join(set_info[0], 'image_02', 'data', set_info[2] + '.png')
    seq_name = os.path.basename(set_info[0])
    print(img_file, seq_name)
    
    # cv2
    # img = cv2.imread(img_file)

    # matplotlib
    img = mpimg.imread(img_file)
    fig,ax = plt.subplots(1)
    ax.imshow(img)

    print(prior_bbs[-1], t_p)
    pred = data_extract_1obj.transform(prior_bbs[-1], t_p)
    gt = data_extract_1obj.transform(prior_bbs[-1], t_gt)
    pred = data_extract_1obj.unnormalize_bb(pred)
    gt = data_extract_1obj.unnormalize_bb(gt)

    # Unnormalize bbs
    if unnormalized:
        bbs[:, 0] = bbs[:, 0] * 1242
        bbs[:, 1] = bbs[:, 1] * 375
        bbs[:, 2] = bbs[:, 2] * 1242
        bbs[:, 3] = bbs[:, 3] * 375

    # Convert to [L, T, W, H]
    for i in range(len(bbs)):
        bbs[i] = data_extract_1obj.center_to_topleft_bb(bbs[i])
    pred = data_extract_1obj.center_to_topleft_bb(pred)
    gt = data_extract_1obj.center_to_topleft_bb(gt)

    # Convert to int
    bbs = bbs.astype(int)
    pred = pred.astype(int)
    gt = gt.astype(int)
    
    a = 1.0 - (.08 * len(bbs))# .2  # .4
    color = (4, 165, 239)
    for i, bb in enumerate(bbs):
        # cv2
        # overlay = img.copy()
        # cv2.rectangle(overlay, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color, 2)
        # cv2.addWeighted(overlay, a, img, 1 - a, 0, img)

        # matplotlib
        if draw_past:
            rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=2, edgecolor=(239/255, 165/255, 4/255), alpha=a, fill=False)
            ax.add_patch(rect)

        a += .08  # .2

    # cv2
    # img = cv2.rectangle(img, (pred[0], pred[1]), (pred[0] + pred[2], pred[1] + pred[3]), (255, 97, 0), 2)  
    # img = cv2.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 255, 0), 2)

    # matplotlib
    true_rect = patches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=2, edgecolor='w', fill=False)
    ax.add_patch(true_rect)
    gen_rect = patches.Rectangle((pred[0], pred[1]), pred[2], pred[3], linestyle='--', linewidth=2, edgecolor='r', fill=False)  #(0/255, 72/255, 255/255)
    ax.add_patch(gen_rect)

    if heatmap is not None:
        # print("HEATMAP", heatmap.shape)
        # print("IMG", img.shape)

        # cv2
        # heatmap = cv2.convertScaleAbs(heatmap)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        # cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # cv2.addWeighted(heatmap, 0.6, img, 0.4, 0, img)

        # matplotlib
        viridis = cm.get_cmap('viridis', 256)
        # gist_heat = cm.get_cmap('gist_heat', 256)

        # heatmap = np.where(heatmap != 0, np.log2(heatmap), 0)
        # ax.imshow(heatmap, cmap=viridis, alpha=0.75)
        # plt.show()

        heatmap = np.sqrt(heatmap)
        ax.imshow(heatmap, cmap=viridis, alpha=0.75)

    plt.axis('off')
    # ax.set_title("sigma_x: {:0.3f}, sigma_y: {:0.3f}, sigma_w: {:0.3f}, sigma_h: {:0.3f}".format(sigma[0], sigma[1], sigma[2], sigma[3]), fontsize=8)
    plt.tight_layout()
    # plt.show()


    if display:
        cv2.imshow('ImageWindow', img)
        cv2.waitKey()
    else:
        # cv2
        # cv2.imwrite(os.path.join(output_dir, seq_name+'_'+set_info[2]+'_'+set_info[3]+'.png'), img)

        # matplotlib
        output_file = os.path.join(output_dir, seq_name+'_'+set_info[2]+'_'+set_info[3]+'_noheat')
        # plt.savefig(output_file + '.png', format='png')
        # plt.savefig(output_file + '.eps', format='eps', dpi=1000)
        plt.savefig(output_file + '.pdf', format='pdf', dpi=1000)
    return

def draw_heatmap(anchor, transforms):
    '''
    transforms = [1000 x 4]
    '''
    print("ANCHOR", anchor.shape)
    # heatmap_overlay = np.zeros((375, 1242, 1), dtype=np.uint16)
    heatmap_overlay = np.zeros((375, 1242), dtype=np.uint16)
    for t in transforms:
        box = data_extract_1obj.transform(anchor, t)
        box = data_extract_1obj.unnormalize_bb(box)
        topleft_box = data_extract_1obj.center_to_topleft_bb(box)
        topleft_box = topleft_box.astype(int)
        if topleft_box[0] > 1242 or topleft_box[0] < 0 or topleft_box[1] > 375 or topleft_box[0] < 0:
            print("BOX:", box)

        # heatmap_overlay[topleft_box[1]:topleft_box[1]+topleft_box[3], topleft_box[0]:topleft_box[0]+topleft_box[2], 0] += 1   # cv2
        l = np.clip(topleft_box[1], 0, 375)
        t = np.clip(topleft_box[0], 0, 1242)
        r = np.clip(topleft_box[1]+topleft_box[3], 0, 375)
        b = np.clip(topleft_box[0]+topleft_box[2], 0, 1242)
        heatmap_overlay[l:r, t:b] += 1        # matplotlib

    # heatmap_overlay.clip(0, 255)
    # viridis = cm.get_cmap('viridis', 256)
    # quadmesh = plt.pcolormesh(heatmap_overlay, cmap=viridis, rasterized=True, vmin=-4, vmax=4)
    # plt.show()

    # heatmap_overlay = cv2.convertScaleAbs(heatmap_overlay)
    # print("HEATMAP1D", heatmap_overlay.shape, np.max(heatmap_overlay), np.sum(heatmap_overlay))
    # heatmap_overlay = cv2.cvtColor(heatmap_overlay, cv2.COLOR_GRAY2RGB)
    # print("HEATMAP", heatmap_overlay.shape, np.max(heatmap_overlay), np.sum(heatmap_overlay))
    # cv2.applyColorMap(heatmap_overlay, cv2.COLORMAP_JET)

    # return heatmap_overlay
    return heatmap_overlay

class Rect:
    def __init__(self, l, t, r, b):
        #assert (l <= r), "l not greater than r"
        #assert (t <= b), "t not greater than b"

        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.area = (r - l) * (b - t)
        if l > r or t > b:
            self.area = 0

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
        intersect = Rect.get_intersection(rect1, rect2)
        if intersect:
            iou = intersect.area / (rect1.area + rect2.area - intersect.area)
            assert (iou > 0), "Non-positive IoU!"
            return iou
        return 0

    @staticmethod
    def get_DE(rect1, rect2):
        c1 = np.array([(rect1.l + rect1.r)/2, (rect1.t + rect1.b)/2])
        c2 = np.array([(rect2.l + rect2.r)/2, (rect2.t + rect2.b)/2])
        return np.linalg.norm(c1 - c2)


def get_IoU(anchor, target_transform, generated_transform, sample_set=None, dataset='kitti_tracking'):
    if dataset == 'kitti_raw_tracklets':
        sample_set = None

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

    return iou, de

def calc_metrics_train(anchor, target_transforms, generated_transforms, sample_set=None):
    """ Calculates displacement error and IoU metrics for 0.5 and 1.0 sec predictions"""

    ious = np.empty(2)
    des = np.empty(2)
    for i, j in enumerate([4, 9]):
        t_bb = data_extract_1obj.transform(anchor, target_transforms[:, j, 0])
        g_bb = data_extract_1obj.transform(anchor, generated_transforms[:, j, 0])
        #t_bb = data_extract_1obj.transform_offset(anchor, target_transforms[:, j, 0])
        #g_bb = data_extract_1obj.transform_offset(anchor, generated_transforms[:, j, 0])
        t_bb = data_extract_1obj.unnormalize_bb(t_bb, sample_set=sample_set)
        g_bb = data_extract_1obj.unnormalize_bb(g_bb, sample_set=sample_set)

        target = Rect.make_cXcYWH(t_bb[0], t_bb[1], t_bb[2], t_bb[3])
        generated = Rect.make_cXcYWH(g_bb[0], g_bb[1], g_bb[2], g_bb[3])

        ious[i] = Rect.get_IoU(target, generated)
        des[i] = Rect.get_DE(target, generated)

    return ious, des

def calc_metrics_all(anchor, target_transforms, generated_transforms, sample_set=None, offset_t=False):
    """ Calculates displacement error and IoU metrics for 0.5 and 1.0 sec predictions"""
    ious = np.empty(10)
    des = np.empty(10)
    for i in range(10):
        if offset_t:
            t_bb = data_extract_1obj.transform_offset(anchor, target_transforms[:, i, 0])
            g_bb = data_extract_1obj.transform_offset(anchor, generated_transforms[:, i, 0])
        else:        
            t_bb = data_extract_1obj.transform(anchor, target_transforms[:, i, 0])
            g_bb = data_extract_1obj.transform(anchor, generated_transforms[:, i, 0])
        t_bb = data_extract_1obj.unnormalize_bb(t_bb, sample_set=sample_set)
        g_bb = data_extract_1obj.unnormalize_bb(g_bb, sample_set=sample_set)

        target = Rect.make_cXcYWH(t_bb[0], t_bb[1], t_bb[2], t_bb[3])
        generated = Rect.make_cXcYWH(g_bb[0], g_bb[1], g_bb[2], g_bb[3])

        ious[i] = Rect.get_IoU(target, generated)
        des[i] = Rect.get_DE(target, generated)


    return ious, des


def calc_metrics_polynomial(anchor, target_transforms, coeffs, sample_set=None):
    """ Calculates displacement error and IoU metrics for 0.5 and 1.0 sec predictions"""
    # coeffs = np.reshape(coeffs, (4, 3))

    ious = np.empty(2)
    des = np.empty(2)
    for i, timestep in enumerate(np.linspace(0.5, 1, 2)):
        # generated_transform = np.dot(coeffs, np.array([timestep, timestep**2, timestep**3, timestep**4, timestep**5]))
        generated_transform = np.dot(coeffs, np.array([timestep, timestep**2, timestep**3, timestep**4]))
        # generated_transform = np.dot(coeffs, np.array([timestep, timestep**2, timestep**3]))
        # generated_transform = np.dot(coeffs, np.array([timestep, timestep**2]))
        # print('generated_transform:', generated_transform)
        t_bb = data_extract_1obj.transform(anchor, target_transforms[:, (i*5)+4])
        g_bb = data_extract_1obj.transform(anchor, generated_transform)
        t_bb = data_extract_1obj.unnormalize_bb(t_bb, sample_set=sample_set)
        g_bb = data_extract_1obj.unnormalize_bb(g_bb, sample_set=sample_set)

        target = Rect.make_cXcYWH(t_bb[0], t_bb[1], t_bb[2], t_bb[3])
        generated = Rect.make_cXcYWH(g_bb[0], g_bb[1], g_bb[2], g_bb[3])

        ious[i] = Rect.get_IoU(target, generated)
        des[i] = Rect.get_DE(target, generated)

    return ious, des


if __name__ == '__main__':

    # samples, samples_info = data_extract_1obj.get_kitti_data(sets=None)
    # samples, samples_info = data_extract_1obj.get_kitti_raw_data()
    xs, ys, samples_info = data_extract_1obj.get_kitti_raw_tracklets(np.linspace(0.1, 1.0, 10))

    print(xs.shape)
    # print(len(ys))

    x = xs[0]
    x_info = samples_info[0]
    print(x)
    print(x_info)

    # drawFrameRects(x_info[0], x_info[1], x_info[2], x, False, None, dataset='kitti_raw', display=True)
    # drawFrameRects(x_info[0], x_info[1], x_info[2], x, False, None, dataset='kitti_raw_tracklets', display=True)
    # drawFrameRects(x_info[0], x_info[1], x_info[2], x, True, None, dataset='kitti_raw_tracklets', display=True)

    # View multiple
    # for i in range(5):
    #     x = samples[i]
    #     x_info = samples_info[i]

    #     print(x)
    #     print(x_info)

    #     drawFrameRects(x_info[0], x_info[1], x_info[2], x, False, None, dataset='kitti_raw_tracklets', display=True)
