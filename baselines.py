import numpy as np
from data_extract_1obj import get_kitti_raw_tracklets, transform_offset
from vis_tool import calc_metrics_all


def get_stag_preds(x):
    '''
    keeps box at anchor for prediction (stagnant future)
    Returns: set of predictions at each timepoint, for each sample. predictions are normalized transformations.
    '''
    return np.zeros([len(x), 4, 10, 1])  # dims: (sample, bb_dim, timepoint)

def get_linear_preds(x):
    '''
    makes linear predictions based on the previous two frames
    Returns: set of predictions at each timepoint, for each sample. predictions are normalized transformations.
    '''
    n = len(x)    

    lin_preds = np.empty([n, 4, 10, 1])  # dims: (sample, bb_dim, timepoint)
    for i in range(n):
        for j in range(4):
            for k in range(10):
                #print(lin_preds[i][j][k])
                #print(x_val[i][1][j])

                # get linear interpolation of the offset transformation
                lin_preds[i][j][k][0] = (k+1)*(x[i][1][j] - x[i][0][j]) #+ x_val[i][1][j]
    
    return lin_preds

def threshold_iou(x, y_pred, y_true, t):
    '''
    returns indices for which the iou is above a threshold (t) at the +1.0s timepoint
    Returns: set of indices
    '''

    ious = np.empty((len(x), 10))
    des = np.empty((len(x), 10))
    for i in range(len(x)):
        ious[i], des[i] = calc_metrics_all(x[i][1], y_true[i], y_pred[i], sample_set=None, offset_t=True)

    indices = np.reshape(np.where(ious[:, 9] > t), -1)
    #print(ious[indices, 9], np.shape(indices))
    return indices

def get_difficulty_ids(x, y):
    '''returns 3 set of ids (easy / med / hard) corresponding to samples in x'''
    lin_preds = get_linear_preds(x)
    stag_preds = get_stag_preds(x)

    easy_ids = threshold_iou(x, stag_preds, y, .5)
    med_ids = np.setdiff1d(threshold_iou(x, lin_preds, y, .5), easy_ids)
    hard_ids = np.setdiff1d(np.setdiff1d(np.arange(len(x)), med_ids), easy_ids)

    print(len(easy_ids) / len(x))
    print(len(med_ids) / len(x))
    print(len(hard_ids) / len(x))
    print((len(easy_ids)+len(med_ids)+len(hard_ids)) / len(x))
    
    return easy_ids, med_ids, hard_ids

def stats_per_difficulty(x, y_true, y_pred, d1, d2, d3, offset_t=False):

    ious_d1 = np.empty((len(d1), 10))
    ious_d2 = np.empty((len(d2), 10))
    ious_d3 = np.empty((len(d3), 10))
    e, m, h = (0, 0, 0)
    for i in range(len(x)):
        if i in d1:
            ious_d1[e], _ = calc_metrics_all(x[i][-1], y_true[i], y_pred[i], sample_set=None, offset_t=offset_t)
            e += 1
        elif i in d2:
            ious_d2[m], _ = calc_metrics_all(x[i][-1], y_true[i], y_pred[i], sample_set=None, offset_t=offset_t)
            m += 1
        else:
            ious_d3[h], _ = calc_metrics_all(x[i][-1], y_true[i], y_pred[i], sample_set=None, offset_t=offset_t)
            h += 1
    return ious_d1, ious_d2, ious_d3

def print_baseline_difficulty_stats(x, y):
    ''' TODO: use stats_per_difficulty '''
    lin_preds = get_linear_preds(x)
    stag_preds = get_stag_preds(x)
    #print('x_val (sample 1):', x_val[0])
    #print('y_val (sample 1, +1 frame):', y_val[0, :, 0])    
    #print('true 1st frame:', transform_offset(x_val[0][1], y_val[0, :, 0]))
    #print('pred 1st frame:', lin_preds[0, :, 0])
    #print('expected pred :', (x_val[0, 1] - x_val[0, 0]))
    #print('all frames:', lin_preds[0, :, :])

    easy_ids, med_ids, hard_ids = get_difficulty_ids(x, y)

    lin_ious_e = np.empty((len(easy_ids), 10))
    lin_ious_m = np.empty((len(med_ids), 10))
    lin_ious_h = np.empty((len(hard_ids), 10))
    stag_ious_e = np.empty((len(easy_ids), 10))
    stag_ious_m = np.empty((len(med_ids), 10))
    stag_ious_h = np.empty((len(hard_ids), 10))
    e, m, h = (0, 0, 0)
    for i in range(len(x)):
        if i in easy_ids:
            lin_ious_e[e], _ = calc_metrics_all(x[i][1], y[i], lin_preds[i], sample_set=None, offset_t=True)
            stag_ious_e[e], _ = calc_metrics_all(x[i][1], y[i], stag_preds[i], sample_set=None, offset_t=True)
            e += 1
        elif i in med_ids:
            lin_ious_m[m], _ = calc_metrics_all(x[i][1], y[i], lin_preds[i], sample_set=None, offset_t=True)
            stag_ious_m[m], _ = calc_metrics_all(x[i][1], y[i], stag_preds[i], sample_set=None, offset_t=True)
            m += 1
        else:
            lin_ious_h[h], _ = calc_metrics_all(x[i][1], y[i], lin_preds[i], sample_set=None, offset_t=True)
            stag_ious_h[h], _ = calc_metrics_all(x[i][1], y[i], stag_preds[i], sample_set=None, offset_t=True)
            h += 1

    print('stagnant: ', np.mean(stag_ious_e[:, 9]), np.mean(stag_ious_m[:, 9]), np.mean(stag_ious_h[:, 9]))
    print('linear  : ', np.mean(lin_ious_e[:, 9]), np.mean(lin_ious_m[:, 9]), np.mean(lin_ious_h[:, 9]))

def get_baseline_data():

    # get test sets
    timepoints = np.linspace(0.1, 1.0, 10)
    np.random.seed(11)
    all_sets = np.arange(38)
    test_sets = np.random.choice(all_sets, (3,10), replace=False)

    # get test data (ie X: the normalized bbs and Y: the offset transformations to the 10 future frames)
    x_val, y_val, val_info = get_kitti_raw_tracklets(timepoints, sets=test_sets[0], class_types=['Car', 'Van', 'Truck'], past_frames=2, offset_t=True)
    y_val = np.reshape(y_val, (len(y_val), 4, 10, 1))    
    
    #print(np.shape(x_val))
    #print(np.shape(y_val))
    return x_val, y_val


if __name__ == '__main__':

    x_val, y_val = get_baseline_data()
    print_baseline_difficulty_stats(x_val, y_val)
    #easy_ids, med_ids, hard_ids = get_difficulty_ids(x_val, y_val)

    '''
    ious = np.empty((len(x_val), 10))
    des = np.empty((len(x_val), 10))
    # get IoU
    for i in range(len(x_val)):
        ious[i], des[i] = calc_metrics_all(x_val[i][1], y_val[i], lin_preds[i], sample_set=None, offset_t=True)
    
    print('IoUs:', np.mean(ious, axis=0))
    print('DEs :', np.mean(des, axis=0))
    print('ADE :', np.mean(des))
    '''


