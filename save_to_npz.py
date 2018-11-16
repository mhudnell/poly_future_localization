import numpy as np
import os
from vis_tool import calc_metrics_all
from data_extract_1obj import get_kitti_raw_tracklets
from poly_model import get_model_poly

MODEL_DIR = '/playpen/mhudnell_cvpr_2019/jtprice/L2-loss/models'
MODEL_NAME = 'septic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
EPOCH = '511'
POLY_ORDER = 5
PAST_FRAMES = 10
TAU = 1.345

def save_to_npz(M, x, y):
    x = x.reshape((-1, PAST_FRAMES*4))
    y = y.reshape((-1, 4, 10, 1))
    num_samples = x.shape[0]
    out = M.predict(x)
    gen_transforms = out[0]

    ious = np.empty((num_samples, 10))
    des = np.empty((num_samples, 10))
    for i in range(x.shape[0]):
        ious[i], des[i] = calc_metrics_all(x[i][-4:], y[i], gen_transforms[i])

    np.savez(
         os.path.join(MODEL_DIR, MODEL_NAME, 'saved_results'),
         target=y,
         pred=gen_transforms)

    print("ADE:", np.mean(des))
    print("FDE:", np.mean(des[:, 9]))
    print("FIOU:", np.mean(ious[:, 9]))

if __name__ == '__main__':
    optimizer = {
        'name': 'adam',
        'lr': .00146,       # default: .001
        'beta_1': .9,       # default: .9
        'beta_2': .999,     # default: .999
        'decay': 0          # default: 0
        }

    np.random.seed(11)
    all_sets = np.arange(38)
    test_sets = np.random.choice(all_sets, (3, 10), replace=False)

    timesteps = np.linspace(0.1, 1.0, 10)
    x_test, y_test, _ = get_kitti_raw_tracklets(
        timesteps, sets=test_sets[0], class_types=['Car', 'Van', 'Truck'], past_frames=PAST_FRAMES)

    weights_path = os.path.join(
        MODEL_DIR, 'weights', 'm_weights_epoch-{}.h5'.format(EPOCH)
        )

    M = get_model_poly(None, POLY_ORDER, timesteps,
        TAU, PAST_FRAMES, optimizer=optimizer, weights_path=weights_path)

    save_to_npz(M, x_test, y_test)