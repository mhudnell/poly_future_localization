#import gan_2obj
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import poly_model
import data_extract_1obj
import numpy as np
import os

import matplotlib.pyplot as plt
from vis_tool import calc_metrics_all

#MODEL_DIR = '/playpen/mhudnell_cvpr_2019/jtprice/L1-loss/models_poly_6'
MODEL_DIR = '/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/models'
#MODEL_DIR = '/playpen/mhudnell_cvpr_2019/mhudnell/poly_future_localization/models'

#MODEL_NAME = 'scaleinv-poly6_past10_t1.345xsig_seed11-0_vehicles-nobike_G3-64_adam-lr0.0005-b10.9-b20.999_bs128_epochs1100'
#MODEL_NAME = 'poly6_past10_t1.345xsig_seed11-0_vehicles-nobike_G3-64_adam-lr0.0005-b10.9-b20.999_bs128_epochs1100'
#MODEL_NAME = 'rnn_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
MODEL_NAME = 'poly6_past10_t1.345xsig_seed11-0_vehicles-nobike_G3-64_adam-lr0.0005-b10.9-b20.999_bs128_epochs1100'
EPOCH = '1100'
POLY_ORDER = 6
PAST_FRAMES = 10
TAU = 1.345
DATA_SEED = 6
OUTPUT_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
#OUTPUT_DIR = './L1figs'
OFFSET_T = False

def run_tests(generator_model, discriminator_model, combined_model, model_name, seed=6, dataset='kitti_tracking'):

    np.random.seed(seed)
    if dataset == 'kitti_tracking':
        all_sets = np.arange(21)
        test_sets = np.random.choice(all_sets, 3, replace=False)
        # train_sets = np.setdiff1d(all_sets, test_sets)
        test_data, test_data_info = data_extract_1obj.get_kitti_data(test_sets)
    elif dataset == 'kitti_raw_tracklets':
        all_sets = np.arange(38)
        test_sets = np.random.choice(all_sets, 8, replace=False)
        test_data, test_data_info = data_extract_1obj.get_kitti_raw_tracklets(sets=test_sets)
        # train_sets = np.setdiff1d(all_sets, test_sets)
    else:
        raise Exception("`dataset` parameter must be one of: ['kitti_tracking', 'kitti_raw_tracklets']")

    gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name, test_data, test_data_info, dataset)

def get_metrics(M, x, y):
    x = x.reshape((-1, PAST_FRAMES*4))
    y = y.reshape((-1, 4, 10, 1))
    num_samples = x.shape[0]

    out = M.predict(x)
    gen_transforms = out[0]

    ious = np.empty((num_samples, 10))
    des = np.empty((num_samples, 10))
    for i in range(x.shape[0]):
        ious[i], des[i] = calc_metrics_all(x[i][-4:], y[i], gen_transforms[i], offset_t=OFFSET_T)


    print("Metrics for model at epoch", EPOCH)
    print("ADE:", np.mean(des))
    print("+0.5sec DE:", np.mean(des[:, 4]))
    print("+1.0sec DE:", np.mean(des[:, 9]))
    print("+0.5sec IoU:", np.mean(ious[:, 4]))
    print("+1.0sec IoU:", np.mean(ious[:, 9]))

    metrics_file = open(os.path.join(OUTPUT_DIR, 'metrics_{}.txt'.format(EPOCH)), 'w') 
    print("Metrics for model at epoch", EPOCH, file=metrics_file)
    print("ADE:", np.mean(des), file=metrics_file)
    print("+0.5sec DE:", np.mean(des[:, 4]), file=metrics_file)
    print("+1.0sec DE:", np.mean(des[:, 9]), file=metrics_file)
    print("+0.5sec IoU:", np.mean(ious[:, 4]), file=metrics_file)
    print("+1.0sec IoU:", np.mean(ious[:, 9]), file=metrics_file)

    iou_over_time(ious)
    timestep_hist(ious[:, 9])
    sigma_iou_scatter(gen_transforms[:, :, 9, 1], ious[:, 9])
    tx_ty_scatter(gen_transforms)
    transf_hist(gen_transforms[:, :, 9, 0], y[:, :, 9, 0])

def iou_over_time(ious):
    fig1, ax1 = plt.subplots()    
    ax1.plot(np.linspace(0.1, 1.0, 10), np.mean(ious, axis=0))
    ax1.set_ylabel('mean IoU')
    ax1.set_xlabel('future timestep (seconds)')
    ax1.set_title('mean IoU over time')
    fig1.savefig(os.path.join(OUTPUT_DIR, 'mean_ious.png'))

def tx_ty_scatter(transforms):
    fig3, ax3 = plt.subplots()
    ax3.scatter(transforms[:, 0, 9, 0], transforms[:, 1, 9, 0])
    ax3.set_xlabel('tx')
    ax3.set_ylabel('ty')
    ax3.set_title('tx v ty')
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'tx_ty_scatter.png'))

def timestep_hist(ious):
    np.savez(
         #os.path.join(MODEL_DIR, MODEL_NAME, 'saved_results'),
         os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/', 'hist_data'),  #'past-{}'.format(PAST_FRAMES)),
         data=ious,
         )


    fig2, ax2 = plt.subplots(1, 1, figsize=(6,5))
    ax2.hist(ious, bins=20, color='r')
    ax2.set_ylim(0, 400)
    ax2.set_xlabel('+1.0s IoU')
    ax2.set_ylabel('count')
    ax2.set_title('+1.0s IoU distribution')
    #fig2.savefig(os.path.join(OUTPUT_DIR, 'timestep_hist.png'))
    fig2.show()

def transf_hist(pred_ts, gt_ts):
    dims = ['x', 'y', 'w', 'h']
    for i, dim in enumerate(dims):
        fig2, ax2 = plt.subplots()
        ax2.hist([pred_ts[:, i], gt_ts[:, i]], bins=30, label=['predicted', 'ground truth'])
        ax2.legend()
        ax2.set_xlabel('+1.0s t{}'.format(dim))
        ax2.set_ylabel('count')
        ax2.set_title('+1.0s t{} distribution'.format(dim))
        fig2.savefig(os.path.join(OUTPUT_DIR, 't{}_hist.png'.format(dim)))

def sigma_iou_scatter(sigmas, ious):
    """sigma v iou scatter plot for 1 timestep"""
    print(sigmas.shape)
    print(ious.shape)
    fig3, ax3 = plt.subplots()
    ax3.scatter(np.mean(sigmas, axis=1), ious)
    ax3.set_xlabel('sigma (uncertainty)')
    ax3.set_ylabel('IoU')
    ax3.set_title('sigma v. +1.0s Iou')
    fig3.savefig(os.path.join(OUTPUT_DIR, 'sigma_iou_scatter.png'))

def multimodel_timestep_hist(test_set, optimizer):

    model_names = [
    'quad-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'cubic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'quartic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'quintic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'sextic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    ]
    model_names = [
    'd5_past2-test_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'd5_past3-test_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'd5_past5-test_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'd5_past7-test_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'quintic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
    ]
    epochs = [550,550,551,550,550]
    past_frames = [2,3,5,7,10]
#    x = x.reshape((-1, past_frames*4))
#    y = y.reshape((-1, 4, 10, 1))
    
    x, y, set_info = data_extract_1obj.get_kitti_raw_tracklets(np.linspace(0.1, 1.0, 10), sets=test_set, class_types=['Car', 'Van', 'Truck'], past_frames=10)
    num_samples = y.shape[0]
    all_ious = np.empty((len(model_names), num_samples, 10))

    tau = 1.345
    poly_order = 5
    for i, model_name in enumerate(model_names):
        #poly_order = 2 + i
        x, y, set_info = data_extract_1obj.get_kitti_raw_tracklets(np.linspace(0.1, 1.0, 10), sets=test_set, class_types=['Car', 'Van', 'Truck'], past_frames=past_frames[i])
        x = x.reshape((-1, past_frames[i]*4))
        y = y.reshape((-1, 4, 10, 1))
        weights_path = os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/models', model_name, 'weights/m_weights_epoch-{}.h5'.format(epochs[i]))
        M = poly_model.get_model_poly(None, poly_order, np.linspace(0.1, 1.0, 10), tau, past_frames[i], optimizer=optimizer, weights_path=weights_path)
        out = M.predict(x)
        gen_transforms = out[0]
        print(weights_path)
        print(gen_transforms.shape)
        np.savez(
             os.path.join(OUTPUT_DIR, "past_{}.npz".format(past_frames[i])),
             target=y,
             pred=gen_transforms)
        ious = np.empty((num_samples, 10))
        des = np.empty((num_samples, 10))
        for j in range(num_samples):
            ious[j], des[j] = calc_metrics_all(x[j][-4:], y[j], gen_transforms[j])
        all_ious[i] = ious
            
        print("ADE:", np.mean(des))
        print("FDE:", np.mean(des[:, 9]))
        print("FIOU:", np.mean(ious[:, 9]))  

    mult_iou_threshold(all_ious)
'''
    hist_groups = [all_ious[i, :, 9] for i in range(len(model_names))]
    
    print(len(hist_groups))
    print(hist_groups[0].shape)
    output_dir = os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/figures')
    fig2, ax2 = plt.subplots()
    ax2.hist(hist_groups, bins=20, label=[2,3,4,5,6])
    ax2.legend()
    ax2.set_xlabel('+1.0s IoU')
    ax2.set_ylabel('count')
    ax2.set_title('degree comparison via +1.0s IoU distributions ')
    fig2.savefig(os.path.join(output_dir, 'poly_degree_comparison.png'))
'''

def mult_iou_threshold(all_ious):
    output_dir = os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/figures')
    
    print(np.where(all_ious[0, :, 9] > .5))
    print(len(np.where(all_ious[0, :, 9] > .5)))


    y = [[len(np.where(all_ious[i, :, 9] > t)[0])/len(all_ious[0]) for t in np.linspace(0.01, 1, 100)] for i in range(len(all_ious))]
    x = [np.linspace(0.01, 1, 100) for i in range(len(all_ious))]

    print(np.array(y).shape)
    fig2, ax2 = plt.subplots()
    #ax2.bar(np.linspace(0.05, 1, 20), y, label=[2,3,4,5,6])
    for i, f in enumerate([2,3,5,7,10]):    
        ax2.plot(x[i], y[i], label='{} frames'.format(f))
    ax2.legend()
    ax2.set_xlabel('+1.0s IoU threshold')
    ax2.set_ylabel('% predictions above threshold')
    ax2.set_title('+1.0s IoU distributions')
    fig2.savefig(os.path.join(output_dir, 'd5-poly_past_threshold.png'))

    for i in range(len(y)):
        print(i+2, "AUC:", np.sum(y[i]) / 100)

if __name__ == '__main__':
    optimizer = {
                'name': 'adam',
                'lr': .00146,        # default: .001
                'beta_1': .9,       # default: .9
                'beta_2': .999,     # default: .999
                'decay': 0       # default: 0
                }

    np.random.seed(DATA_SEED)
    all_sets = np.arange(38)
    test_sets = np.random.choice(all_sets, (3,10), replace=False)
    #print(test_sets)

    x_test, y_test, _ = data_extract_1obj.get_kitti_raw_tracklets(np.linspace(0.1, 1.0, 10), sets=test_sets[0], class_types=['Car', 'Van', 'Truck'], past_frames=PAST_FRAMES, offset_t=OFFSET_T)
    
    weights_path = os.path.join(MODEL_DIR, MODEL_NAME, 'weights/m_weights_epoch-{}.h5'.format(EPOCH))
    M = poly_model.get_model_poly(None, POLY_ORDER, np.linspace(0.1, 1.0, 10), TAU, PAST_FRAMES, optimizer=optimizer,
	 weights_path=weights_path)

    get_metrics(M, x_test, y_test) 
    #multimodel_timestep_hist(test_sets[0], optimizer=optimizer)


