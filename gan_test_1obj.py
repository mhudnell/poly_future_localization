#import gan_1obj
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import poly_model
import data_extract_1obj
import numpy as np
import os

import matplotlib.pyplot as plt
from vis_tool import calc_metrics_all

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

def get_metrics(output_dir, M, x, y, set_info):
    x = x.reshape((-1, 40))
    y = y.reshape((-1, 4, 10, 1))
    num_samples = x.shape[0]
    out = M.predict(x)
    gen_transforms = out[0]
    print("len(out):", len(out))
    print(out[0].shape)
    print(out[1].shape)

    ious = np.empty((num_samples, 10))
    des = np.empty((num_samples, 10))
    for i in range(x.shape[0]):
        ious[i], des[i] = calc_metrics_all(x[i][-4:], y[i], gen_transforms[i])

    print("ADE:", np.mean(des))
    print("FDE:", np.mean(des[:, 9]))
    print("FIOU:", np.mean(ious[:, 9]))  

    #print(np.linspace(0.1,1.0,10).shape, np.mean(ious, axis=0).shape)
    fig1, ax1 = plt.subplots()    
    ax1.plot(np.linspace(0.1, 1.0, 10), np.mean(ious, axis=0))
    ax1.set_ylabel('mean IoU')
    ax1.set_xlabel('future timestep (seconds)')
    ax1.set_title('mean IoU over time')
    fig1.savefig(os.path.join(output_dir, 'mean_ious.png'))

    #timestep_hist(output_dir, ious[:, 9])
    #sigma_iou_scatter(output_dir, gen_transforms[:, :, 9, 1], ious[:, 9])
    tx_ty_scatter(output_dir, gen_transforms)

def show_failures(output_dir, sigmas, transforms, set_info):
    ...

def tx_ty_scatter(output_dir, transforms):
    fig3, ax3 = plt.subplots()
    ax3.scatter(transforms[:, 0, 9, 0], transforms[:, 1, 9, 0])
    ax3.set_xlabel('tx')
    ax3.set_ylabel('ty')
    ax3.set_title('tx v ty')
    fig3.savefig(os.path.join(output_dir, 'tx_ty_scatter.png'))

def timestep_hist(output_dir, ious):
    fig2, ax2 = plt.subplots()
    ax2.hist(ious, bins=20)
    ax2.set_xlabel('+1.0s IoU')
    ax2.set_ylabel('count')
    ax2.set_title('+1.0s IoU distribution')
    fig2.savefig(os.path.join(output_dir, 'timestep_hist.png'))

def sigma_iou_scatter(output_dir, sigmas, ious):
    """sigma v iou scatter plot for 1 timestep"""
    print(sigmas.shape)
    print(ious.shape)
    fig3, ax3 = plt.subplots()
    ax3.scatter(np.mean(sigmas, axis=1), ious)
    ax3.set_xlabel('sigma (uncertainty)')
    ax3.set_ylabel('IoU')
    ax3.set_title('sigma v. +1.0s Iou')
    fig3.savefig(os.path.join(output_dir, 'sigma_iou_scatter.png'))

if __name__ == '__main__':
    ## TESTING ##
    data_cols = []
    for fNum in range(1, 12):
        for char in ['L', 'T', 'W', 'H']:
            data_cols.append(char + str(fNum))

    # LOAD MODEL
    model_name = 'quartic_sigma-2coeff-abs_red-sum_huber_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_g3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
    epoch = '511'
#    generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\gen_weights_epoch-' + epoch + '.h5'
#    discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\discrim_weights_epoch-' + epoch + '.h5'
    optimizer = {
                'name': 'adam',
                'lr': .00146,        # default: .001
                'beta_1': .9,       # default: .9
                'beta_2': .999,     # default: .999
                'decay': 0       # default: 0
                }

    #generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, generator_model_path=generator_model_path,
    #    discriminator_model_path=discriminator_model_path, optimizer=optimizer, w_adv=1.0)

    #run_tests(generator_model, discriminator_model, combined_model, model_name, seed=10, dataset='kitti_raw_tracklets')

    np.random.seed(11)
    all_sets = np.arange(38)
    test_sets = np.random.choice(all_sets, (3,10), replace=False)
    x_test, y_test, set_info = data_extract_1obj.get_kitti_raw_tracklets(np.linspace(0.1, 1.0, 10), sets=test_sets[0], class_types=['Car', 'Van', 'Truck'])
    output_dir = os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/models', model_name)
    weights_path = os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/models', model_name, 'weights/m_weights_epoch-{}.h5'.format(epoch))
    print(weights_path)
    poly_order = 4
    tau = 1.345
    M = poly_model.get_model_poly(None, poly_order, np.linspace(0.1, 1.0, 10), tau, optimizer=optimizer,
	 weights_path=weights_path)

    get_metrics(output_dir, M, x_test, y_test, set_info) 

    # Deprecated
    # step = '13700'
    # generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\gen_weights_step_' + step + '.h5'
    # discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\discrim_weights_step_' + step + '.h5'

    # gan_1obj.test_model(generator_model, discriminator_model, combined_model, model_name)
    # gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name)
    # gan_1obj.test_model_IOU(generator_model, discriminator_model, combined_model, model_name)
