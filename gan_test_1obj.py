#import gan_1obj
# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import poly_model
import data_extract_1obj
import numpy as np
import os
import scipy

import matplotlib.pyplot as plt
from vis_tool import calc_metrics_all
import vis_tool

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

def get_metrics(output_dir, M, x, y, set_info, past_frames):
    x_in = x.reshape((-1, past_frames*4))
    y = y.reshape((-1, 4, 10, 1))
    num_samples = x.shape[0]
    out = M.predict(x_in)
    gen_transforms = out[0]
    print("len(out):", len(out))
    print(out[0].shape)
    print(out[1].shape)

    ious = np.empty((num_samples, 10))
    des = np.empty((num_samples, 10))
    for i in range(x.shape[0]):
        ious[i], des[i] = calc_metrics_all(x_in[i][-4:], y[i], gen_transforms[i])

    print("ADE:", np.mean(des))
    print("FDE:", np.mean(des[:, 9]))
    print("FIOU:", np.mean(ious[:, 9]))  

    #print(np.linspace(0.1,1.0,10).shape, np.mean(ious, axis=0).shape)
    # fig1, ax1 = plt.subplots()    
    # ax1.plot(np.linspace(0.1, 1.0, 10), np.mean(ious, axis=0))
    # ax1.set_ylabel('mean IoU')
    # ax1.set_xlabel('future timestep (seconds)')
    # ax1.set_title('mean IoU over time')
    # fig1.savefig(os.path.join(output_dir, 'mean_ious.png'))

    #timestep_hist(output_dir, ious[:, 9])
    #sigma_iou_scatter(output_dir, gen_transforms[:, :, 9, 1], ious[:, 9])
    # tx_ty_scatter(output_dir, gen_transforms)
    show_failures(output_dir, ious[:, 9], gen_transforms[:, :, :, 1], gen_transforms[:, :, :, 0], x, y, set_info)
    #show_success(output_dir, ious[:, 9], gen_transforms[:, :, :, 1], gen_transforms[:, :, :, 0], x, y, set_info)
    #transf_hist(output_dir, gen_transforms[:, 0, 9, 0], y[:, 0, 9, 0])

def show_failures(output_dir, ious, sigmas, transforms, x, y, set_info):
    start = 2500
    stop = 3300
    # output_dir = os.path.join(output_dir, 'failure_cases_mplt', 'range{}-{}_.1-.5'.format(start, stop))
    output_dir = os.path.join(output_dir, 'failure_cases_mplt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ids = data_extract_1obj.get_batch_ids(x.shape[0], x.shape[0])
    count = 0
    resultsFile = open(os.path.join(output_dir, 'results.txt'), 'w')
    # for i in ids[start:stop]:
    for i in range(len(ids)):
        if count == 20:
            break
        if ious[i] > 0.1 and ious[i] < 0.5 and set_info[i][2] == '0000000151' and set_info[i][3] == '16':
        # if ious[i] < 0.1:
        # if set_info[i][2] == '0000000213' and set_info[i][3] == '55':
            heatmap_overlay = create_heatmap(sigmas[i,:,9], transforms[i,:,9], x[i][-1])
            # vis_tool.draw_p_and_gt(set_info[i], x[i], transforms[i, :, 9], y[i, :, 9], output_dir, heatmap=heatmap_overlay, sigma=sigmas[i,:,9], draw_past=True)
            vis_tool.draw_p_and_gt(set_info[i], x[i], transforms[i, :, 9], y[i, :, 9], output_dir, heatmap=None, sigma=sigmas[i,:,9], draw_past=True)

            print("seq:", os.path.basename(set_info[i][0]), file=resultsFile)
            print("target frame / obj:", set_info[i][2], "/", set_info[i][3], file=resultsFile)
            print("  mean sigma:", np.mean(sigmas[i, :, 9]), file=resultsFile)
            print("  sigmas:", sigmas[i, :, 9], "\n", file=resultsFile)
            count +=1

def show_success(output_dir, ious, sigmas, transforms, x, y, set_info):
    start = 0
    stop = 500
    output_dir = os.path.join(output_dir, 'success_cases_mplt')
    # output_dir = os.path.join(output_dir, 'success_cases_mplt', 'range{}-{}'.format(start, stop))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ids = data_extract_1obj.get_batch_ids(x.shape[0], x.shape[0])
    count = 0
    resultsFile = open(os.path.join(output_dir, 'results.txt'), 'w')
    # for i in ids[start:stop]:
    for i in range(len(ids)):
        if count == 20:
            break
        if ious[i] > 0.9:
            heatmap_overlay = create_heatmap(sigmas[i,:,9], transforms[i,:,9], x[i][-1])
            vis_tool.draw_p_and_gt(set_info[i], x[i], transforms[i, :, 9], y[i, :, 9], output_dir, heatmap=heatmap_overlay, sigma=sigmas[i,:,9], draw_past=True)

            print("seq:", os.path.basename(set_info[i][0]), file=resultsFile)
            print("target frame / obj:", set_info[i][2], "/", set_info[i][3], file=resultsFile)
            print("  mean sigma:", np.mean(sigmas[i, :, 9]), file=resultsFile)
            print("  sigmas:", sigmas[i, :, 9], "\n", file=resultsFile)
            count +=1

def create_heatmap(sigma, mean_t, anchor):
    ts = np.empty((1000, 4))
    for i in range(4):
        ts[:, i] = sample_transfs(mean_t[i], sigma[i], 1000)
    heatmap_overlay = vis_tool.draw_heatmap(anchor, ts)
    return heatmap_overlay

def sample_transfs(mean, sigma, num_samples):
    xs = np.linspace(-5*sigma, 5*sigma, 1000)


    pdf = np.array([get_p(x, sigma) for x in xs])
    pdf /= np.sum(pdf)
    cdf = np.cumsum(pdf) #/ np.sum(pdf)
    # plt.hist(cdf, bins=100, histtype='step', cumulative=1)
    # plt.plot(xs, pdf)
    # plt.plot(xs, cdf)
    # plt.title('cdf')
    # plt.show()


    # icdf = np.percentile(cdf, range(0, 101))
    # icdf = (np.percentile(cdf, range(0, 101)) - 0.5) * 5*sigma
    icdf = calc_icdf(xs, cdf)
    # plt.plot(np.linspace(0.0, 1.0, 99), icdf)
    # plt.title('icdf')
    # plt.show()

    samples = np.random.uniform(size=num_samples) * 99  #101?
    # print(samples)
    # print(samples.shape)
    # print(icdf.shape)
    x_s = icdf[samples.astype(int)]
    # print(x_s.shape)
    # plt.scatter(x_s, np.zeros_like(x_s))
    # plt.title('x_s')
    # plt.show()
    return x_s + mean

def calc_icdf(x, cdf):
    i = 0
    step = 0.01
    t = step
    icdf = []
    while t <= 1.:
        if cdf[i] <= t and t <= cdf[i+1]:
            icdf.append((x[i] * (cdf[i + 1] - t) + x[i+1] * (t - cdf[i])) / (cdf[i+1] - cdf[i]))
            # icdf.append((x[i] * (t - cdf[i]) + x[i+1] * (cdf[i+1] - t)) / (cdf[i+1] - cdf[i]))
            t += step
        i += 1

    return np.array(icdf)

def get_p(d_x, sigma):
    tau = 1.345*sigma
    c = sigma*np.sqrt(2*np.pi)*scipy.special.erf(tau/(sigma*np.sqrt(2))) + (2*np.square(sigma)/tau)*np.exp(-np.square(tau)/(2*np.square(sigma)))
    p = (1/c)*np.exp(np.where(np.abs(d_x) < tau, gauss(d_x, sigma), laplace(d_x, sigma, tau)))
    return p

def gauss(d_x, sigma):
    return -np.square(d_x)/(2*np.square(sigma))

def laplace(d_x, sigma, tau):
    return (-tau/(sigma*np.sqrt(2)))*np.abs(d_x) + np.square(tau)/(2*np.square(sigma))



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

def transf_hist(output_dir, pred_ts, gt_ts):
    fig2, ax2 = plt.subplots()
    ax2.hist([pred_ts, gt_ts], bins=30, label=['predicted', 'ground truth'])
    ax2.legend()
    ax2.set_xlabel('+1.0s tx')
    ax2.set_ylabel('count')
    ax2.set_title('+1.0s tx distribution')
    fig2.savefig(os.path.join(output_dir, 'tx_hist.png'))

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

def multimodel_timestep_hist(x, y, past_frames, optimizer):

    model_names = [
    'quad-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'cubic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'quartic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'quintic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    'sextic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600',
    ]
    epochs = [300, 300, 300, 300, 300, 300]

    x = x.reshape((-1, past_frames*4))
    y = y.reshape((-1, 4, 10, 1))
    num_samples = x.shape[0]
    all_ious = np.empty((len(model_names), num_samples, 10))

    tau = 1.345
    for i, model_name in enumerate(model_names):
        poly_order = 2 + i
        weights_path = os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/models', model_name, 'weights/m_weights_epoch-{}.h5'.format(epochs[i]))
        M = poly_model.get_model_poly(None, poly_order, np.linspace(0.1, 1.0, 10), tau, past_frames, optimizer=optimizer, weights_path=weights_path)
        out = M.predict(x)
        gen_transforms = out[0]
        print(weights_path)
        print(gen_transforms.shape)
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
    for i in range(5):    
        ax2.plot(x[i], y[i], label=i+2)
    ax2.legend()
    ax2.set_xlabel('+1.0s IoU threshold')
    ax2.set_ylabel('# above')
    ax2.set_title('degree comparison via +1.0s IoU distributions ')
    fig2.savefig(os.path.join(output_dir, 'poly_degree_threshold.png'))

    for i in range(len(y)):
        print(i+2, "AUC:", np.sum(y[i]) / 100)

if __name__ == '__main__':
    ## TESTING ##
    data_cols = []
    for fNum in range(1, 12):
        for char in ['L', 'T', 'W', 'H']:
            data_cols.append(char + str(fNum))

    # LOAD MODEL
    # model_name = 'quartic_sigma-2coeff-abs_red-sum_huber_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_g3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
    model_name = 'poly6_past10_t1.345xsig_seed11-0_vehicles-nobike_G3-64_adam-lr0.0005-b10.9-b20.999_bs128_epochs1100'
    #model_name = 'quartic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
    #model_name = 'sextic-test_t1.345xsig_seed-11-test0f_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
    # model_name = 'd5_past5-test_t1.345xsig_seed-11-test0_vehicles-nobike_7-fold_G3-64_adam-lr0.00146-b10.9-b20.999_bs512_epochs600'
    # epoch = '511'
    epoch = '1100'
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
    past_frames = 10
    np.random.seed(11)
    all_sets = np.arange(38)
    test_sets = np.random.choice(all_sets, (3,10), replace=False)
    x_test, y_test, set_info = data_extract_1obj.get_kitti_raw_tracklets(np.linspace(0.1, 1.0, 10), sets=test_sets[0], class_types=['Car', 'Van', 'Truck'], past_frames=past_frames)
    
    # multimodel_timestep_hist(x_test, y_test, past_frames, optimizer=optimizer)

    # output_dir = os.path.join('/playpen/mhudnell_cvpr_2019/mhudnell/maxgan/models', model_name)
    output_dir = os.path.join('C:\\Users\\Max\\Research\\maxGAN\\models', model_name)
    weights_path = os.path.join(output_dir, 'weights', 'm_weights_epoch-{}.h5'.format(epoch))
    poly_order = 6
    tau = 1.345
    M = poly_model.get_model_poly(None, poly_order, np.linspace(0.1, 1.0, 10), tau, past_frames, optimizer=optimizer,
    weights_path=weights_path)
    get_metrics(output_dir, M, x_test, y_test, set_info, past_frames) 


    # Deprecated
    # step = '13700'
    # generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\gen_weights_step_' + step + '.h5'
    # discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\discrim_weights_step_' + step + '.h5'

    # gan_1obj.test_model(generator_model, discriminator_model, combined_model, model_name)
    # gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name)
    # gan_1obj.test_model_IOU(generator_model, discriminator_model, combined_model, model_name)
