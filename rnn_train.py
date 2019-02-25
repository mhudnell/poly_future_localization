#import gan_1obj
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import logging
import data_extract_1obj
import rnn_model

OUTPUT_DIR = '/playpen/mhudnell_cvpr_2019/jtprice/maxGAN/models'

def log_hyperparams(model_name=None, output_dir=None, train_sets=None, val_sets=None, epochs=None, batch_size=None, k_d=None, k_g=None, optimizer=None, show=None, dataset=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = open(os.path.join(output_dir, 'hyperparameters.txt'), 'w')
    print('Logging hyperparameters:', file=log_file)
    print('  --model_name: {}'.format(model_name), file=log_file)
    print('  --output_dir: {}'.format(output_dir), file=log_file)
    print('  --dataset: {}'.format(dataset), file=log_file)
    print('  --train_sets: {}'.format(train_sets), file=log_file)
    print('  --val_sets: {}'.format(val_sets), file=log_file)
    print('  --epochs: {}'.format(epochs), file=log_file)
    print('  --batch_size: {}'.format(batch_size), file=log_file)
    print('  --k_d: {}'.format(k_d), file=log_file)
    print('  --k_g: {}'.format(k_g), file=log_file)
    print('  --show: {}'.format(show), file=log_file)

    if optimizer:
        print('  --optimizer: {}'.format(optimizer['name']), file=log_file)
        print('    |-lr: {}'.format(optimizer['lr']), file=log_file)
        print('    |-beta1: {}'.format(optimizer['beta_1']), file=log_file)
        print('    |-beta2: {}'.format(optimizer['beta_2']), file=log_file)
        print('    |-decay: {}'.format(optimizer['decay']), file=log_file)

def plot_loss(model_name, epochs, nb_steps, steps_per_epoch, output_dir, M_losses, val_losses,
              train_ious, val_ious, train_des, val_des, show_adv=True): #, avg_gen_pred, avg_real_pred, 
    # PLOT LOSS
    x_steps = np.arange(1, nb_steps+1) / steps_per_epoch
    x_epochs = np.arange(1, epochs+1)
    fig = plt.figure(figsize=(18, 12), dpi=72)
    ax1 = fig.add_subplot(111)
    # ax1.set_ylim([0.0, 1.0])

    ax1.plot(x_steps, M_losses, label='loss')
    ax1.plot(x_epochs, val_losses, label='val_loss')

    ax1.legend(loc=3)
    fig.suptitle(model_name, fontsize=12)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('loss value', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

    # PLOT IOU
    fig = plt.figure(figsize=(18, 12), dpi=72)
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0.0, 1.0])

    # ax1.plot(x_steps, M_losses, label='loss')
    ax1.plot(x_steps, train_ious[:, 0], label='iou +0.5s')
    ax1.plot(x_steps, train_ious[:, 1], label='iou +1.0s')
    # ax1.plot(x_steps, train_des, label='DE')

    # x_epochs = np.arange(1, epochs+1)
    # ax1.plot(x_epochs, val_losses, label='val_loss')
    ax1.plot(x_epochs, val_ious[:, 0], label='val_iou +0.5s')
    ax1.plot(x_epochs, val_ious[:, 1], label='val_iou +1.0s')
    # ax1.plot(x_epochs, val_des, label='val_DE')

    ax1.legend(loc=3)
    fig.suptitle(model_name, fontsize=12)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('loss value', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_plot.png'))
    # plt.show()

    # PLOT DISCRIM PREDICTIONS
    # fig = plt.figure(figsize=(18, 12), dpi=72)
    # ax1 = fig.add_subplot(111)

    # ax1.plot(x_epochs, avg_gen_pred, label='avg_gen_pred')
    # ax1.plot(x_epochs, avg_real_pred, label='avg_real_pred')

    # ax1.legend(loc=1)
    # fig.suptitle(model_name, fontsize=12)
    # plt.xlabel('epoch', fontsize=18)
    # plt.ylabel('avg prediction', fontsize=16)

    # plt.savefig(os.path.join(output_dir, 'discrim_prediction_plot.png'))

def save_losses(output_dir, val_losses, val_ious, val_des):
    # Make loss dir
    if not os.path.exists(os.path.join(output_dir, 'losses')):
        os.makedirs(os.path.join(output_dir, 'losses'))

    # Save losses
    # with open(output_dir+'losses\\G_loss.pkl', 'wb') as f:
    #     pickle.dump(G_loss, f)
    # with open(output_dir+'losses\\D_loss_fake.pkl', 'wb') as f:
    #     pickle.dump(D_loss_fake, f)
    # with open(output_dir+'losses\\D_loss_real.pkl', 'wb') as f:
    #     pickle.dump(D_loss_real, f)
    # with open(output_dir+'losses\\D_loss.pkl', 'wb') as f:
    #     pickle.dump(D_loss, f)
    # with open(output_dir+'losses\\avg_gen_pred.pkl', 'wb') as f:
    #     pickle.dump(avg_gen_pred, f)
    # with open(output_dir+'losses\\avg_real_pred.pkl', 'wb') as f:
    #     pickle.dump(avg_real_pred, f)

    with open(os.path.join(output_dir, 'losses', 'val_losses.pkl'), 'wb') as f:
        pickle.dump(val_losses, f)
    with open(os.path.join(output_dir, 'losses', 'val_ious.pkl'), 'wb') as f:
        pickle.dump(val_ious, f)
    # with open(output_dir + 'losses\\val_des.pkl', 'wb') as f:
    #     pickle.dump(val_des, f)

# def train_single(model_specs, train_data=None, train_data_info=None, val_data=None, val_data_info=None):
def train_single(model_specs, train_sets, val_sets, dataset='kitti_tracking'):
    """Train a single model on the given data"""
    [model_name, starting_step, data_cols,
     label_cols, label_dim, optimizer, w_adv,
     epochs, batch_size, k_d, k_g,
     show, output_dir] = model_specs

    hidden_layer_dim = 64
    timepoints = np.linspace(0.1, 1.0, 10)
    timepoint_step = 0.1
    tau = 1.345 # huber threshold is tau * sigma

    # Get Data
    if dataset == 'kitti_tracking':
        ...
        # train_data, train_data_info = data_extract_1obj.get_kitti_data(train_sets)
        # val_data, val_data_info = data_extract_1obj.get_kitti_data(val_sets)
    elif dataset == 'kitti_raw_tracklets':
        x_train, y_train, train_info = data_extract_1obj.get_kitti_raw_tracklets(timepoints, sets=train_sets, class_types=['Car', 'Van', 'Truck'])
        x_val, y_val, val_info = data_extract_1obj.get_kitti_raw_tracklets(timepoints, sets=val_sets, class_types=['Car', 'Van', 'Truck'])
        # print(train_data.shape, train_data.shape[0] / (train_data.shape[0] + val_data.shape[0]))
        # print(val_data.shape, val_data.shape[0] / (train_data.shape[0] + val_data.shape[0]))
    else:
        raise Exception("`dataset` parameter must be one of: ['kitti_tracking', 'kitti_raw_tracklets']")

    # Log hyperparameters
    steps_per_epoch = len(x_train) // batch_size
    nb_steps = steps_per_epoch*epochs
    log_hyperparams(model_name=model_name, output_dir=output_dir, train_sets=train_sets, val_sets=val_sets,
                    epochs=epochs, batch_size=batch_size, k_d=k_d, k_g=k_g, optimizer=optimizer, show=show,
                    dataset=dataset)

    # Get Model
    
    M = rnn_model.get_model_rnn(
        output_dir, hidden_layer_dim, timepoints, timepoint_step ,tau,
        optimizer=optimizer)

    # Train Model
    model_components = [model_name, starting_step, data_cols,
                        label_cols, label_dim,  # optimizer,
                        M,
                        epochs, batch_size, k_d, k_g,
                        show, output_dir]
    # [G_losses, D_losses, val_losses, train_ious, val_ious, train_des, val_des, avg_gen_pred, avg_real_pred] = gan_1obj.training_steps_GAN(x_train, x_val, y_train, y_val, train_info, val_info, model_components)
    [M_losses, val_losses, train_ious, val_ious, train_des, val_des] = rnn_model.train_rnn(x_train, x_val, y_train, y_val, train_info, val_info, model_components)
    # Save losses
    save_losses(output_dir, val_losses, val_ious, val_des)

    # Plot losses
    plot_loss(model_name, epochs, nb_steps, steps_per_epoch, output_dir, M_losses, val_losses,
              train_ious, val_ious, train_des, val_des, show_adv=(w_adv != 0.0))

    min_smoothl1_idx = np.argmin(val_losses)
    max_iou_idx = np.argmax(val_ious, axis=0)
    min_de_idx = np.argmin(val_des, axis=0)
    final_epoch = len(val_ious)

    print("Training / Testing completed. Showing test scores:\n")
    print("Best smoothL1 ({}): {}".format(min_smoothl1_idx + 1, val_losses[min_smoothl1_idx]))
    print("Best IoU ({}): {}".format(max_iou_idx + 1, val_ious[max_iou_idx]))
    print("Best DE ({}): {}".format(min_de_idx + 1, val_des[min_de_idx]))
    print("Final smoothL1 ({}): {}".format(final_epoch, val_losses[-1]))
    print("Final IoU ({}): {}".format(final_epoch, val_ious[-1]))
    print("Final DE ({}): {}".format(final_epoch, val_des[-1]))

    resultsFile = open(os.path.join(model_specs[-1], 'results.txt'), 'w')
    print("Training / Testing completed. Showing test scores:\n", file=resultsFile)
    print("Best smoothL1 ({}): {}".format(min_smoothl1_idx + 1, val_losses[min_smoothl1_idx]), file=resultsFile)
    print("Best IoU ({}): {}".format(max_iou_idx + 1, val_ious[max_iou_idx]), file=resultsFile)
    print("Best DE ({}): {}".format(min_de_idx + 1, val_des[min_de_idx]), file=resultsFile)
    print("Final smoothL1 ({}): {}".format(final_epoch, val_losses[-1]), file=resultsFile)
    print("Final IoU ({}): {}".format(final_epoch, val_ious[-1]), file=resultsFile)
    print("Final DE ({}): {}".format(final_epoch, val_des[-1]), file=resultsFile)

    return [val_losses, val_ious, val_des]


def train_k_fold(k, model_specs, dataset='kitti_tracking', seed=6):
    """Perform k-fold cross validation"""
    output_dir = model_specs[-1]

    np.random.seed(seed)
    if dataset == 'kitti_tracking':
        all_sets = np.arange(21)
        test_sets = np.random.choice(all_sets, 3, replace=False)
        remaining = np.setdiff1d(all_sets, test_sets)
        validation_groups = np.random.choice(remaining, (k, len(remaining)//k), replace=False)
    elif dataset == 'kitti_raw_tracklets':
        all_sets = np.arange(38)
        test_sets = np.random.choice(all_sets, 8, replace=False)
        remaining = np.setdiff1d(all_sets, test_sets)
        validation_groups = np.random.choice(remaining, (k, len(remaining)//k), replace=False)
    else:
        raise Exception("`dataset` parameter must be one of: ['kitti_tracking', 'kitti_raw_tracklets']")

    all_smoothl1 = np.empty((k, epochs))
    all_ious = np.empty((k, epochs))
    all_DEs = np.empty((k, epochs))
    for i in range(k):
        model_specs[-1] = os.path.join(output_dir, 'fold-' + str(i))

        train_sets = np.setdiff1d(remaining, validation_groups[i])
        [val_losses, val_ious, val_DEs] = train_single(model_specs, train_sets, validation_groups[i], dataset=dataset)

        all_smoothl1[i] = val_losses[:, 2].flatten()
        all_ious[i] = val_ious
        all_DEs[i] = val_DEs

    avgs_smoothl1 = np.mean(all_smoothl1, axis=0)
    min_smoothl1_idx = np.argmin(avgs_smoothl1)
    avgs_iou = np.mean(all_ious, axis=0)
    max_iou_idx = np.argmax(avgs_iou)
    avgs_DEs = np.mean(all_DEs, axis=0)
    min_DE_idx = np.argmin(avgs_DEs)

    print("K-fold cross validation completed.")
    print("Epoch for optimal smoothL1:", min_smoothl1_idx + 1)
    print("  -- mean smoothL1:", avgs_smoothl1[min_smoothl1_idx])
    print("  -- individual smoothL1s:", all_smoothl1[:, min_smoothl1_idx])
    print("Epoch for optimal iou:", max_iou_idx + 1)
    print("  -- mean iou:", avgs_iou[max_iou_idx])
    print("  -- individual ious:", all_ious[:, max_iou_idx])
    print("Epoch for optimal DEs:", min_DE_idx + 1)
    print("  -- mean DE:", avgs_DEs[min_DE_idx])
    print("  -- individual ious:", all_DEs[:, min_DE_idx])

    resultsFile = open(os.path.join(output_dir, 'results.txt'), 'w')
    resultsFile.write("Epoch for optimal smoothL1: {}\n".format(min_smoothl1_idx + 1))
    resultsFile.write("  -- mean smoothL1: {}\n".format(avgs_smoothl1[min_smoothl1_idx]))
    resultsFile.write("  -- individual smoothL1s: {}\n".format(all_smoothl1[:, min_smoothl1_idx]))
    resultsFile.write("Epoch for optimal iou: {}\n".format(max_iou_idx + 1))
    resultsFile.write("  -- mean iou: {}\n".format(avgs_iou[max_iou_idx]))
    resultsFile.write("  -- individual ious: {}\n".format(all_ious[:, max_iou_idx]))
    resultsFile.write("Epoch for optimal DEs: {}\n".format(min_DE_idx + 1))
    resultsFile.write("  -- mean DE: {}\n".format(avgs_DEs[min_DE_idx]))
    resultsFile.write("  -- individual DEs: {}\n".format(all_DEs[:, min_DE_idx]))

def train_k_fold_joint(k, model_specs, dataset='kitti_tracking', seed=6, stopping_epoch=None):
    output_dir = model_specs[-1]
    model_specs[-1] = os.path.join(output_dir,'joint')

    np.random.seed(seed)
    if dataset == 'kitti_tracking':
        all_sets = np.arange(21)
        test_sets = np.random.choice(all_sets, 3, replace=False)
        train_sets = np.setdiff1d(all_sets, test_sets)
    elif dataset == 'kitti_raw_tracklets':
        all_sets = np.arange(38)
        test_sets = np.random.choice(all_sets, 10, replace=False)
        train_sets = np.setdiff1d(all_sets, test_sets)
    else:
        raise Exception("`dataset` parameter must be one of: ['kitti_tracking', 'kitti_raw_tracklets']")

    # train_single(model_specs, train_sets, test_sets, dataset=dataset)
    [test_losses, test_ious, test_DEs] = train_single(model_specs, train_sets, test_sets, dataset=dataset)

    # min_smoothl1_idx = np.argmin(test_losses[:, 2])
    min_smoothl1_idx = np.argmin(test_losses)
    max_iou_idx = np.argmax(test_ious, axis=0)
    min_de_idx = np.argmin(test_DEs, axis=0)
    final_epoch = len(test_ious)

    print("Training / Testing completed. Showing test scores:\n")
    if stopping_epoch:
        print("smoothL1 at stopping_epoch({}): {}".format(stopping_epoch, test_losses[stopping_epoch - 1, 2]))
        print("IoU at stopping_epoch({}): {}".format(stopping_epoch, test_ious[stopping_epoch - 1]))
        print("DE at stopping_epoch({}): {}".format(stopping_epoch, test_DEs[stopping_epoch - 1]))
    # print("Best smoothL1 ({}): {}".format(min_smoothl1_idx + 1, test_losses[min_smoothl1_idx, 2]))
    print("Best smoothL1 ({}): {}".format(min_smoothl1_idx + 1, test_losses[min_smoothl1_idx]))

    print("Best IoU ({}): {}".format(max_iou_idx + 1, test_ious[max_iou_idx]))
    print("Best DE ({}): {}".format(min_de_idx + 1, test_DEs[min_de_idx]))
    # print("Final smoothL1 ({}): {}".format(final_epoch, test_losses[-1, 2]))
    print("Final smoothL1 ({}): {}".format(final_epoch, test_losses[-1]))
    print("Final IoU ({}): {}".format(final_epoch, test_ious[-1]))
    print("Final DE ({}): {}".format(final_epoch, test_DEs[-1]))

    resultsFile = open(os.path.join(model_specs[-1], 'results.txt'), 'w')
    print("Training / Testing completed. Showing test scores:\n", file=resultsFile)
    if stopping_epoch:
        print("smoothL1 at stopping_epoch({}): {}".format(stopping_epoch, test_losses[stopping_epoch - 1, 2]), file=resultsFile)
        print("IoU at stopping_epoch({}): {}".format(stopping_epoch, test_ious[stopping_epoch - 1]), file=resultsFile)
        print("DE at stopping_epoch({}): {}".format(stopping_epoch, test_DEs[stopping_epoch - 1]), file=resultsFile)
    # print("Best smoothL1 ({}): {}".format(min_smoothl1_idx + 1, test_losses[min_smoothl1_idx, 2]), file=resultsFile)
    print("Best smoothL1 ({}): {}".format(min_smoothl1_idx + 1, test_losses[min_smoothl1_idx]), file=resultsFile)
    print("Best IoU ({}): {}".format(max_iou_idx + 1, test_ious[max_iou_idx]), file=resultsFile)
    print("Best DE ({}): {}".format(min_de_idx + 1, test_DEs[min_de_idx]), file=resultsFile)
    # print("Final smoothL1 ({}): {}".format(final_epoch, test_losses[-1, 2]), file=resultsFile)
    print("Final smoothL1 ({}): {}".format(final_epoch, test_losses[-1]), file=resultsFile)
    print("Final IoU ({}): {}".format(final_epoch, test_ious[-1]), file=resultsFile)
    print("Final DE ({}): {}".format(final_epoch, test_DEs[-1]), file=resultsFile)

if __name__ == '__main__':
    # Define Training Parameters
    data_cols = []
    for fNum in range(1, 12):
        for char in ['L', 'T', 'W', 'H']:
            data_cols.append(char + str(fNum))
    label_cols = []
    label_dim = 0
    epochs = 600
    batch_size = 512 #4096 #4096  #7811  #15623 #1024  # 128, 64
    # steps_per_epoch = num_samples // batch_size  # ~1 epoch (35082 / 32 =~ 1096, 128: 274, 35082: 1)  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
    # nb_steps = steps_per_epoch*epochs  # 50000 # Add one for logging of the last interval
    starting_step = 0
    seed = 11

    k_d = 0  # 1 number of discriminator network updates per adversarial training step
    k_g = 1  # 1 number of generator network updates per adversarial training step
    w_adv = 0.0

    optimizer = {
                'name': 'adam',
                'lr': .00146,        # default: .001
                'beta_1': .9,       # default: .9
                'beta_2': .999,     # default: .999
                'decay': 0       # default: 0
                }
    model_name = 'quartic_sigma-2coeff-abs_red-sum_huber_t1.345xsig_seed-{}-TEST0f_VEHICLES-NOBIKE_7-fold_G3-64_{}-lr{}-b1{}-b2{}_bs{}_epochs{}'.format(
        seed, optimizer['name'], optimizer['lr'], optimizer['beta_1'], optimizer['beta_2'], batch_size, epochs
        )
    # model_name = 'maxGAN_SHOW-D-LEARN_1s-pred_G6-64_D3-32_w-adv{}_{}-lr{}-b1{}-b2{}_bs{}_kd{}_kg{}_epochs{}'.format(
    #     w_adv, optimizer['name'], optimizer['lr'], optimizer['beta_1'], optimizer['beta_2'], batch_size, k_d, k_g, epochs
    #     )

    output_dir = os.path.join(OUTPUT_DIR, model_name)
    show = True

    # Train Model
    model_specs = [model_name, starting_step, data_cols,
                   label_cols, label_dim, optimizer, w_adv,
                   epochs, batch_size, k_d, k_g,
                   show, output_dir]

    # # Train single model with random 30-8 split (kitti_raw_tracklets dataset)
    np.random.seed(11)
    all_sets = np.arange(38)
    test_sets = np.random.choice(all_sets, (3,10), replace=False)
    remaining = np.setdiff1d(all_sets, test_sets[0])
    #val_sets = np.random.choice(remaining, 5, replace=False) 
    #train_sets = np.setdiff1d(remaining, val_sets)
    train_single(model_specs, remaining, test_sets[0], dataset='kitti_raw_tracklets')

    # # Perform k-fold cross validation
    # train_k_fold(6, model_specs)
    # train_k_fold(6, model_specs, dataset='kitti_raw_tracklets', seed=10)

    # Train final k-fold model over all training / validation data
    # train_k_fold_joint(6, model_specs)
    #train_k_fold_joint(7, model_specs, dataset='kitti_raw_tracklets', seed=seed, stopping_epoch=None)
