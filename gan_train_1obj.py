import gan_1obj
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import logging
import data_extract_1obj

def log_hyperparams(model_name=None, output_dir=None, epochs=None, steps_per_epoch=None, nb_steps=None, batch_size=None, k_d=None, k_g=None, optimizer=None, show=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.basicConfig(filename=output_dir+'hyperparameters.log', level=logging.DEBUG, format='%(levelname)s:%(message)s', filemode='w')
    logging.info('Logging hyperparameters:')
    logging.info('  --model_name: {}'.format(model_name))
    logging.info('  --output_dir: {}'.format(output_dir))
    logging.info('  --epochs: {}'.format(epochs))
    logging.info('  --steps_per_epoch: {}'.format(steps_per_epoch))
    logging.info('  --nb_steps: {}'.format(nb_steps))
    logging.info('  --batch_size: {}'.format(batch_size))
    logging.info('  --k_d: {}'.format(k_d))
    logging.info('  --k_g: {}'.format(k_g))
    logging.info('  --show: {}'.format(show))

    if optimizer:
        logging.info('  --optimizer: {}'.format(optimizer['name']))
        logging.info('    |-lr: {}'.format(optimizer['lr']))
        logging.info('    |-beta1: {}'.format(optimizer['beta_1']))
        logging.info('    |-beta2: {}'.format(optimizer['beta_2']))
        logging.info('    |-decay: {}'.format(optimizer['decay']))

# def plot_loss()

def save_losses(G_loss, D_loss_fake, D_loss_real, D_loss, avg_gen_pred, avg_real_pred):
    # Make loss dir
    if not os.path.exists(output_dir + 'losses\\'):
        os.makedirs(output_dir + 'losses\\')

    # Save losses
    with open(output_dir+'losses\\G_loss.pkl', 'wb') as f:
        pickle.dump(G_loss, f)
    with open(output_dir+'losses\\D_loss_fake.pkl', 'wb') as f:
        pickle.dump(D_loss_fake, f)
    with open(output_dir+'losses\\D_loss_real.pkl', 'wb') as f:
        pickle.dump(D_loss_real, f)
    with open(output_dir+'losses\\D_loss.pkl', 'wb') as f:
        pickle.dump(D_loss, f)
    with open(output_dir+'losses\\avg_gen_pred.pkl', 'wb') as f:
        pickle.dump(avg_gen_pred, f)
    with open(output_dir+'losses\\avg_real_pred.pkl', 'wb') as f:
        pickle.dump(avg_real_pred, f)

if __name__ == '__main__':
    # Define Training Parameters
    data_cols = []
    for fNum in range(1, 12):
        for char in ['L', 'T', 'W', 'H']:
            data_cols.append(char + str(fNum))
    label_cols = []
    label_dim = 0
    batch_size = 1024  # 128, 64
    samples, _ = data_extract_1obj.get_kitti_training()
    num_samples = len(samples)
    print(num_samples)
    steps_per_epoch = num_samples // batch_size  # ~1 epoch (35082 / 32 =~ 1096, 128: 274, 35082: 1)  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
    epochs = 50
    starting_step = 0
    nb_steps = steps_per_epoch*epochs  # 50000 # Add one for logging of the last interval

    k_d = 1  # 1 number of discriminator network updates per adversarial training step
    k_g = 1  # 1 number of generator network updates per adversarial training step

    optimizer = {
                'name': 'adam',
                'lr': .001,        # default: .001
                'beta_1': .9,       # default: .9
                'beta_2': .999,     # default: .999
                'decay': 0       # default: 0
                }
    model_name = 'maxGAN_scale_split2_G6-64_D3-32_0.5adv_{}-{}lr-{}beta1-{}beta2_bs{}_kd{}_kg{}_epochs{}'.format(optimizer['name'], optimizer['lr'], optimizer['beta_1'], optimizer['beta_2'],
                                                                                                             batch_size, k_d, k_g, epochs)
    output_dir = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\'
    show = True

    # Log Hyperparameters
    log_hyperparams(model_name=model_name, output_dir=output_dir, epochs=epochs, steps_per_epoch=steps_per_epoch, nb_steps=nb_steps,
                batch_size=batch_size, k_d=k_d, k_g=k_g, optimizer=optimizer, show=show)
    
    # Get Model
    generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, optimizer=optimizer)
    print("metrics_names:", combined_model.metrics_names)

    # Train Model
    model_components = [model_name, starting_step,
                        data_cols, label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        nb_steps, batch_size, k_d, k_g,
                        steps_per_epoch, show, output_dir]
    [G_loss, D_loss_fake, D_loss_real, D_loss, avg_gen_pred, avg_real_pred] = gan_1obj.training_steps_GAN(model_components)

    # losses = [G_loss, D_loss_fake, D_loss_real, D_loss, avg_gen_pred, avg_real_pred]

    

    # PLOT LOSS
    x = np.arange(nb_steps)
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)

    G_loss = np.array(G_loss)
    ax1.plot(x, D_loss_fake, label='d_loss_fake')
    ax1.plot(x, D_loss_real, label='d_loss_real')
    ax1.plot(x, D_loss, label='d_loss')
    ax1.plot(x, G_loss[:, 1], label='g_loss_adv')
    ax1.plot(x, G_loss[:, 2], label='smoothL1')
    ax1.plot(x, G_loss[:, 0], label='g_loss')
    ax1.legend(loc=1)
    fig.suptitle(model_name, fontsize=20)
    plt.xlabel('number of steps', fontsize=18)
    plt.ylabel('loss', fontsize=16)

    plt.savefig(output_dir + 'loss_plot.png')

    # PLOT DISCRIM PREDICTIONS

    x = np.arange(epochs)
    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_subplot(111)

    ax1.plot(x, avg_gen_pred, label='avg_gen_pred')
    ax1.plot(x, avg_real_pred, label='avg_real_pred')

    ax1.legend(loc=1)
    fig.suptitle(model_name, fontsize=20)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('avg prediction', fontsize=16)

    plt.savefig(output_dir + 'discrim_prediction_plot.png')

    # if __name__ == '__main__':
        
