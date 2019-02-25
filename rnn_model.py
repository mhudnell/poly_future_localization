import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import keras
#from keras import applications
from tensorflow.python.keras import applications
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
import tensorflow as tf
import os
import re

from vis_tool import get_IoU, Rect, calc_metrics, calc_metrics_mult
import data_extract_1obj

def smoothL1(y_true, y_pred):
    tmp = tf.abs(y_pred - y_true)
    condition = tf.less(tmp, 1.)
    return tf.reduce_sum(tf.where(condition, tf.scalar_mul(0.5, tf.square(tmp)), tmp - 0.5), axis=-1)

# returns the Huber loss without assuming a fixed sigma in the distribution
# tau: threshold for the distribution switch
def huber_generator(k):  # tau, k
    # y_true: Bx4xTx1 (keras requires this to be 4 dimensions)
    # y_pred: Bx4xTx2 (mean, stdev)
    def huber(y_true, y_pred):
        mu, sigma = y_pred[...,0], y_pred[...,1]
        mu = tf.reshape(mu, [-1,4,10,1])
        sigma = tf.reshape(sigma, [-1,4,10,1])

        inv_sigma_sq = 1. / tf.square(sigma)
        tau = k*sigma
        # tau = tf.clip_by_value(tau, 0.0, 1.0)

        abs_diff = tf.abs(y_true - mu)
        squared_diff = tf.square(y_true - mu)
        huber_loss = inv_sigma_sq * tf.where(
            tf.less(abs_diff, tau),
            0.5 * squared_diff,
            (tau * abs_diff - 0.5 * tau * tau))
        
        confidence_penalty = tf.log(
            sigma * np.sqrt(2. * np.pi) * tf.erf((tau / np.sqrt(2.)) / sigma) +
            (2. / tau) * tf.square(sigma) * tf.exp(
                (-0.5 * tau * tau) * inv_sigma_sq))

        return tf.reduce_sum(tf.add(huber_loss, confidence_penalty))

    return huber

# hidden_state_dim: number of values to use for the hidden state of the GRU
# timepoints: list of future timepoints (offset from current) at which to
#   produce outputs
# timepoint_step: rate of output for one application of the RNN
# returns: Bx4xTx2 Tensor, with batch size B, 4 parameters, T timepoints, and
#   a parameter mean and stdev per (parameter, timepoint)
def define_rnn_network(hidden_state_dim, timepoints, timepoint_step):
    rnn_input = layers.Input(shape=(40, ), name="g_input")
    x = layers.Dense(64, activation="relu")(rnn_input)
    x = layers.Dense(64, activation="relu")(x)
    # The last output predicts the initial state for the RNN.
    gru_state = layers.Dense(hidden_state_dim, activation="relu")(x)

    # The RNN state is passed through a separate regressor to obtain the (mean,
    # scale) for each of the 4 dimensions.
    gru = layers.GRU(hidden_state_dim)
    intermediate_regressor = layers.Dense(64, activation="relu")
    mean_and_scale_regressor = layers.Dense(
        8, activation="linear", name="mean_and_scale_regressor")
    # Ensure the regressed scale is positive; also reshape to Bx8 -> Bx1x8.
    kSigmaMin = 1e-3  # avoid poor conditioning
    absolute_scale_op = layers.Lambda(
        lambda x: K.concatenate(
            (x[:,:4], K.abs(x[:,4:]) + kSigmaMin), axis=1)[:,tf.newaxis,:])
    output_regressor = lambda x: \
        absolute_scale_op(mean_and_scale_regressor(intermediate_regressor(x)))

    current_output = output_regressor(gru_state) # predict the first output
    t = timepoint_step
    next_timepoint_idx = 0

    outputs = []

    while t <= timepoints[-1] or np.isclose(t, timepoints[-1]):
        if np.isclose(t, timepoints[next_timepoint_idx]):
            outputs.append(current_output)
            t = timepoints[next_timepoint_idx] # avoid any accumulative drift
            next_timepoint_idx += 1
            if next_timepoint_idx == len(timepoints):
                break
        #current_output_with_time = layers.Lambda(
        #    lambda x: K.concatenate((x, K.zeros_like(x)[:,:,:1] + t)))(
        #    current_output)
        gru_state = gru(current_output, initial_state=gru_state)
        current_output = output_regressor(gru_state)
        t += timepoint_step

    assert(len(outputs) == len(timepoints))

    # Join the T [Bx1x8] outputs into a Bx8xT tensor, then split in half down
    # the first axis and reform into a Bx4xTx2 tensor.
    def rejoin_op(x):
        x = K.stack(x, axis=-1)
        return K.stack((x[:,0,:4,:], x[:,0,4:,:]), axis=-1)
    output = layers.Lambda(rejoin_op, name="transforms")(outputs)

    M = models.Model(
        inputs=[rnn_input], outputs=[output], name='rnn_regressor')

    return M



def get_model_rnn(output_dir, hidden_state_dim, timepoints, timepoint_step,
                   tau, optimizer=None, weights_path=None):
    K.set_learning_phase(1)
    M = define_rnn_network(hidden_state_dim, timepoints, timepoint_step)

    if optimizer and optimizer['name']=='adam':
        adam = optimizers.Adam(lr=optimizer['lr'], beta_1=optimizer['beta_1'],
           beta_2=optimizer['beta_2'], decay=optimizer['decay'], amsgrad=False)
    else:
        raise Exception('Must specify optimizer.')

    M.compile(optimizer=adam, loss={'transforms': huber_generator(tau)})
    print(M.summary())

    # Add model outputs to return values
    # output will now be: [g_loss, g_loss_adv, smooth_l1, d_output, g_output]
    M.metrics_tensors += M.outputs
    print("metrics_names:", M.metrics_names)

    # Log model structure to json
    # with open(os.path.join(output_dir, 'M_model.json'), "w") as f:
    #     f.write(M.to_json(indent=4))

    if weights_path:
        print('Loading model')
        M.load_weights(weights_path, by_name=True)

    return M

def train_rnn(x_train, x_val, y_train, y_val, train_info, val_info, model_components):
    """ """
    [model_name, starting_step, data_cols,
     label_cols, label_dim,
     M,
     epochs, batch_size, k_d, k_g,
     show, output_dir] = model_components
    steps_per_epoch = len(x_train) // batch_size
    nb_steps = steps_per_epoch*epochs
    val_input = x_val.reshape((len(x_val), 40))
    val_target = y_val.reshape((len(y_val), 4, 10, 1))

    print('len(x_train):', len(x_train), 'batch_size:', batch_size, 'steps_per_epoch:', steps_per_epoch)
    print('x_train: ', x_train.shape)
    print('x_val: ', x_val.shape)
    print('y_train: ', y_train.shape)
    print('y_val: ', y_val.shape)

    # Store loss values for returning.
    M_losses = np.empty(nb_steps)          # [smoothL1]
    val_losses = np.empty(epochs)
    train_ious = np.empty((nb_steps, 2))    # Store .5 and 1.0 sec predictions
    val_ious = np.empty((epochs, 2))
    train_des = np.empty((nb_steps, 2))
    val_des = np.empty((epochs, 2))

    if not os.path.exists(os.path.join(output_dir, 'weights')):
        os.makedirs(os.path.join(output_dir, 'weights'))
    lossFile = open(os.path.join(output_dir, 'losses.txt'), 'w')

    for i in range(1, nb_steps+1):  # range(1, nb_steps+1)
        K.set_learning_phase(1)
        batch_ids = data_extract_1obj.get_batch_ids(len(x_train), batch_size)
        gen_input = x_train[batch_ids].reshape((batch_size, 40))
        gen_target = y_train[batch_ids].reshape((batch_size, 4, 10, 1))
        #gen_input = x_train[batch_ids]
        #gen_target = y_train[batch_ids]

        #data_extract_1obj.random_flip_batch(gen_input, gen_target)
        #gen_input = gen_input.reshape((batch_size, 40))
        #gen_target = gen_target.reshape((batch_size, 4, 10, 1))
        ### TRAIN (y = 1) bc want pos feedback for tricking discrim (want discrim to output 1)
        M_loss = M.train_on_batch(gen_input, {'transforms': gen_target})
        # print('M_loss:', len(M_loss))
        # print(M_loss[2][0])
        # print(M_loss[3][0])
        gen_transforms = M_loss[1]

        # Calculate IoU and DE metrics.
        batch_ious = np.empty((batch_size, 2))
        batch_des = np.empty((batch_size, 2))
        for j in range(batch_size):
            batch_ious[j], batch_des[j] = calc_metrics_mult(gen_input[j][-4:], gen_target[j], gen_transforms[j])

        avg_iou = np.mean(batch_ious, axis=0)
        avg_de = np.mean(batch_des, axis=0)

        M_losses[i-1] = M_loss[0]
        train_ious[i-1] = avg_iou
        train_des[i-1] = avg_de

        # Evaluate on validation / Save weights / Log loss every epoch
        if not i % steps_per_epoch:
            K.set_learning_phase(0)
            epoch = i // steps_per_epoch

            # Evaluate on validation set
            num_val_samples = len(val_input)
            val_loss = M.test_on_batch(val_input, {'transforms': val_target})
            gen_transforms = val_loss[1]

            val_batch_ious = np.empty((num_val_samples, 2))
            val_batch_des = np.empty((num_val_samples, 2))
            for j in range(num_val_samples):
                val_batch_ious[j], val_batch_des[j] = calc_metrics_mult(val_input[j][-4:], val_target[j], gen_transforms[j])

            # Print first sample.
            print(gen_transforms[0, :, 4])
            print(gen_transforms[0, :, 9])
            # t_bb = data_extract_1obj.transform(val_input[0][-4:], val_target[0][:, 9])
            # t_bb = data_extract_1obj.unnormalize_bb(t_bb, sample_set=None)
            # g_bb = data_extract_1obj.transform(val_input[0][-4:], y_preds[0][:, 9])
            # g_bb = data_extract_1obj.unnormalize_bb(g_bb, sample_set=None)
            # print("proposal: ", g_bb)
            # print("target: ", t_bb)

            val_avg_iou = np.mean(val_batch_ious, axis=0)
            val_avg_de = np.mean(val_batch_des, axis=0)

            val_losses[epoch-1] = val_loss[0]
            val_ious[epoch-1] = val_avg_iou
            val_des[epoch-1] = val_avg_de

            # Log loss info to console / file
            print('Epoch: {} of {}'.format(epoch, nb_steps // steps_per_epoch))
            print('train_losses: {}'.format(M_losses[i-1]))
            print('val_losses: {}'.format(val_losses[epoch-1]))
            print('train_ious: {}'.format(train_ious[i-1]))
            print('val_ious: {}'.format(val_ious[epoch-1]))
            print('train_des: {}'.format(train_des[i-1]))
            print('val_des: {}'.format(val_des[epoch-1]))
            
            print('Epoch: {} of {}'.format(epoch, nb_steps // steps_per_epoch), file=lossFile)
            print('train_losses: {}'.format(M_losses[i-1]), file=lossFile)
            print('val_losses: {}'.format(val_losses[epoch-1]), file=lossFile)
            print('train_ious: {}'.format(train_ious[i-1]), file=lossFile)
            print('val_ious: {}'.format(val_ious[epoch-1]), file=lossFile)
            print('train_des: {}'.format(train_des[i-1]), file=lossFile)
            print('val_des: {}'.format(val_des[epoch-1]), file=lossFile)

            # Checkpoint: Save model weights
            weight_path = os.path.join(output_dir, 'weights', 'M_weights_epoch-{}.h5'.format(epoch))
            M.save_weights(weight_path)

    return [M_losses, val_losses, train_ious, val_ious, train_des, val_des]
