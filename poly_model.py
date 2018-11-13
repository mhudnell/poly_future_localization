import numpy as np
import matplotlib.pyplot as plt
from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
import tensorflow as tf
import os
import re

from vis_tool import drawFrameRects, get_IoU, Rect, calc_metrics, calc_metrics_polynomial, calc_metrics_mult
import data_extract_1obj


def smoothL1(y_true, y_pred):
    tmp = tf.abs(y_pred - y_true)
    condition = tf.less(tmp, 1.)
    return tf.reduce_sum(tf.where(condition, tf.scalar_mul(0.5, tf.square(tmp)), tmp - 0.5), axis=-1)

def huber_generator(tau):

    def smoothL1(y_true, y_pred):
        abs_diff = tf.abs(y_pred - y_true)
        condition = tf.less(abs_diff, tau)

        out = tf.reduce_sum(
            tf.where(
                condition,
                tf.scalar_mul(0.5, tf.square(abs_diff)),
                tau*(abs_diff - 0.5*tau)),
                axis=-1)
        return out

    return smoothL1

# poly_order: highest degree on x in the polynomial (e.g., t^2 + t => 2)
# timepoints: list of future timepoints (offset from current) at which to
#   produce outputs
# returns: Bx4xT Tensor, with batch size B, 4 parameters, and T timepoints
def define_poly_network(poly_order, timepoints):
    poly_input = layers.Input(shape=(40, ), name="g_input")
    x = layers.Dense(64, activation="relu")(poly_input)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dense(64, activation="relu")(x)
    coeffs = layers.Dense(4 * poly_order, activation="linear")(x)
    coeffs = layers.Reshape((4, poly_order), name="coeffs")(coeffs)
    timepoints = K.constant(
        [[pow(t, i) for t in timepoints] for i in range(1, poly_order + 1)])
    output_op = layers.Lambda(lambda x: K.dot(x, timepoints), name="transforms")

    poly_output = output_op(coeffs)
    M = models.Model(inputs=[poly_input], outputs=[poly_output, coeffs], name='poly_regressor')

    return M


def get_model_poly(output_dir, poly_order, timepoints, tau, optimizer=None):
    K.set_learning_phase(1)
    M = define_poly_network(poly_order, timepoints)

    if optimizer and optimizer['name']=='adam':
        adam = optimizers.Adam(lr=optimizer['lr'], beta_1=optimizer['beta_1'], beta_2=optimizer['beta_2'], decay=optimizer['decay'])
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

    return M

def train_poly(x_train, x_val, y_train, y_val, train_info, val_info, model_components):
    """ """
    [model_name, starting_step, data_cols,
     label_cols, label_dim,
     M,
     epochs, batch_size, k_d, k_g,
     show, output_dir] = model_components
    steps_per_epoch_0 = len(x_train) // batch_size[0]
    steps_per_epoch_1 = len(x_train) // batch_size[1]
    num_total_epochs = epochs[0] + epochs[1]
    nb_steps = steps_per_epoch_0*epochs[0] + steps_per_epoch_1*epochs[1]
    val_input = x_val.reshape((len(x_val), 40))
    val_target = y_val

    # print('len(x_train):', len(x_train), 'batch_size:', batch_size, 'steps_per_epoch:', steps_per_epoch)
    print('x_train: ', x_train.shape)
    print('x_val: ', x_val.shape)
    print('y_train: ', y_train.shape)
    print('y_val: ', y_val.shape)

    # Store loss values for returning.
    M_losses = np.empty(nb_steps)          # [smoothL1]
    val_losses = np.empty(num_total_epochs)
    train_ious = np.empty((nb_steps, 2))    # Store .5 and 1.0 sec predictions
    val_ious = np.empty((num_total_epochs, 2))
    train_des = np.empty((nb_steps, 2))
    val_des = np.empty((num_total_epochs, 2))

    if not os.path.exists(os.path.join(output_dir, 'weights')):
        os.makedirs(os.path.join(output_dir, 'weights'))
    lossFile = open(os.path.join(output_dir, 'losses.txt'), 'w')

    curr_batch_size = batch_size[0]
    curr_steps_per_epoch = steps_per_epoch_0
    for i in range(1, nb_steps+1):

        if i == steps_per_epoch_0*epochs[0] + 1:
            print("SWITCHING BATCH SIZE", batch_size[1])
            curr_batch_size = batch_size[1]
            curr_steps_per_epoch = steps_per_epoch_1

        K.set_learning_phase(1)
        batch_ids = data_extract_1obj.get_batch_ids(len(x_train), curr_batch_size)
        gen_input = x_train[batch_ids]
        gen_target = y_train[batch_ids]

        # Randomly flip horizontally
        # data_extract_1obj.random_flip_batch(gen_input, gen_target)
        gen_input = gen_input.reshape((curr_batch_size, 40))
        M_loss = M.train_on_batch(gen_input, {'transforms': gen_target})
        # print('M_loss:', len(M_loss))
        # print(M_loss[2][0])
        # print(M_loss[3][0])
        gen_transforms = M_loss[2]

        # Calculate IoU and DE metrics.
        batch_ious = np.empty((curr_batch_size, 2))
        batch_des = np.empty((curr_batch_size, 2))
        for j in range(curr_batch_size):
            batch_ious[j], batch_des[j] = calc_metrics_mult(gen_input[j][-4:], gen_target[j], gen_transforms[j])

        avg_iou = np.mean(batch_ious, axis=0)
        avg_de = np.mean(batch_des, axis=0)

        M_losses[i-1] = M_loss[0]
        train_ious[i-1] = avg_iou
        train_des[i-1] = avg_de

        # Evaluate on validation / Save weights / Log loss every epoch
        if not i % curr_steps_per_epoch:
            K.set_learning_phase(0)
            # epoch = i // curr_steps_per_epoch
            epoch = min(i, steps_per_epoch_0*epochs[0]) // steps_per_epoch_0 + max(0, i-steps_per_epoch_0*epochs[0]) // steps_per_epoch_1

            # Evaluate on validation set
            num_val_samples = len(val_input)
            val_loss = M.test_on_batch(val_input, {'transforms': val_target})
            # coeffs = val_loss[3]
            gen_transforms = val_loss[2]

            val_batch_ious = np.empty((num_val_samples, 2))
            val_batch_des = np.empty((num_val_samples, 2))
            for j in range(num_val_samples):
                val_batch_ious[j], val_batch_des[j] = calc_metrics_mult(val_input[j][-4:], val_target[j], gen_transforms[j])

            # Print first sample.
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
            print('Epoch: {} of {}'.format(epoch, num_total_epochs))    # nb_steps // steps_per_epoch
            print('train_losses: {}'.format(M_losses[i-1]))
            print('val_losses: {}'.format(val_losses[epoch-1]))
            print('train_ious: {}'.format(train_ious[i-1]))
            print('val_ious: {}'.format(val_ious[epoch-1]))
            print('train_des: {}'.format(train_des[i-1]))
            print('val_des: {}'.format(val_des[epoch-1]))
            
            print('Epoch: {} of {}'.format(epoch, num_total_epochs), file=lossFile) # nb_steps // steps_per_epoch
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