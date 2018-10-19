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

from vis_tool import drawFrameRects, get_IoU, Rect
import data_extract_1obj

# def iou_metric(input):
#     def iou(y_true, y_pred):
#         print(input.get_shape())
#         anchor = tf.slice(input, [36], [4])
#         print(anchor.get_shape())
#         # sess = tf.InteractiveSession()
#         # anchor = input.eval()[-4:]
#         # return get_IoU(anchor, y_true.eval(), y_pred.eval())
#     return iou

def smoothL1(y_true, y_pred):
    tmp = tf.abs(y_pred - y_true)
    condition = tf.less(tmp, 1.)
    return tf.reduce_sum(tf.where(condition, tf.scalar_mul(0.5, tf.square(tmp)), tmp - 0.5), axis=-1)

# def combined_loss(y_true, y_pred, a=0.5, b=0.5):
#     return a*losses.binary_crossentropy(y_true, y_pred) + b*smoothL1(y_true, y_pred)

def generator_network(x, discrim_input_dim, base_n_count):
    # x = layers.Dense(base_n_count)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.LeakyReLU(0.1)(x)
    # x = layers.Dense(base_n_count*2)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.LeakyReLU(0.1)(x)
    # x = layers.Dense(base_n_count*4)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(4, activation="linear", name="g_output")(x)
    return x

def discriminator_network(x, discrim_input_dim, base_n_count):
    # x = layers.Dense(base_n_count*4)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.LeakyReLU(0.1)(x)
    # x = layers.Dense(base_n_count*2)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.LeakyReLU(0.1)(x)
    # x = layers.Dense(base_n_count)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return x

def define_models_GAN(gen_input_dim, discrim_input_dim, base_n_count, type=None):
    G_input = layers.Input(shape=(gen_input_dim, ), name="g_input")
    G_output = generator_network(G_input, discrim_input_dim, base_n_count)

    D_input = layers.Input(shape=(discrim_input_dim,))
    D_output = discriminator_network(D_input, discrim_input_dim, base_n_count)

    # This creates models which include the Input layer + hidden dense layers + output layer
    G = models.Model(inputs=[G_input], outputs=[G_output], name='generator')
    D = models.Model(inputs=[D_input], outputs=[D_output], name='discriminator')

    # 1. G takes G_input as input, returns a generated tensor
    # 2. D takes generated tensor + G_input as input, returns a tensor which is the combined output
    C_G_output = G(G_input)
    C_output = D(layers.concatenate([G_input, C_G_output]))
    C = models.Model(inputs=[G_input], outputs=[C_output, C_G_output], name='combined')

    return G, D, C

def training_steps_GAN(train_data, train_data_info, val_data, val_data_info, model_components):
    """ """
    [model_name, starting_step, data_cols,
     label_cols, label_dim,
     generator_model, discriminator_model, combined_model,
     epochs, batch_size, k_d, k_g,
     show, output_dir] = model_components
    steps_per_epoch = len(train_data) // batch_size
    nb_steps = steps_per_epoch*epochs

    # G_loss, D_loss_fake, D_loss_real, D_loss, val_losses = [], [], [], [], []
    G_losses = np.empty((nb_steps, 3))
    D_losses = np.empty((nb_steps, 3))
    val_losses = np.empty((nb_steps, 3))
    ious = np.empty((nb_steps, 2))          # Stores train and val data

    # Store average discrim prediction for generated and real samples every epoch.
    avg_gen_pred, avg_real_pred = [], []

    if not os.path.exists(output_dir + 'weights\\'):
        os.makedirs(output_dir + 'weights\\')
    lossFile = open(output_dir + 'losses.txt', 'w')

    # Log model structure to json
    with open(output_dir+"D_model.json", "w") as f:
        f.write(discriminator_model.to_json(indent=4))
    with open(output_dir+"G_model.json", "w") as f:
        f.write(generator_model.to_json(indent=4))

    for i in range(1, nb_steps+1):  # range(1, nb_steps+1)
        K.set_learning_phase(1)  # 1 = train

        # TRAIN DISCRIMINATOR on real and generated images
        for _ in range(k_d):
            batch = data_extract_1obj.get_batch(train_data, batch_size)
            gen_input = batch[:, :10*4]  # Only keep first 10 bounding boxes for gen input (11th is the target)

            g_z = generator_model.predict(gen_input)
            g_z = np.concatenate((gen_input, g_z), axis=1)

            ### TRAIN ON REAL (y = 1) w/ noise
            D_loss_real = discriminator_model.train_on_batch(batch, np.random.uniform(low=0.999, high=1.0, size=batch_size))      # 0.7, 1.2 GANs need noise to prevent loss going to zero

            ### TRAIN ON GENERATED (y = 0) w/ noise
            D_loss_fake = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size))    # 0.0, 0.3
            D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

        # Only keep most recent loss value (if k_d > 1)
        D_losses[i-1] = np.array([D_loss, D_loss_real, D_loss_fake])

        # TRAIN GENERATOR on real inputs and outputs
        for _ in range(k_g):
            batch = data_extract_1obj.get_batch(train_data, batch_size)
            gen_input = batch[:, :10*4]  # Only keep first 10 bounding boxes for gen input (11th is the target)
            gen_target = batch[:, -4:]  # Get last (target) bounding box

            ### TRAIN (y = 1) bc want pos feedback for tricking discrim (want discrim to output 1)
            G_loss = combined_model.train_on_batch(gen_input, {'discriminator': np.random.uniform(low=0.999, high=1.0, size=batch_size),
                                                               'generator': gen_target})
            y_preds = G_loss[4]
            avg_iou = np.mean([get_IoU(gen_input[i][-4:], gen_target[i], y_preds[i]) for i in range(len(y_preds))])

            K.set_learning_phase(0)
            # Run validation batch
            batch = data_extract_1obj.get_batch(val_data, batch_size)
            gen_input = batch[:, :10*4]
            gen_target = batch[:, -4:]
            val_loss = combined_model.test_on_batch(gen_input, {'discriminator': np.random.uniform(low=0.999, high=1.0, size=batch_size),
                                                                'generator': gen_target})
            val_avg_iou = np.mean([get_IoU(gen_input[i][-4:], gen_target[i], val_loss[4][i]) for i in range(len(y_preds))])

        G_losses[i-1] = G_loss[:3]
        val_losses[i-1] = val_loss[:3]
        ious[i-1] = [avg_iou, val_avg_iou]

        # Save weights / Log loss every epoch
        if not i % steps_per_epoch:
            K.set_learning_phase(0) # 0 = test
            epoch = i // steps_per_epoch

            a_g_p, a_r_p = test_discrim(train_data, generator_model, discriminator_model, combined_model)
            avg_gen_pred.append(a_g_p)
            avg_real_pred.append(a_r_p)

            print('Epoch: {} of {}'.format(epoch, nb_steps // steps_per_epoch))
            print('D_losses: {}'.format(D_losses[i-1]))
            print('G_losses: {}'.format(G_losses[i-1]))
            print('val_losses: {}'.format(val_losses[i-1]))
            print('ious: {}'.format(ious[i-1]))
            print('avg_gen_pred: {} | avg_real_pred: {}\n'.format(a_g_p, a_r_p))

            lossFile.write('Epoch: {} of {}.\n'.format(epoch, nb_steps // steps_per_epoch))
            lossFile.write('D_losses: {}'.format(D_losses[i-1]))
            lossFile.write('G_losses: {}'.format(G_losses[i-1]))
            lossFile.write('val_losses: {}'.format(val_losses[i-1]))
            lossFile.write('avg_gen_pred: {} | avg_real_pred: {}\n'.format(a_g_p, a_r_p))

            # Checkpoint: Save model weights
            model_checkpoint_base_name = output_dir + 'weights\\{}_weights_epoch-{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('gen', epoch))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discrim', epoch))

    return [G_losses, D_losses, val_losses, ious, avg_gen_pred, avg_real_pred]

def get_model(data_cols, generator_model_path=None, discriminator_model_path=None, loss_pickle_path=None, seed=0, optimizer=None):
    discrim_input_dim = len(data_cols)
    gen_input_dim = 40
    base_n_count = 128
    show = True

    # Define network models.
    K.set_learning_phase(1)  # 1 = train
    G, D, C = define_models_GAN(gen_input_dim, discrim_input_dim, base_n_count)

    # adam = optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.999)
    if optimizer and optimizer['name']=='adam':
        adam = optimizers.Adam(lr=optimizer['lr'], beta_1=optimizer['beta_1'], beta_2=optimizer['beta_2'], decay=optimizer['decay'])
    else:
        adam = optimizers.Adam()

    # G.compile(optimizer=adam, loss='binary_crossentropy')
    D.compile(optimizer=adam, loss='binary_crossentropy')
    print(D.summary())
    D.trainable = False  # Freeze discriminator weights in combined model (we want to improve model by improving generator, rather than making the discriminator worse)
    C.compile(optimizer=adam, loss={'discriminator': 'binary_crossentropy', 'generator': smoothL1},
              loss_weights={'discriminator': 0.5, 'generator': 0.5})

    # Add model outputs to return values
    C.metrics_tensors += C.outputs

    if show:
        print(G.summary())
        print(D.summary())
        print(C.summary())

    # LOAD WEIGHTS (and previous loss logs) IF PROVIDED
    # if loss_pickle_path:
    #     print('Loading loss pickles')
    #     [G_loss, D_loss_fake, D_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        G.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        D.load_weights(discriminator_model_path, by_name=True)

    return G, D, C

# def test_model(generator_model, discriminator_model, combined_model, model_name):
#   """Test model on a hand-picked sample from the data set."""
#   data_x, data_y, files_x, files_y = get_data()

#   data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\bounding box images\\'
#   if not os.path.exists(data_dir):
#     os.makedirs(data_dir)

#   test_x_pretty = data_x[20000]
#   test_y_pretty = data_y[20000]
#   print("file_x: ", files_x[20000])
#   print("file_x: ", files_y[20000])

#   test_x = np.reshape(test_x_pretty, (1, -1))
#   test_y = np.reshape(test_y_pretty, (1, -1))

#   test_g_z = generator_model.predict(test_x)
#   test_g_z = np.concatenate((test_x, test_g_z), axis=1)
#   test_g_z_pretty = np.reshape(test_g_z, (11,4))

#   dpred_real = discriminator_model.predict(test_y)
#   dpred_gen = discriminator_model.predict(test_g_z)
#   print("dpred_real: ", dpred_real," dpred_gen: ", dpred_gen)

#   # Undo normalization.
#   test_g_z_pretty[:,0] = test_g_z_pretty[:,0] * 1240  # L
#   test_g_z_pretty[:,1] = test_g_z_pretty[:,1] * 374   # T
#   test_g_z_pretty[:,2] = test_g_z_pretty[:,2] * 1240  # W
#   test_g_z_pretty[:,3] = test_g_z_pretty[:,3] * 374   # H
#   test_y_pretty[:,0] = test_y_pretty[:,0] * 1240
#   test_y_pretty[:,1] = test_y_pretty[:,1] * 374
#   test_y_pretty[:,2] = test_y_pretty[:,2] * 1240
#   test_y_pretty[:,3] = test_y_pretty[:,3] * 374

#   # # Log results.
#   # realfile = open(data_dir+'real.txt', 'w')
#   # genfile = open(data_dir+'gen.txt', 'w')
#   # realfile.write("%s\n" % test_y_pretty[10])
#   # genfile.write("%s\n" % test_g_z_pretty[10])

#   # Draw Results.
#   frames = ['000040.png']
#   print("test_g_z_pretty: ",test_g_z_pretty)
#   print("test_y_pretty: ",test_y_pretty)
#   drawFrameRects('0016', frames[0], test_g_z_pretty, isGen=True, folder_dir=data_dir)
#   drawFrameRects('0016', frames[0], test_y_pretty, isGen=False, folder_dir=data_dir)

#   return


def test_model_multiple(generator_model, discriminator_model, combined_model, model_name):
    """Test model on a hand-picked set of samples from the data set."""
    # samples, samples_info = data_extract_1obj.get_kitti_training(normalize=True)
    samples, samples_info = data_extract_1obj.get_kitti_testing(normalize=True)

    data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\bounding box images\\'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # test_set = [0, 1234, 2345, 3456, 10000, 15000, 19999, 20000]
    np.random.seed(7)
    test_set = np.random.choice(len(samples), 20, replace=False)

    for i in test_set:
        sample_set = samples_info[i][0]
        frame = samples_info[i][1]
        object_id = samples_info[i][2]
        print("sample_set:", sample_set, "frame:", frame, "object_id:", object_id)

        target = samples[i]

        target_vector = np.reshape(target, (1, -1))
        gen_input = target_vector[:, :10*4]

        g_z = generator_model.predict(gen_input)
        gen_out = np.concatenate((gen_input, g_z), axis=1)
        generated = np.reshape(gen_out, (11, 4))

        d_pred_real = discriminator_model.predict(target_vector)
        d_pred_gen = discriminator_model.predict(gen_out)
        print("d_pred_real:", d_pred_real, "d_pred_gen:", d_pred_gen)
        print("generated_transform:", generated[-1])
        print("target_transform:", target[-1])
        print("iou:", get_IoU(target[-2], target[-1], generated[-1], sample_set=sample_set))
        # with tf.Session() as sess:
        #     print("smoothL1 loss:", sess.run(smoothL1(target[-1], generated[-1])), "\n")

        # Undo normalization.
        # data_extract_1obj.unnormalize_sample(generated, sample_set)
        # data_extract_1obj.unnormalize_sample(target, sample_set)

        # Draw Results.
        drawFrameRects(sample_set, frame, object_id, generated, isGen=True, folder_dir=data_dir)
        drawFrameRects(sample_set, frame, object_id, target, isGen=False, folder_dir=data_dir)

    return

def test_model_IOU(generator_model, discriminator_model, combined_model, model_name):
    # samples, samples_info = data_extract_1obj.get_kitti_training(normalize=True)
    samples, samples_info = data_extract_1obj.get_kitti_testing(normalize=True)
    print(samples.shape)
    vectors = samples.reshape((len(samples), -1))
    print(vectors.shape)
    ious = np.empty(len(vectors))
    # gen_input = vectors[:10*4]
    for i, sample in enumerate(vectors):
        sample_set = samples_info[i][0]

        gen_input = sample[:10*4]
        # print(gen_input.shape)
        g_z = generator_model.predict(gen_input.reshape((1, -1)))

        # print(samples[i][-1].shape, g_z.shape)
        ious[i] = get_IoU(samples[i][-2], samples[i][-1], g_z[0], sample_set=sample_set)

    print("avg IOU:", np.mean(ious))
    return


def test_discrim(train_data, generator_model, discriminator_model, combined_model):
    """Test the discriminator by having it produce a realness score for generated and target images in a sample set."""
    # samples, _ = data_extract_1obj.get_kitti_training(normalize=True)
    batch = data_extract_1obj.get_batch(train_data, 590)  # , seed=7

    gen_correct = 0
    gen_incorrect = 0
    gen_unsure = 0

    real_correct = 0
    real_incorrect = 0
    real_unsure = 0

    d_preds_gen = np.zeros(shape=len(batch))
    d_preds_real = np.zeros(shape=len(batch))

    for i in range(len(batch)):
        target_vector = batch[i].reshape((1, -1))  # Keras expects a 2d input to predict

        # target_vector = np.reshape(target, (1, -1))  # Not needed, each sample in batch is already vectorized
        gen_input = target_vector[:, :10*4]  # Leave out target
        g_z = generator_model.predict(gen_input)
        gen_out = np.concatenate((gen_input, g_z), axis=1)

        d_pred_real = discriminator_model.predict(target_vector)
        d_pred_gen = discriminator_model.predict(gen_out)
        d_preds_gen[i] = d_pred_gen
        d_preds_real[i] = d_pred_real

        if d_pred_gen == 1.0:
            gen_incorrect += 1
        elif d_pred_gen == 0.0:
            gen_correct += 1
        else:
            gen_unsure += 1

        if d_pred_real == 1.0:
            real_correct += 1
        elif d_pred_real == 0.0:
            real_incorrect += 1
        else:
            real_unsure += 1

    avg_pred_gen = np.average(d_preds_gen)
    avg_pred_real = np.average(d_preds_real)

    # print("gen_correct: ", gen_correct," gen_incorrect: ", gen_incorrect, " gen_unsure: ", gen_unsure, " avg_output: ", avg_gen_pred)
    # print("real_correct: ", real_correct," real_incorrect: ", real_incorrect, " real_unsure: ", real_unsure, " avg_output: ", avg_real_pred)
    return avg_pred_gen, avg_pred_real
