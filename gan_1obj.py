import numpy as np
import matplotlib.pyplot as plt

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import os
import re

from vis_tool import drawFrameRects

def smoothL1(y_true, y_pred):
    tmp = tf.abs(y_pred - y_true)
    condition = tf.less(tmp, 1.)
    return tf.reduce_sum(tf.where(condition, tf.scalar_mul(0.5,tf.square(tmp)), tmp - 0.5), axis=-1)


def get_data_batch(data_x, data_y, batch_size, seed=0):
    """
    Retreive (batch_size) number of samples, and return a set of training data (x) and its corresponding target (y) data.
    Data_x and data_y must be the same length and samples in these arrays should be in the same order (i.e. data_y[i] corresponds to the 'future' positions of data_x[i])

    Args:
        data_x (array): The full set of observed data.
        data_y (array): The full set of target data.
        batch_size (int): The number of samples to return.
        seed (int): A seed to set for the random sample generator, to be used for debug purposes.
    """
    numSamples = len(data_x)
    if numSamples != len(data_y): print('ERROR')

    # get a random start index in range: [0, numSamples]
    start_i = (batch_size * seed) % numSamples
    stop_i = start_i + batch_size

    # generates a list of randomly shuffled indices
    shuffle_seed = (batch_size * seed) // numSamples
    np.random.seed(shuffle_seed)
    indices = np.random.choice(numSamples, size=numSamples, replace=False )
    indices = list(indices) + list(indices) # duplicate to cover ranges past the end of the set (when stop_i > numSamples)

    # acquire random batches
    x = data_x[ indices[ start_i: stop_i ] ]
    y = data_y[ indices[ start_i: stop_i ] ]

    if batch_size == 1:
        print(indices[ start_i: stop_i ])
    
    return np.reshape(x, (batch_size, -1) ), np.reshape(y, (batch_size, -1) )

def get_data():
    """
    Retrieve past and future data from the specified folders, and return the values in arrays.
    Also return the file paths associated with each set of data.

    Returns:
        past_all ((35082,10,4) array): 
        future_all ((35082,10,4) array): 
        past_files ((35082,1) array): 
        future_files: ((35082,1) array): 
    """
    past_all = np.empty([35082, 10, 4])
    past_files = []

    file_num1 = 0
    for path, subdirs, files in os.walk('F:\\Car data\\label_02_extracted\\past_1obj_LTWH'):
        for name in files:
            fpath = os.path.join(path, name)
            past_files.append(fpath)
            f = open(fpath,'r')
            past_one = np.empty([10, 4])
            for i in range(10):
                line = f.readline()
                if not line:
                    break
                words = line.split()
                past_one[i] = [float(word) for word in words]
            past_all[file_num1] = past_one
            file_num1 += 1

    future_all = np.empty([35082, 11, 4])
    future_files = []

    file_num2 = 0
    for path, subdirs, files in os.walk('F:\\Car data\\label_02_extracted\\future_1obj_LTWH'):
        for name in files:
            fpath = os.path.join(path, name)
            future_files.append(fpath)
            f = open(fpath,'r')
            future_one = np.empty([11, 4])
            for i in range(11):
                line = f.readline()
                if not line:
                    break
                words = line.split()
                future_one[i] = [float(word) for word in words]
            future_all[file_num2] = future_one
            file_num2 += 1

    # Normalize LTWH values to range [0..1]
    past_all[:,:,0] = past_all[:,:,0] / 1240        # L
    past_all[:,:,1] = past_all[:,:,1] / 374         # T
    past_all[:,:,2] = past_all[:,:,2] / 1240        # W
    past_all[:,:,3] = past_all[:,:,3] / 374         # H
    future_all[:,:,0] = future_all[:,:,0] / 1240
    future_all[:,:,1] = future_all[:,:,1] / 374
    future_all[:,:,2] = future_all[:,:,2] / 1240
    future_all[:,:,3] = future_all[:,:,3] / 374

    return past_all, future_all, past_files, future_files


def generator_network(x, discrim_input_dim, base_n_count): 
    x = layers.Dense(base_n_count)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count*2)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count*4)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(4)(x)
    return x

def discriminator_network(x, discrim_input_dim, base_n_count):
    x = layers.Dense(base_n_count*4)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count*2)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x

def define_models_GAN(gen_input_dim, discrim_input_dim, base_n_count, type=None):
    generator_input_tensor = layers.Input(shape=(gen_input_dim, ))
    generated_image_tensor = generator_network(generator_input_tensor, discrim_input_dim, base_n_count)								# Input layer used here

    generated_or_real_image_tensor = layers.Input(shape=(discrim_input_dim,))
    
    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim, base_n_count)
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, discrim_input_dim, base_n_count)

    # This creates models which include the Input layer + hidden dense layers + output layer
    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')		# Input layer used here a second time
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    # 1. generator_model takes generator_input_tensor as input, returns a generated tensor
    # 2. discriminator_model takes generated tensor as input, returns a tensor which is the combined output
    # combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_output = discriminator_model(layers.concatenate([generator_input_tensor, generator_model(generator_input_tensor)]))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')
    
    return generator_model, discriminator_model, combined_model

def training_steps_GAN(model_components):
    
    [ model_name, starting_step, data_cols,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        nb_steps, batch_size, k_d, k_g,
                        log_interval, show, output_dir] = model_components 

    data_x, data_y, _, _ = get_data()
    combined_loss, disc_loss_generated, disc_loss_real, disc_loss = [], [], [], []

    # Store average discrim prediction for generated and real samples every epoch.
    avg_gen_pred, avg_real_pred = [], []


    if not os.path.exists(output_dir + 'weights\\'):
        os.makedirs(output_dir + 'weights\\')
    lossFile = open(output_dir + 'losses.txt', 'w')
    
    for i in range(1, nb_steps+1):
        K.set_learning_phase(1) # 1 = train

        # TRAIN DISCRIMINATOR on real and generated images
        #
        # k_d [1]: num of discriminator model updates per training step
        # batch_size [32]: the number of samples trained on during each step (if == len(data_x) then equivalent to 1 epoch?)
        for j in range(k_d):
            np.random.seed(i+j)
            x, y = get_data_batch(data_x, data_y, batch_size, seed=i+j)
            if i == 1 and j == 0: 
                print("x.shape: ", x.shape)
                print(x[0])
            
            g_z = generator_model.predict(x)
            if i == 1 and j == 0:
                print("g_z.shape: ", g_z.shape)
                print(g_z[0])
            g_z = np.concatenate((x, g_z), axis=1)
            if i == 1 and j == 0:
                print("new g_z.shape: ", g_z.shape)
                print(g_z[0])
            
            ### TRAIN ON REAL (y = 1) w/ noise
            disc_real_results = discriminator_model.train_on_batch(y, np.random.uniform(low=0.999, high=1.0, size=batch_size))      # 0.7, 1.2 GANs need noise to prevent loss going to zero

            ### TRAIN ON GENERATED (y = 0) w/ noise
            disc_gen_results = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size))    # 0.0, 0.3
            d_l = 0.5 * np.add(disc_real_results, disc_gen_results)

        disc_loss_real.append(disc_real_results)
        disc_loss_generated.append(disc_gen_results)
        disc_loss.append(d_l)
        
        # TRAIN GENERATOR on real inputs and outputs
        #
        # k_g [1]: num of generator model updates per training step
        for j in range(k_g):
            np.random.seed(i+j)
            x, y = get_data_batch(data_x, data_y, batch_size, seed=i+j)
            
            ### TRAIN (y = 1) bc want pos feedback for tricking discrim (want discrim to output 1)
            comb_results = combined_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size))    # 0.7, 1.2

        combined_loss.append(comb_results)

        # SAVE WEIGHTS / PLOT IMAGES
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            lossFile.write('Step: {} of {}.\n'.format(i, starting_step + nb_steps))
            K.set_learning_phase(0) # 0 = test

            # half learning rate every 5 epochs
            if not i % (log_interval*5): # UPDATE LEARNING RATE
                # They all share an optimizer, so this decreases the lr for all models
                K.set_value(generator_model.optimizer.lr, K.get_value(generator_model.optimizer.lr) / 2)
                print('~~~~~~~~~~~~~~~DECREMENTING lr to: ', K.get_value(generator_model.optimizer.lr), ", ", K.get_value(discriminator_model.optimizer.lr))

                        
            # LOSS SUMMARIES  
            print('lrs: '+ str(K.get_value(generator_model.optimizer.lr)) + ', ' + str(K.get_value(discriminator_model.optimizer.lr)) + ', ' + str(K.get_value(combined_model.optimizer.lr)))

            print('D_loss_gen: {}.\tD_loss_real: {}.'.format(disc_loss_generated[-1], disc_loss_real[-1]))
            lossFile.write('D_loss_gen: {}.\tD_loss_real: {}.\n'.format(disc_loss_generated[-1], disc_loss_real[-1]))

            print('G_loss: {}.\t\tD_loss: {}.'.format(combined_loss[-1], disc_loss[-1]))
            lossFile.write('G_loss: {}.\t\t\tD_loss: {}.\n'.format(combined_loss[-1], disc_loss[-1]))

            # if starting_step+nb_steps - i < log_interval*4:
            a_g_p, a_r_p = test_discrim(generator_model, discriminator_model, combined_model)
            print('avg_gen_pred: {}.\tavg_real_pred: {}.\n'.format(a_g_p, a_r_p))
            lossFile.write('avg_gen_pred: {}.\tavg_real_pred: {}.\n\n'.format(a_g_p, a_r_p))

            avg_gen_pred.append(a_g_p)
            avg_real_pred.append(a_r_p)
            
            # SAVE MODEL CHECKPOINTS
            model_checkpoint_base_name = output_dir + 'weights\\{}_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('gen', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discrim', i))
    
    return [combined_loss, disc_loss_generated, disc_loss_real, disc_loss, avg_gen_pred, avg_real_pred]

def get_model(data_cols, generator_model_path = None, discriminator_model_path = None, loss_pickle_path = None, seed=0, lr=5e-4):
    gen_input_dim = 40 # (32) needs to be ~discrim_input_dim
    base_n_count = 128
    show = True

    np.random.seed(seed)
    discrim_input_dim = len(data_cols)
    
    # Define network models.
    K.set_learning_phase(1)  # 1 = train
    generator_model, discriminator_model, combined_model = define_models_GAN(gen_input_dim, discrim_input_dim, base_n_count)
    
    adam = optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.trainable = False  # Freeze discriminator weights in combined model (we want to improve model by improving generator, rather than making the discriminator worse)
    combined_model.compile(optimizer=adam, loss='binary_crossentropy')
    
    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())
    
    # LOAD WEIGHTS (and previous loss logs) IF PROVIDED
    if loss_pickle_path:
        print('Loading loss pickles')
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

    return generator_model, discriminator_model, combined_model

def test_model(generator_model, discriminator_model, combined_model, model_name):
    """Test model on a hand-picked sample from the data set."""
    data_x, data_y, files_x, files_y = get_data()

    data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\bounding box images\\'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    test_x_pretty = data_x[20000]
    test_y_pretty = data_y[20000]
    print("file_x: ", files_x[20000])
    print("file_x: ", files_y[20000])
    
    test_x = np.reshape(test_x_pretty, (1, -1))
    test_y = np.reshape(test_y_pretty, (1, -1))

    test_g_z = generator_model.predict(test_x)
    test_g_z = np.concatenate((test_x, test_g_z), axis=1)
    test_g_z_pretty = np.reshape(test_g_z, (11,4))

    dpred_real = discriminator_model.predict(test_y)
    dpred_gen = discriminator_model.predict(test_g_z)
    print("dpred_real: ", dpred_real," dpred_gen: ", dpred_gen)

    # Undo normalization.
    test_g_z_pretty[:,0] = test_g_z_pretty[:,0] * 1240  # L
    test_g_z_pretty[:,1] = test_g_z_pretty[:,1] * 374   # T
    test_g_z_pretty[:,2] = test_g_z_pretty[:,2] * 1240  # W
    test_g_z_pretty[:,3] = test_g_z_pretty[:,3] * 374   # H
    test_y_pretty[:,0] = test_y_pretty[:,0] * 1240
    test_y_pretty[:,1] = test_y_pretty[:,1] * 374
    test_y_pretty[:,2] = test_y_pretty[:,2] * 1240
    test_y_pretty[:,3] = test_y_pretty[:,3] * 374

    # # Log results.
    # realfile = open(data_dir+'real.txt', 'w')
    # genfile = open(data_dir+'gen.txt', 'w')
    # realfile.write("%s\n" % test_y_pretty[10])
    # genfile.write("%s\n" % test_g_z_pretty[10])

    # Draw Results.
    frames = ['000040.png']
    print("test_g_z_pretty: ",test_g_z_pretty)
    print("test_y_pretty: ",test_y_pretty)
    drawFrameRects('0016', frames[0], test_g_z_pretty, isGen=True, folder_dir=data_dir)
    drawFrameRects('0016', frames[0], test_y_pretty, isGen=False, folder_dir=data_dir)

    return


def test_model_multiple(generator_model, discriminator_model, combined_model, model_name):
    """Test model on a hand-picked set of samples from the data set."""
    data_x, data_y, files_x, files_y = get_data()

    data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\bounding box images\\'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    test_set = [0, 1234, 2345, 3456, 10000, 15000, 19999, 20000]

    for i in test_set:
        regex_str = 'F:\\\\Car data\\\\label_02_extracted\\\\past_1obj_LTWH\\\\(.+?)\\\\past.*?_objId(.+?)_frameNum(.+?)\.txt'
        m = re.search(regex_str, files_x[i])
        folder = m.group(1)
        objId = m.group(2)
        frame = m.group(3).zfill(6)
        print(folder, ' ', frame)

        test_x_pretty = data_x[i]
        test_y_pretty = data_y[i]
        
        test_x = np.reshape(test_x_pretty, (1, -1))
        test_y = np.reshape(test_y_pretty, (1, -1))

        test_g_z = generator_model.predict(test_x)
        test_g_z = np.concatenate((test_x, test_g_z), axis=1)
        test_g_z_pretty = np.reshape(test_g_z, (11,4))

        dpred_real = discriminator_model.predict(test_y)
        dpred_gen = discriminator_model.predict(test_g_z)
        print("dpred_real: ", dpred_real," dpred_gen: ", dpred_gen)

        # Undo normalization.
        test_g_z_pretty[:,0] = test_g_z_pretty[:,0] * 1240
        test_g_z_pretty[:,1] = test_g_z_pretty[:,1] * 374
        test_g_z_pretty[:,2] = test_g_z_pretty[:,2] * 1240
        test_g_z_pretty[:,3] = test_g_z_pretty[:,3] * 374
        test_y_pretty[:,0] = test_y_pretty[:,0] * 1240
        test_y_pretty[:,1] = test_y_pretty[:,1] * 374
        test_y_pretty[:,2] = test_y_pretty[:,2] * 1240
        test_y_pretty[:,3] = test_y_pretty[:,3] * 374

        # Draw Results.
        drawFrameRects(folder, frame, objId, test_g_z_pretty, isGen=True, folder_dir=data_dir)
        drawFrameRects(folder, frame, objId, test_y_pretty, isGen=False, folder_dir=data_dir)

    return

def test_discrim(generator_model, discriminator_model, combined_model):
    """Test the discriminator by having it produce a realness score for generated and target images in a sample set."""
    data_x, data_y, files_x, files_y = get_data()
    data_x, data_y = get_data_batch(data_x, data_y, 590, seed=7)
    gen_correct = 0
    gen_incorrect = 0
    gen_unsure = 0

    real_correct = 0
    real_incorrect = 0
    real_unsure = 0

    discrim_outs_gen = np.zeros(shape=len(data_x))
    discrim_outs_real = np.zeros(shape=len(data_x))

    for i in range(len(data_x)):
        test_x_pretty = data_x[i]
        test_y_pretty = data_y[i]
        test_x = np.reshape(test_x_pretty, (1, -1))
        test_y = np.reshape(test_y_pretty, (1, -1))
        test_g_z = generator_model.predict(test_x)
        test_g_z = np.concatenate((test_x, test_g_z), axis=1)
        # test_g_z_pretty = np.reshape(test_g_z, (1,4))

        dpred_real = discriminator_model.predict(test_y)
        dpred_gen = discriminator_model.predict(test_g_z)
        discrim_outs_gen[i] = dpred_gen
        discrim_outs_real[i] = dpred_real

        if dpred_gen == 1.0:
            gen_incorrect += 1
        elif dpred_gen == 0.0:
            gen_correct += 1
        else:
            gen_unsure += 1

        if dpred_real == 1.0:
            real_correct += 1
        elif dpred_real == 0.0:
            real_incorrect += 1
        else:
            real_unsure += 1
    
    avg_gen_pred = np.average(discrim_outs_gen)
    avg_real_pred = np.average(discrim_outs_real)

    # print("gen_correct: ", gen_correct," gen_incorrect: ", gen_incorrect, " gen_unsure: ", gen_unsure, " avg_output: ", avg_gen_pred)
    # print("real_correct: ", real_correct," real_incorrect: ", real_incorrect, " real_unsure: ", real_unsure, " avg_output: ", avg_real_pred)
    return avg_gen_pred, avg_real_pred
