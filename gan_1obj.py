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

'''
Returns x and y (past and future) for batch_size # of samples
Data_x and data_y must be the same length and samples in these arrays should be in the same order (i.e. data_y[i] corresponds to the 'future' positions of data_x[i])
'''
def get_data_batch(data_x, data_y, batch_size, seed=0):
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
    # print('x.shape: ', x.shape)
    # print('y.shape: ', y.shape)
    
    return np.reshape(x, (batch_size, -1) ), np.reshape(y, (batch_size, -1) )

'''
TODO: make sure future_all[i] corresponds to the 'future' positions of past_all[i]
'''
def get_data():
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

    # NORMALIZE
    # past_all[:,:,0] = past_all[:,:,0] / (1240 / 2) - 1
    # past_all[:,:,1] = past_all[:,:,1] / (374 / 2) - 1
    # past_all[:,:,2] = past_all[:,:,2] / (1240 / 2) - 1
    # past_all[:,:,3] = past_all[:,:,3] / (374 / 2) - 1
    # future_all[:,:,0] = future_all[:,:,0] / (1240 / 2) - 1
    # future_all[:,:,1] = future_all[:,:,1] / (374 / 2) - 1
    # future_all[:,:,2] = future_all[:,:,2] / (1240 / 2) - 1
    # future_all[:,:,3] = future_all[:,:,3] / (374 / 2) - 1
    past_all[:,:,0] = past_all[:,:,0] / 1240
    past_all[:,:,1] = past_all[:,:,1] / 374
    past_all[:,:,2] = past_all[:,:,2] / 1240
    past_all[:,:,3] = past_all[:,:,3] / 374
    future_all[:,:,0] = future_all[:,:,0] / 1240
    future_all[:,:,1] = future_all[:,:,1] / 374
    future_all[:,:,2] = future_all[:,:,2] / 1240
    future_all[:,:,3] = future_all[:,:,3] / 374

    # print("FILE PATHS")
    # print(past_files[-20:])
    # print(future_files[-20:])
    return past_all, future_all, past_files, future_files

# def generator_network(x, discrim_input_dim, base_n_count): 
#     x = layers.Dense(base_n_count, activation='relu')(x)
#     x = layers.Dense(base_n_count*2, activation='relu')(x)
#     x = layers.Dense(base_n_count*4, activation='relu')(x)
#     # x = layers.Dense(discrim_input_dim)(x)    
#     x = layers.Dense(4)(x)
#     return x    # returns a tensor

# def discriminator_network(x, discrim_input_dim, base_n_count):
#     x = layers.Dense(base_n_count*4, activation='relu')(x)
#     # x = layers.Dropout(0.1)(x)
#     x = layers.Dense(base_n_count*2, activation='relu')(x)
#     # x = layers.Dropout(0.1)(x)
#     x = layers.Dense(base_n_count, activation='relu')(x)
#     x = layers.Dense(1, activation='sigmoid')(x)
#     # x = layers.Dense(1)(x)
#     return x    # returns a tensor

def generator_network(x, discrim_input_dim, base_n_count): 
    # x = layers.Dense(base_n_count, activation='relu')(x)
    # x = layers.Dense(base_n_count*2, activation='relu')(x)
    # x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(base_n_count)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count*2)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count*4)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(4)(x)
    return x 	# returns a tensor

def discriminator_network(x, discrim_input_dim, base_n_count):
    # x = layers.Dense(base_n_count*4, activation='relu')(x)
    # # x = layers.Dropout(0.1)(x)
    # x = layers.Dense(base_n_count*2, activation='relu')(x)
    # # x = layers.Dropout(0.1)(x)
    # x = layers.Dense(base_n_count, activation='relu')(x)
    # x = layers.Dense(1, activation='sigmoid')(x)

    x = layers.Dense(base_n_count*4)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count*2)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(base_n_count)(x)
    x = layers.LeakyReLU(0.1)(x)
    # x = layers.Dense(1)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x 	# returns a tensor

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

    # 1. Generator_model takes generator_input_tensor as input, returns a generated tensor
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
    # disc_acc_real, disc_acc_generated, combined_acc = [], [], []
    avg_gen_pred, avg_real_pred = [], []    # store average discrim prediction for generated and real samples every epoch.


    if not os.path.exists(output_dir + 'weights\\'):
        os.makedirs(output_dir + 'weights\\')
    lossFile = open(output_dir + 'losses.txt', 'w')
    
    for i in range(1, nb_steps+1):
        K.set_learning_phase(1) # 1 = train

        ''' TRAIN DISCRIMINATOR on real and generated images
        #   k_d [1]: num of discriminator model updates per training step
        #   batch_size [32]: the number of samples trained on during each step (if == len(data_x) then equivalent to 1 epoch?)
        #
        '''
        for j in range(k_d):
        # if i==0 or combined_loss[-1] < 8:
            np.random.seed(i+j)
            x, y = get_data_batch(data_x, data_y, batch_size, seed=i+j)
            if i == 1 and j == 0: 
                print("x.shape: ", x.shape)
                print(x[0])
            
            # z = np.random.normal(size=(batch_size, 200))  # TRAIN FROM NOISE
            # g_z = generator_model.predict(z)              # TRAIN FROM NOISE
            g_z = generator_model.predict(x)
            if i == 1 and j == 0:
                print("g_z.shape: ", g_z.shape)
                print(g_z[0])
            g_z = np.concatenate((x, g_z), axis=1)
            if i == 1 and j == 0:
                print("new g_z.shape: ", g_z.shape)
                print(g_z[0])
            
            ### TRAIN ON REAL (y = 1) w/ noise
            disc_real_results = discriminator_model.train_on_batch(y, np.random.uniform(low=0.999, high=1.0, size=batch_size))      # 0.7, 1.2 # GANs need noise to prevent loss going to zero # 0.999, 1.0
            # disc_real_results = discriminator_model.train_on_batch(y, np.random.uniform(low=0.0, high=0.001, size=batch_size))  # FLIPPED
            # disc_real_results = discriminator_model.train_on_batch(y, np.ones(batch_size))

            ### TRAIN ON GENERATED (y = 0) w/ noise
            disc_gen_results = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size))    # 0.0, 0.3 # GANs need noise to prevent loss going to zero # 0.0, 0.001
            # disc_gen_results = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # FLIPPED
            # disc_gen_results = discriminator_model.train_on_batch(g_z, np.zeros(batch_size))
            d_l = 0.5 * np.add(disc_real_results, disc_gen_results)

        disc_loss_real.append(disc_real_results)
        # disc_acc_real.append(disc_real_results[1])
        disc_loss_generated.append(disc_gen_results)
        # disc_acc_generated.append(disc_gen_results[1])
        disc_loss.append(d_l)
        
        ''' TRAIN GENERATOR on real inputs and outputs
        #   k_g [1]: num of generator model updates per training step
        #   
        '''
        for j in range(k_g):
            np.random.seed(i+j)
            x, y = get_data_batch(data_x, data_y, batch_size, seed=i+j)
            # z = np.random.normal(size=(batch_size, 200))  # TRAIN FROM NOISE
            
            ### TRAIN (y = 1) bc want pos feedback for tricking discrim (want discrim to output 1)
            comb_results = combined_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size))    # 0.7, 1.2 # GANs need noise to prevent loss going to zero # 0.999, 1.0
            # comb_results = combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=batch_size))  # TRAIN FROM NOISE
            # comb_results = combined_model.train_on_batch(z, np.random.uniform(low=0.0, high=0.001, size=batch_size))    # FLIPPED
            # comb_results = combined_model.train_on_batch(x, np.ones(batch_size))

        combined_loss.append(comb_results)
        # combined_acc.append(comb_results[1])

        # SAVE WEIGHTS / PLOT IMAGES
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            lossFile.write('Step: {} of {}.\n'.format(i, starting_step + nb_steps))
            K.set_learning_phase(0) # 0 = test

            # half learning rate every 5 epochs
            if not i % (log_interval*5): # UPDATE LEARNING RATE
                # They all share an optimizer, so this decreases the lr for all models
                K.set_value(generator_model.optimizer.lr, K.get_value(generator_model.optimizer.lr) / 2)
                # K.set_value(discriminator_model.optimizer.lr, K.get_value(discriminator_model.optimizer.lr) / 2)
                print('~~~~~~~~~~~~~~~DECREMENTING lr to: ', K.get_value(generator_model.optimizer.lr), ", ", K.get_value(discriminator_model.optimizer.lr))

                        
            # LOSS SUMMARIES  
            print('lrs: '+ str(K.get_value(generator_model.optimizer.lr)) + ', ' + str(K.get_value(discriminator_model.optimizer.lr)) + ', ' + str(K.get_value(combined_model.optimizer.lr)))

            print('D_loss_gen: {}.\tD_loss_real: {}.'.format(disc_loss_generated[-1], disc_loss_real[-1]))
            lossFile.write('D_loss_gen: {}.\tD_loss_real: {}.\n'.format(disc_loss_generated[-1], disc_loss_real[-1]))

            print('G_loss: {}.\t\tD_loss: {}.'.format(combined_loss[-1], disc_loss[-1]))
            lossFile.write('G_loss: {}.\t\t\tD_loss: {}.\n'.format(combined_loss[-1], disc_loss[-1]))

            # if starting_step+nb_steps - i < log_interval*4:
            a_g_p, a_r_p = testDiscrim(generator_model, discriminator_model, combined_model)
            print('avg_gen_pred: {}.\tavg_real_pred: {}.\n'.format(a_g_p, a_r_p))
            lossFile.write('avg_gen_pred: {}.\tavg_real_pred: {}.\n\n'.format(a_g_p, a_r_p))

            avg_gen_pred.append(a_g_p)
            avg_real_pred.append(a_r_p)
            
            # if show:
            #     PlotData( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, discrim_input_dim=discrim_input_dim, 
            #                 save=False, prefix= output_dir + model_name + '_' + str(i) )
            
            # SAVE MODEL CHECKPOINTS
            model_checkpoint_base_name = output_dir + 'weights\\{}_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('gen', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discrim', i))
    
    return [combined_loss, disc_loss_generated, disc_loss_real, disc_loss, avg_gen_pred, avg_real_pred]

def getModel(data_cols, generator_model_path = None, discriminator_model_path = None, loss_pickle_path = None, seed=0, lr=5e-4):
    gen_input_dim = 40 # 32 # needs to be ~discrim_input_dim
    base_n_count = 128 # 128
    show = True

    np.random.seed(seed)     # set random seed
    discrim_input_dim = len(data_cols)
    
    # DEFINE NETWORK MODELS
    K.set_learning_phase(1) # 1 = train
    generator_model, discriminator_model, combined_model = define_models_GAN(gen_input_dim, discrim_input_dim, base_n_count)
    
    # COMPILE MODELS
    adam = optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.9)
    # adam_gen = optimizers.Adam(lr=.0005, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss='binary_crossentropy') # binary_crossentropy    # adam_gen
    discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.trainable = False   # freeze discriminator weights in combined model (we want to improve model by improving generator, rather than making the discriminator worse)
    combined_model.compile(optimizer=adam, loss='binary_crossentropy')   # adam_gen
    
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

'''
    Tests model on the first sample in the data set.
    TODO: implement
'''
def testModel(generator_model, discriminator_model, combined_model, model_name):
    data_x, data_y, files_x, files_y = get_data()

    # print(files_x[:10])
    # print(files_y[:10])

    # print(data_x[:,:,0])
    # print(data_x[:,:,2])

    data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\bb images\\'+model_name+'\\'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    test_x_pretty = data_x[20000]
    test_y_pretty = data_y[20000]
    print("file_x: ", files_x[20000])
    print("file_x: ", files_y[20000])
    
    test_x = np.reshape(test_x_pretty, (1, -1))
    test_y = np.reshape(test_y_pretty, (1, -1))

    # test_z = np.random.normal(size=(1, 200, ))    # SAMPLE FROM NOISE

    test_g_z = generator_model.predict(test_x)
    test_g_z = np.concatenate((test_x, test_g_z), axis=1)
    test_g_z_pretty = np.reshape(test_g_z, (11,4))

    dpred_real = discriminator_model.predict(test_y)
    dpred_gen = discriminator_model.predict(test_g_z)
    print("dpred_real: ", dpred_real," dpred_gen: ", dpred_gen)

    # UNDO NORMALIZATION
    # test_g_z_pretty[:,0] = (test_g_z_pretty[:,0] + 1) * (1240 / 2)
    # test_g_z_pretty[:,1] = (test_g_z_pretty[:,1] + 1) * (374 / 2)
    # test_g_z_pretty[:,2] = (test_g_z_pretty[:,2] + 1) * (1240 / 2)
    # test_g_z_pretty[:,3] = (test_g_z_pretty[:,3] + 1) * (374 / 2)
    # test_y_pretty[:,0] = (test_y_pretty[:,0] + 1) * (1240 / 2)
    # test_y_pretty[:,1] = (test_y_pretty[:,1] + 1) * (374 / 2)
    # test_y_pretty[:,2] = (test_y_pretty[:,2] + 1) * (1240 / 2)
    # test_y_pretty[:,3] = (test_y_pretty[:,3] + 1) * (374 / 2)
    test_g_z_pretty[:,0] = test_g_z_pretty[:,0] * 1240
    test_g_z_pretty[:,1] = test_g_z_pretty[:,1] * 374
    test_g_z_pretty[:,2] = test_g_z_pretty[:,2] * 1240
    test_g_z_pretty[:,3] = test_g_z_pretty[:,3] * 374
    test_y_pretty[:,0] = test_y_pretty[:,0] * 1240
    test_y_pretty[:,1] = test_y_pretty[:,1] * 374
    test_y_pretty[:,2] = test_y_pretty[:,2] * 1240
    test_y_pretty[:,3] = test_y_pretty[:,3] * 374

    # WRITE RESULTS TO FILE
    realfile = open(data_dir+'real.txt', 'w')
    genfile = open(data_dir+'gen.txt', 'w')
    # for item in test_y_pretty:
    #     realfile.write("%s\n" % item)
    # for item in test_g_z_pretty:
    #     genfile.write("%s\n" % item)
    realfile.write("%s\n" % test_y_pretty[10])
    genfile.write("%s\n" % test_g_z_pretty[10])

    # DRAW RESULTS
    frames = ['000040.png']
    print("test_g_z_pretty: ",test_g_z_pretty)
    print("test_y_pretty: ",test_y_pretty)
    drawFrameRects('0016', frames[0], test_g_z_pretty, isGen=True, folder_dir=data_dir)
    drawFrameRects('0016', frames[0], test_y_pretty, isGen=False, folder_dir=data_dir)

    return

'''
    Tests model on the first sample in the data set.
'''
def testModelMult(generator_model, discriminator_model, combined_model, model_name):
    data_x, data_y, files_x, files_y = get_data()

    data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\bb images\\'+model_name+'\\'
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

        # UNDO NORMALIZATION
        test_g_z_pretty[:,0] = test_g_z_pretty[:,0] * 1240
        test_g_z_pretty[:,1] = test_g_z_pretty[:,1] * 374
        test_g_z_pretty[:,2] = test_g_z_pretty[:,2] * 1240
        test_g_z_pretty[:,3] = test_g_z_pretty[:,3] * 374
        test_y_pretty[:,0] = test_y_pretty[:,0] * 1240
        test_y_pretty[:,1] = test_y_pretty[:,1] * 374
        test_y_pretty[:,2] = test_y_pretty[:,2] * 1240
        test_y_pretty[:,3] = test_y_pretty[:,3] * 374

        # # WRITE RESULTS TO FILE
        # realfile = open(data_dir+'real'+str(i)+'.txt', 'w')
        # genfile = open(data_dir+'gen'+str(i)+'.txt', 'w')
        # # for item in test_y_pretty:
        # #     realfile.write("%s\n" % item)
        # # for item in test_g_z_pretty:
        # #     genfile.write("%s\n" % item)
        # realfile.write("%s\n" % test_y_pretty[10])
        # genfile.write("%s\n" % test_g_z_pretty[10])

        # DRAW RESULTS
        drawFrameRects(folder, frame, objId, test_g_z_pretty, isGen=True, folder_dir=data_dir)
        drawFrameRects(folder, frame, objId, test_y_pretty, isGen=False, folder_dir=data_dir)

    return

def testDiscrim(generator_model, discriminator_model, combined_model):
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
        # test_z = np.random.normal(size=(1, 200, ))    # SAMPLE FROM NOISE
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
    
    # print(discrim_outs_gen[5500:5700])
    # print(discrim_outs_real[5500:5700])
    # print(discrim_outs_gen)
    # print(discrim_outs_real)
    avg_gen_pred = np.average(discrim_outs_gen)
    avg_real_pred = np.average(discrim_outs_real)

    # print("gen_correct: ", gen_correct," gen_incorrect: ", gen_incorrect, " gen_unsure: ", gen_unsure, " avg_output: ", avg_gen_pred)
    # print("real_correct: ", real_correct," real_incorrect: ", real_incorrect, " real_unsure: ", real_unsure, " avg_output: ", avg_real_pred)

    return avg_gen_pred, avg_real_pred

# data_cols = []
# for frame in range(1,6):        # currently produces the wrong order
#     for char in ['L', 'T', 'R', 'B']:
#         for obj in range(1,6):
#             data_cols.append('f' + str(frame) + char + str(obj))

# ## TRAINING ##
# # CREATE NEW MODEL
# lr = .0001 # 5e-4, 5e-5
# generator_model, discriminator_model, combined_model = getModel(data_cols, lr=lr)

# # DEFINE TRAINING PARAMS
# label_cols = []
# label_dim = 0
# log_interval = 60 # ~ 1 epoch # 50, 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
# nb_steps = log_interval*10 + 1 # 50000 # Add one for logging of the last interval
# batch_size = 32 # 128, 64
# k_d = 3  # 1 number of discriminator network updates per adversarial training step
# k_g = 2  # 1 number of generator network updates per adversarial training step

# starting_step = 0
# model_name = 'noiseGAN_bs{}_lr{}_kd{}_kg{}'.format(batch_size, lr, k_d, k_g)
# # model_name = 'maxGAN_bs{}_lr{}_kd{}_kg{}'.format(batch_size, lr, k_d, k_g)
# data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\'+model_name+'\\'
# show = True

# model_components = [ model_name, starting_step,
#                     data_cols, label_cols, label_dim,
#                     generator_model, discriminator_model, combined_model,
#                     nb_steps, batch_size, k_d, k_g,
#                     log_interval, data_dir, show]
    
# [combined_loss, disc_loss_generated, disc_loss_real, disc_loss, combined_acc, disc_acc_generated, disc_acc_real] = training_steps_GAN(model_components)

# # PLOT LOSS
# x = np.arange(nb_steps)
# fig = plt.figure(figsize=(11,8))
# ax1 = fig.add_subplot(111)

# ax1.plot(x, disc_loss_generated, label='discrim loss gen')
# ax1.plot(x, disc_loss_real, label='discrim loss real')
# ax1.plot(x, disc_loss, label='discrim loss')
# ax1.plot(x, combined_loss, label='generator loss')
# ax1.legend(loc=1)
# fig.suptitle(model_name, fontsize=20)
# plt.xlabel('number of steps', fontsize=18)
# plt.ylabel('loss', fontsize=16)

# plt.savefig('loss plots\\' + model_name + '_loss_plot.png')

# ## TESTING ##
# # LOAD MODEL
# # generator_model, discriminator_model, combined_model = getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__gen_weights_step_100.h5',
# #                                                                      discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__discrim_weights_step_100.h5')

# generator_model, discriminator_model, combined_model = getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\noiseGAN_bs32_lr0.0001_kd2_kg1\\gen_weights_step_600.h5',
#                                                                      discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\noiseGAN_bs32_lr0.0001_kd2_kg1\\discrim_weights_step_600.h5')

# testModel(generator_model, discriminator_model, combined_model)
# # testDiscrim(generator_model, discriminator_model, combined_model)