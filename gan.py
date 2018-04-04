import numpy as np
import matplotlib.pyplot as plt

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import os

from vis_tool import drawObjectRects

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
    past_all = np.empty([7703, 50, 4])
    past_files = []

    file_num1 = 0
    for path, subdirs, files in os.walk('F:\\Car data\\label_02_extracted\\past'):
        for name in files:
            fpath = os.path.join(path, name)
            past_files.append(fpath)
            f = open(fpath,'r')
            past_one = np.empty([50, 4])
            for i in range(50):
                line = f.readline()
                if not line:
                    break
                words = line.split()
                past_one[i] = [float(word) for word in words]
            past_all[file_num1] = past_one
            file_num1 += 1

    future_all = np.empty([7703, 25, 4])
    future_files = []

    file_num2 = 0
    for path, subdirs, files in os.walk('F:\\Car data\\label_02_extracted\\future'):
        for name in files:
            fpath = os.path.join(path, name)
            future_files.append(fpath)
            f = open(fpath,'r')
            future_one = np.empty([25, 4])
            for i in range(25):
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

def generator_network(x, pred_dim, base_n_count): 
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(pred_dim)(x)    
    return x 	# returns a tensor

def discriminator_network(x, pred_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    # x = layers.Dense(1)(x)
    return x 	# returns a tensor

def define_models_GAN(past_dim, pred_dim, base_n_count, type=None):
    generator_input_tensor = layers.Input(shape=(past_dim, ))
    generated_image_tensor = generator_network(generator_input_tensor, pred_dim, base_n_count)								# Input layer used here

    generated_or_real_image_tensor = layers.Input(shape=(pred_dim,))
    
    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim, base_n_count)
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, pred_dim, base_n_count)

    # This creates models which include the Input layer + hidden dense layers + output layer
    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')		# Input layer used here a second time
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    # 1. Generator_model takes generator_input_tensor as input, returns a generated tensor
    # 2. discriminator_model takes generated tensor as input, returns a tensor which is the combined output
    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')
    
    return generator_model, discriminator_model, combined_model

def training_steps_GAN(model_components):
    
    [ cache_prefix, starting_step, data_cols,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        nb_steps, batch_size, k_d, k_g,
                        log_interval, data_dir, show] = model_components 

    data_x, data_y, _, _ = get_data()

    combined_loss, disc_loss_generated, disc_loss_real, disc_loss = [], [], [], []
    disc_acc_real, disc_acc_generated, combined_acc = [], [], []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    lossFile = open(data_dir + 'losses.txt', 'w')
    
    for i in range(starting_step, starting_step+nb_steps):
        K.set_learning_phase(1) # 1 = train

        ''' TRAIN DISCRIMINATOR on generated images
        #
        #   k_d [1]: num of discriminator model updates per training step
        #   batch_size [128]: the number of samples trained on during each step (if == len(data_x) then equivalent to 1 epoch?)
        #
        '''
        for j in range(k_d):
        # if i==0 or combined_loss[-1] < 8:
            np.random.seed(i+j)
            x, y = get_data_batch(data_x, data_y, batch_size, seed=i+j)
            
            z = np.random.normal(size=(batch_size, 200))
            # g_z = generator_model.predict(z)
            g_z = generator_model.predict(x)
            
            # TRAIN ON REAL (y = 1) w/ noise
            disc_real_results = discriminator_model.train_on_batch(y, np.random.uniform(low=0.999, high=1.0, size=batch_size))      # 0.7, 1.2 # GANs need noise to prevent loss going to zero # 0.999, 1.0
            # disc_real_results = discriminator_model.train_on_batch(y, np.random.uniform(low=0.0, high=0.001, size=batch_size))  # FLIPPED
            # disc_real_results = discriminator_model.train_on_batch(y, np.ones(batch_size))

            # TRAIN ON GENERATED (y = 0) w/ noise
            disc_gen_results = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size))    # 0.0, 0.3 # GANs need noise to prevent loss going to zero # 0.0, 0.001
            # disc_gen_results = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # FLIPPED
            # disc_gen_results = discriminator_model.train_on_batch(g_z, np.zeros(batch_size))
            d_l = 0.5 * np.add(disc_real_results[0], disc_gen_results[0])

        disc_loss_real.append(disc_real_results[0])
        disc_acc_real.append(disc_real_results[1])
        disc_loss_generated.append(disc_gen_results[0])
        disc_acc_generated.append(disc_gen_results[1])
        disc_loss.append(d_l)
        
        ''' TRAIN GENERATOR on real inputs and outputs
        #   k_g [1]: num of generator model updates per training step
        #   
        #   
        '''
        for j in range(k_g):
            np.random.seed(i+j)
            x, y = get_data_batch(data_x, data_y, batch_size, seed=i+j)
            # z = np.random.normal(size=(batch_size, 200))
            
            # TRAIN (y = 1) bc want pos feedback for tricking discrim (want discrim to output 1)
            comb_results = combined_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size))    # 0.7, 1.2 # GANs need noise to prevent loss going to zero # 0.999, 1.0
            # comb_results = combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=batch_size))  # train from noise
            # comb_results = combined_model.train_on_batch(z, np.random.uniform(low=0.0, high=0.001, size=batch_size))    # FLIPPED
            # comb_results = combined_model.train_on_batch(x, np.ones(batch_size))

        combined_loss.append(comb_results[0])
        combined_acc.append(comb_results[1])

        # SAVE WEIGHTS / PLOT IMAGES
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            lossFile.write('Step: {} of {}.\n'.format(i, starting_step + nb_steps))
            K.set_learning_phase(0) # 0 = test
                        
            # LOSS SUMMARIES  
            # print( 'Losses: G, D Gen, D Real, Xgb: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1], xgb_losses[-1]) )
            # print( 'D Real - D Gen: {:.4f}'.format(disc_loss_real[-1]-disc_loss_generated[-1]) )
            print('D_loss_gen: {}.\tD_acc_gen: {}.'.format(disc_loss_generated[-1], disc_acc_generated[-1]))
            lossFile.write('D_loss_gen: {}.\tD_acc_gen: {}.\n'.format(disc_loss_generated[-1], disc_acc_generated[-1]))
            print('D_loss_real: {}.\tD_acc_real: {}.'.format(disc_loss_real[-1], disc_acc_real[-1]))
            lossFile.write('D_loss_real: {}.\tD_acc_real: {}.\n'.format(disc_loss_real[-1], disc_acc_real[-1]))
            print('D_loss: {}.'.format(disc_loss[-1]))
            lossFile.write('D_loss: {}.\n'.format(disc_loss[-1]))
            print('G_loss: {}.\t\tG_acc: {}.\n'.format(combined_loss[-1], combined_acc[-1]))
            lossFile.write('G_loss: {}.\t\t\tG_acc: {}.\n\n'.format(combined_loss[-1], combined_acc[-1]))
            # print('xgboost accuracy: {}'.format(xgb_losses[-1]) )
            
            # if show:
            #     PlotData( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, pred_dim=pred_dim, 
            #                 save=False, prefix= data_dir + cache_prefix + '_' + str(i) )
            
            # SAVE MODEL CHECKPOINTS
            # model_checkpoint_base_name = data_dir + cache_prefix + '_{}_weights_step_{}.h5'
            model_checkpoint_base_name = data_dir + '{}_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('gen', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discrim', i))
            # pickle.dump([combined_loss, disc_loss_generated, disc_loss_real, xgb_losses], 
            #     open( data_dir + cache_prefix + '_losses_step_{}.pkl'.format(i) ,'wb'))

            # PRINT GENERATED EXAMPLE
            # test_x = data_x[0]
            # test_x = np.reshape(test_x, (1, -1) )
            # test_g_z = generator_model.predict(test_x)
            # print(np.reshape(test_g_z, (1,25,4)))

    # test_x, test_y = get_data_batch(data_x, data_y, 1, seed=0)
    # test_y = data_y[0]

    # print(test_y)
    
    return [combined_loss, disc_loss_generated, disc_loss_real, disc_loss, combined_acc, disc_acc_generated, disc_acc_real]

def getModel(data_cols, generator_model_path = None, discriminator_model_path = None, loss_pickle_path = None, seed=0, lr=5e-4):
    past_dim = 200 # 32 # needs to be ~pred_dim
    base_n_count = 128 # 128
    show = True

    np.random.seed(seed)     # set random seed
    pred_dim = len(data_cols)
    
    # DEFINE NETWORK MODELS
    K.set_learning_phase(1) # 1 = train
    generator_model, discriminator_model, combined_model = define_models_GAN(past_dim, pred_dim, base_n_count)
    
    # COMPILE MODELS
    adam = optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss='mean_squared_error') # binary_crossentropy
    discriminator_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    discriminator_model.trainable = False   # freeze discriminator weights in combined model (we want to improve model by improving generator, rather than making the discriminator worse)
    combined_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
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
'''
def testModel(generator_model, discriminator_model, combined_model):
    data_x, data_y, files_x, files_y = get_data()

    # print(data_x[:,:,0])
    # print(data_x[:,:,2])

    test_x_pretty = data_x[0]
    test_y_pretty = data_y[0]
    test_x = np.reshape(test_x_pretty, (1, -1))
    test_y = np.reshape(test_y_pretty, (1, -1))
    test_z = np.random.normal(size=(1, 200, ))

    test_g_z = generator_model.predict(test_z) # test_x
    test_g_z_pretty = np.reshape(test_g_z, (25,4))

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
    realfile = open('bb images\\real.txt', 'w')
    genfile = open('bb images\\gen.txt', 'w')
    for item in test_y_pretty:
        realfile.write("%s\n" % item)
    for item in test_g_z_pretty:
        genfile.write("%s\n" % item)

    # DRAW RESULTS
    frames = ['000010.png', '000011.png', '000012.png', '000013.png', '000014.png']
    for i in range(5):
        bb_i = i * 5
        drawObjectRects(frames[i], test_g_z_pretty[bb_i:bb_i+5], isGen=True)
        drawObjectRects(frames[i], test_y_pretty[bb_i:bb_i+5], isGen=False)

    return

def testDiscrim(generator_model, discriminator_model, combined_model):
    data_x, data_y, files_x, files_y = get_data()
    gen_correct = 0
    gen_incorrect = 0
    gen_unsure = 0

    real_correct = 0
    real_incorrect = 0
    real_unsure = 0

    discrim_outs_gen = np.zeros(shape=len(data_x))
    discrim_outs_real = np.zeros(shape=len(data_x))

    for i in range(len(data_x)):
        test_x_pretty = data_x[0]
        test_y_pretty = data_y[0]
        test_x = np.reshape(test_x_pretty, (1, -1))
        test_y = np.reshape(test_y_pretty, (1, -1))
        test_z = np.random.normal(size=(1, 200, ))
        test_g_z = generator_model.predict(test_z) #test_x
        test_g_z_pretty = np.reshape(test_g_z, (25,4))

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
    
    print(discrim_outs_gen[5500:5700])
    print(discrim_outs_real[5500:5700])
    print("gen_correct: ", gen_correct," gen_incorrect: ", gen_incorrect, " gen_unsure: ", gen_unsure, " avg_output: ", np.average(discrim_outs_gen))
    print("real_correct: ", real_correct," real_incorrect: ", real_incorrect, " real_unsure: ", real_unsure, " avg_output: ", np.average(discrim_outs_real))

    return

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
# cache_prefix = 'noiseGAN_bs{}_lr{}_kd{}_kg{}'.format(batch_size, lr, k_d, k_g)
# # cache_prefix = 'maxGAN_bs{}_lr{}_kd{}_kg{}'.format(batch_size, lr, k_d, k_g)
# data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\'+cache_prefix+'\\'
# show = True

# model_components = [ cache_prefix, starting_step,
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
# fig.suptitle(cache_prefix, fontsize=20)
# plt.xlabel('number of steps', fontsize=18)
# plt.ylabel('loss', fontsize=16)

# plt.savefig('loss plots\\' + cache_prefix + '_loss_plot.png')

# ## TESTING ##
# # LOAD MODEL
# # generator_model, discriminator_model, combined_model = getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__gen_weights_step_100.h5',
# #                                                                      discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__discrim_weights_step_100.h5')

# generator_model, discriminator_model, combined_model = getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\noiseGAN_bs32_lr0.0001_kd2_kg1\\gen_weights_step_600.h5',
#                                                                      discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\noiseGAN_bs32_lr0.0001_kd2_kg1\\discrim_weights_step_600.h5')

# testModel(generator_model, discriminator_model, combined_model)
# # testDiscrim(generator_model, discriminator_model, combined_model)