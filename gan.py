import numpy as np

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import os

# def get_data_batch(train, batch_size, seed=0):
#     # # random sampling - some samples will have excessively low or high sampling, but easy to implement
#     # np.random.seed(seed)
#     # x = train.loc[ np.random.choice(train.index, batch_size) ].values
    
#     # iterate through shuffled indices, so every sample gets covered evenly
#     start_i = (batch_size * seed) % len(train)
#     stop_i = start_i + batch_size
#     shuffle_seed = (batch_size * seed) // len(train)
#     np.random.seed(shuffle_seed)
#     train_ix = np.random.choice( list(train.index), replace=False, size=len(train) ) # wasteful to shuffle every time
#     train_ix = list(train_ix) + list(train_ix) # duplicate to cover ranges past the end of the set
#     x = train.loc[ train_ix[ start_i: stop_i ] ].values
    
#     return np.reshape(x, (batch_size, -1) )

'''
Returns x and y (past and future) for batch_size # of samples
Data_x and data_y must be the same length and samples in these arrays should be in the same order (i.e. data_y[i] corresponds to the 'future' positions of data_x[i])
'''
def max_get_data_batch(data_x, data_y, batch_size, seed=0):
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
def max_get_data():
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


def adversarial_training_GAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

    [past_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ] = arguments
    
    np.random.seed(seed)     # set random seed
    
    pred_dim = len(data_cols)
    print('pred_dim: ', pred_dim)
    print('data_cols: ', data_cols)
    
    label_dim = 0
    with_class = False
    # if len(label_cols) > 0: 
    #     with_class = True
    #     label_dim = len(label_cols)
    #     print('label_dim: ', label_dim)
    #     print('label_cols: ', label_cols)
    
    # define network models
    
    K.set_learning_phase(1) # 1 = train
    
    cache_prefix = 'GAN'
    generator_model, discriminator_model, combined_model = define_models_GAN(past_dim, pred_dim, base_n_count)
    
    # compile models

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss='binary_crossentropy')
    
    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
    
    # if loss_pickle_path:
    #     print('Loading loss pickles')
    #     [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    # if generator_model_path:
    #     print('Loading generator model')
    #     generator_model.load_weights(generator_model_path, by_name=True)
    # if discriminator_model_path:
    #     print('Loading discriminator model')
    #     discriminator_model.load_weights(discriminator_model_path, by_name=True)

    model_components = [ cache_prefix, with_class, starting_step,
                        train, data_cols, pred_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        past_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ]
        
    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_GAN(model_components)
    # print('combined_loss: ', combined_loss)
    # print('disc_loss_generated: ', disc_loss_generated)
    # print('disc_loss_real: ', disc_loss_real)
    # print('xgb_losses: ', xgb_losses)


def training_steps_GAN(model_components):
    
    [ cache_prefix, with_class, starting_step,
                        train, data_cols, pred_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        past_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ] = model_components 

    data_x, data_y, _, _ = max_get_data()
    
    for i in range(starting_step, starting_step+nb_steps):
        K.set_learning_phase(1) # 1 = train

        # train the discriminator
        for j in range(k_d):
            np.random.seed(i+j)
            # z = np.random.normal(size=(batch_size, past_dim))
            # x = get_data_batch(train, batch_size, seed=i+j)
            x, y = max_get_data_batch(data_x, data_y, batch_size, seed=i+j)
            
            g_z = generator_model.predict(x)
#             x = np.vstack([x,g_z]) # code to train the discriminator on real and generated data at the same time, but you have to run network again for separate losses
#             classes = np.hstack([np.zeros(batch_size),np.ones(batch_size)])
#             d_l_r = discriminator_model.train_on_batch(x, classes)
            
            # TRAIN ON REAL (y = 1)
            d_l_r = discriminator_model.train_on_batch(y, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
            # TRAIN ON GENERATED (y = 0)
            d_l_g = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size)) # 0.0, 0.3 # GANs need noise to prevent loss going to zero

            # d_l_r = discriminator_model.train_on_batch(x, np.ones(batch_size)) # without noise
            # d_l_g = discriminator_model.train_on_batch(g_z, np.zeros(batch_size)) # without noise
        disc_loss_real.append(d_l_r)
        disc_loss_generated.append(d_l_g)
        
        # train the generator
        for j in range(k_g):
            np.random.seed(i+j)
            # # GENERATE RANDOM INPUT FOR THE GENERATOR NETWORK
            # z = np.random.normal(size=(batch_size, past_dim))
            x, y = max_get_data_batch(data_x, data_y, batch_size, seed=i+j)
            
            # loss = combined_model.train_on_batch(z, np.ones(batch_size)) # without noise
            # loss = combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
            loss = combined_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size))
        combined_loss.append(loss)
        
        # # Determine xgb loss each step, after training generator and discriminator
        # if not i % 10: # 2x faster than testing each step...
        #     K.set_learning_phase(0) # 0 = test
        #     test_size = 492 # test using all of the actual fraud data
        #     x = get_data_batch(train, test_size, seed=i)
        #     z = np.random.normal(size=(test_size, past_dim))
        #     if with_class:
        #         labels = x[:,-label_dim:]
        #         g_z = generator_model.predict([z, labels])
        #     else:
        #         g_z = generator_model.predict(z)
        #     xgb_loss = CheckAccuracy( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, pred_dim=pred_dim )
        #     xgb_losses = np.append(xgb_losses, xgb_loss)

        # # Saving weights and plotting images
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            K.set_learning_phase(0) # 0 = test
                        
            # loss summaries      
            # print( 'Losses: G, D Gen, D Real, Xgb: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1], xgb_losses[-1]) )
            # print( 'D Real - D Gen: {:.4f}'.format(disc_loss_real[-1]-disc_loss_generated[-1]) )
            print('Generator model loss: {}.'.format(combined_loss[-1]))
            print('Discriminator model loss gen: {}.'.format(disc_loss_generated[-1]))
            print('Discriminator model loss real: {}.'.format(disc_loss_real[-1]))
            # print('xgboost accuracy: {}'.format(xgb_losses[-1]) )
            
        #     if show:
        #         PlotData( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, pred_dim=pred_dim, 
        #                     save=False, prefix= data_dir + cache_prefix + '_' + str(i) )
            
            # save model checkpoints
            model_checkpoint_base_name = data_dir + cache_prefix + '_{}_model_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))
        #     pickle.dump([combined_loss, disc_loss_generated, disc_loss_real, xgb_losses], 
        #         open( data_dir + cache_prefix + '_losses_step_{}.pkl'.format(i) ,'wb'))

    # test_x, test_y = max_get_data_batch(data_x, data_y, 1, seed=0)
    test_x = data_x[0]
    test_y = data_y[0]
    test_x = np.reshape(test_x, (1, -1) )
    test_g_z = generator_model.predict(test_x)

    print(test_y)
    print(test_g_z)
    
    return [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]

def createModel():
    past_dim = 200 # 32 # needs to be ~pred_dim
    base_n_count = 128 # 128

    nb_steps = 500 + 1 # 50000 # Add one for logging of the last interval
    batch_size = 128 # 64

    k_d = 1  # number of critic network updates per adversarial training step
    k_g = 1  # number of generator network updates per adversarial training step
    critic_pre_train_steps = 100 # 100  # number of steps to pre-train the critic before starting adversarial training
    log_interval = 50 # 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
    learning_rate = 5e-4 # 5e-5
    # data_dir = '/F/Car data/kitti/data_tracking/training/label_02'
    data_dir = 'C:\\Users\\Max\\Research\\weights\\'
    generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None

    # show = False
    show = True

    # train = fraud_w_classes.copy().reset_index(drop=True) # fraud only with labels from classification

    # train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
    # label_cols = [ i for i in train.columns if 'Class' in i ]
    # data_cols = [ i for i in train.columns if i not in label_cols ]
    # data_cols = ['L1', 'T1', 'R1', 'B1', 'L2', 'T2', 'R2', 'B2', 'L3', 'T3', 'R3', 'B3', 'L4', 'T4', 'R4', 'B4', 'L5', 'T5', 'R5', 'B5']
    data_cols = []
    for frame in range(1,6):        # currently produces the wrong order
        for char in ['L', 'T', 'R', 'B']:
            for obj in range(1,6):
                data_cols.append('f' + str(frame) + char + str(obj))
    # train[ data_cols ] = train[ data_cols ] / 10 # scale to random noise size, one less thing to learn
    # train_no_label = train[ data_cols ]
    train_no_label = None

    learning_rate = 5e-4 # 5e-5
    arguments = [past_dim, nb_steps, batch_size, 
                 k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ]

    adversarial_training_GAN(arguments, train_no_label, data_cols ) # GAN

createModel()