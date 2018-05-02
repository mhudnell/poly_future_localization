
import gan_1obj
import numpy as np
import matplotlib.pyplot as plt

data_cols = ['L', 'T', 'R', 'B']


## TRAINING ##
# CREATE NEW MODEL
lr = .002 # .0005 # 5e-4, 5e-5
generator_model, discriminator_model, combined_model = gan_1obj.getModel(data_cols, lr=lr)

# DEFINE TRAINING PARAMS
label_cols = []
label_dim = 0
log_interval = 274 # ~1 epoch (35082 / 32 =~ 1096) # 50, 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
epochs = 10
nb_steps = log_interval*epochs # 50000 # Add one for logging of the last interval
batch_size = 128 # 128, 64
k_d = 3  # 1 number of discriminator network updates per adversarial training step
k_g = 1  # 1 number of generator network updates per adversarial training step

starting_step = 0
cache_prefix = 'maxGAN1obj_bs{}_lr{}_kd{}_kg{}_steps{}_expDecay5'.format(batch_size, lr, k_d, k_g, nb_steps)
# cache_prefix = 'maxGAN_bs{}_lr{}_kd{}_kg{}'.format(batch_size, lr, k_d, k_g)
data_dir = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\'+cache_prefix+'\\'
show = True

model_components = [ cache_prefix, starting_step,
                    data_cols, label_cols, label_dim,
                    generator_model, discriminator_model, combined_model,
                    nb_steps, batch_size, k_d, k_g,
                    log_interval, data_dir, show]
    
[combined_loss, disc_loss_generated, disc_loss_real, disc_loss, avg_gen_pred, avg_real_pred] = gan_1obj.training_steps_GAN(model_components)

# PLOT LOSS
x = np.arange(nb_steps)
fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(x, disc_loss_generated, label='discrim loss gen')
ax1.plot(x, disc_loss_real, label='discrim loss real')
ax1.plot(x, disc_loss, label='discrim loss')
ax1.plot(x, combined_loss, label='generator loss')
ax1.legend(loc=1)
fig.suptitle(cache_prefix, fontsize=20)
plt.xlabel('number of steps', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.savefig('loss plots\\' + cache_prefix + '.png')

# PLOT DISCRIM PREDICTIONS

x = np.arange(epochs)
fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(x, avg_gen_pred, label='avg_gen_pred')
ax1.plot(x, avg_real_pred, label='avg_real_pred')

ax1.legend(loc=1)
fig.suptitle(cache_prefix, fontsize=20)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('avg prediction', fontsize=16)

plt.savefig('discrim plots\\' + cache_prefix + '.png')