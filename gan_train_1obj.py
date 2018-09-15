import gan_1obj
import numpy as np
import matplotlib.pyplot as plt

data_cols = []
for fNum in range(1,12):
    for char in ['L', 'T', 'W', 'H']:
        data_cols.append(char + str(fNum))

print("data_cols: ", data_cols)

## TRAINING ##
# CREATE NEW MODEL
lr = .0005 # .0005 # 5e-4, 5e-5
generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, lr=lr)

# DEFINE TRAINING PARAMS
label_cols = []
label_dim = 0
log_interval = 274 # ~1 epoch (35082 / 32 =~ 1096, 128: 174)  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
epochs = 10
nb_steps = log_interval*epochs # 50000 # Add one for logging of the last interval
batch_size = 128 # 128, 64
k_d = 2  # 1 number of discriminator network updates per adversarial training step
k_g = 1  # 1 number of generator network updates per adversarial training step

starting_step = 0
model_name = 'maxGAN_combLoss_bs{}_lr{}expDecay5_kd{}_kg{}_steps{}'.format(batch_size, lr, k_d, k_g, nb_steps)
output_dir = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\'
show = True

model_components = [model_name, starting_step,
                    data_cols, label_cols, label_dim,
                    generator_model, discriminator_model, combined_model,
                    nb_steps, batch_size, k_d, k_g,
                    log_interval, show, output_dir]

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
fig.suptitle(model_name, fontsize=20)
plt.xlabel('number of steps', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.savefig(output_dir + 'loss_plot.png')

# PLOT DISCRIM PREDICTIONS

x = np.arange(epochs)
fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(x, avg_gen_pred, label='avg_gen_pred')
ax1.plot(x, avg_real_pred, label='avg_real_pred')

ax1.legend(loc=1)
fig.suptitle(model_name, fontsize=20)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('avg prediction', fontsize=16)

plt.savefig(output_dir + 'discrim_prediction_plot.png')