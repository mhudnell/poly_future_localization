import gan_1obj
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

data_cols = []
for fNum in range(1,12):
    for char in ['L', 'T', 'W', 'H']:
        data_cols.append(char + str(fNum))

print("data_cols: ", data_cols)

# def train_new_model(lr, batch_size, k_d, k_g, ):

## TRAINING ##
# CREATE NEW MODEL
lr = 0.0002 # .0005 # 5e-4, 5e-5
generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, lr=lr)
print("metrics_names:", combined_model.metrics_names)

# DEFINE TRAINING PARAMS
label_cols = []
label_dim = 0
log_interval = 1  # ~1 epoch (35082 / 32 =~ 1096, 128: 274, 35082: 1)  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
epochs = 3
nb_steps = log_interval*epochs  # 50000 # Add one for logging of the last interval
batch_size = 35082  # 128, 64
k_d = 1  # 1 number of discriminator network updates per adversarial training step
k_g = 1  # 1 number of generator network updates per adversarial training step

starting_step = 0
# model_name = 'maxGAN_6_layer44G_RELU_0.0adv_bs{}_lr{}expDecay5_kd{}_kg{}_steps{}'.format(batch_size, lr, k_d, k_g, nb_steps)
model_name = 'maxGAN_logloss_bs{}_lr{}expDecay5_kd{}_kg{}_steps{}'.format(batch_size, lr, k_d, k_g, nb_steps)
output_dir = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\'
show = True

model_components = [model_name, starting_step,
                    data_cols, label_cols, label_dim,
                    generator_model, discriminator_model, combined_model,
                    nb_steps, batch_size, k_d, k_g,
                    log_interval, show, output_dir]

[G_loss, D_loss_fake, D_loss_real, D_loss, avg_gen_pred, avg_real_pred] = gan_1obj.training_steps_GAN(model_components)

losses = [G_loss, D_loss_fake, D_loss_real, D_loss, avg_gen_pred, avg_real_pred]

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
    
