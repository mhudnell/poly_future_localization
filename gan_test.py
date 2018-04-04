
import gan



## TESTING ##

data_cols = []
for frame in range(1,6):        # currently produces the wrong order
    for char in ['L', 'T', 'R', 'B']:
        for obj in range(1,6):
            data_cols.append('f' + str(frame) + char + str(obj))

# LOAD MODEL
# generator_model, discriminator_model, combined_model = getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__gen_weights_step_100.h5',
#                                                                      discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__discrim_weights_step_100.h5')
generator_model, discriminator_model, combined_model = gan.getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\maxGAN_bs32_lr0.0005_kd2_kg1_steps2401\\gen_weights_step_2400.h5',
                                                                     discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\maxGAN_bs32_lr0.0005_kd2_kg1_steps2401\\discrim_weights_step_2400.h5')

# gan.testModel(generator_model, discriminator_model, combined_model)
gan.testDiscrim(generator_model, discriminator_model, combined_model)