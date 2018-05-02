
import gan_1obj

## TESTING ##

data_cols = ['L', 'T', 'R', 'B']

# LOAD MODEL
# generator_model, discriminator_model, combined_model = getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__gen_weights_step_100.h5',
#                                                                      discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\GAN_noise_4e-4__discrim_weights_step_100.h5')
folder_dir = 'maxGAN1obj_bs128_lr0.002_kd3_kg1_steps2740_expDecay5'
step = '2740'
generator_model, discriminator_model, combined_model = gan_1obj.getModel(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\'+folder_dir+'\\gen_weights_step_'+step+'.h5',
                                                                     discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\weights\\'+folder_dir+'\\discrim_weights_step_'+step+'.h5')

cache_prefix = folder_dir + '_step' + step
# gan.testModelMult(generator_model, discriminator_model, combined_model, cache_prefix)
gan_1obj.testModel(generator_model, discriminator_model, combined_model, cache_prefix)
# gan.testDiscrim(generator_model, discriminator_model, combined_model)
