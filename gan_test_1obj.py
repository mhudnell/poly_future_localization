import gan_1obj

## TESTING ##
data_cols = []
for fNum in range(1,12):
    for char in ['L', 'T', 'W', 'H']:
        data_cols.append(char + str(fNum))
print(data_cols)

# LOAD MODEL
model_name = 'maxGAN_0.5_0.999_bs35082_lr0.0002expDecay5_kd5_kg1_steps15'
step = '15'
generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\weights\\gen_weights_step_'+step+'.h5',
                                                                          discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\weights\\discrim_weights_step_'+step+'.h5')

# gan_1obj.test_model(generator_model, discriminator_model, combined_model, model_name)
gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name)
