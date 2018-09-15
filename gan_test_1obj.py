import gan_1obj

## TESTING ##
data_cols = []
for fNum in range(1,12):
    for char in ['L', 'T', 'W', 'H']:
        data_cols.append(char + str(fNum))
print(data_cols)

# LOAD MODEL
model_name = 'maxGAN_new_data_bs128_lr0.0005expDecay5_kd2_kg1_steps2740'
step = '2740'
generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\weights\\gen_weights_step_'+step+'.h5',
                                                                            discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\weights\\discrim_weights_step_'+step+'.h5')

# gan_1obj.test_model(generator_model, discriminator_model, combined_model, model_name)
gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name)
