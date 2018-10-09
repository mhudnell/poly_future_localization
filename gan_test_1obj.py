import gan_1obj

## TESTING ##
data_cols = []
for fNum in range(1,12):
    for char in ['L', 'T', 'W', 'H']:
        data_cols.append(char + str(fNum))
print(data_cols)

# LOAD MODEL
model_name = 'maxGAN_centered_G6-64_D3-32_0.5adv_bs128_lr0.0002_kd1_kg1_epochs50'
step = '13700'
generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\weights\\gen_weights_step_'+step+'.h5',
                                                                          discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\'+model_name+'\\weights\\discrim_weights_step_'+step+'.h5')

# gan_1obj.test_model(generator_model, discriminator_model, combined_model, model_name)
gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name)
gan_1obj.test_model_IOU(generator_model, discriminator_model, combined_model, model_name)
