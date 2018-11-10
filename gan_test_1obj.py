import gan_1obj
import data_extract_1obj
import numpy as np

def run_tests(generator_model, discriminator_model, combined_model, model_name, seed=6, dataset='kitti_tracking'):

    np.random.seed(seed)
    if dataset == 'kitti_tracking':
        all_sets = np.arange(21)
        test_sets = np.random.choice(all_sets, 3, replace=False)
        # train_sets = np.setdiff1d(all_sets, test_sets)
        test_data, test_data_info = data_extract_1obj.get_kitti_data(test_sets)
    elif dataset == 'kitti_raw_tracklets':
        all_sets = np.arange(38)
        test_sets = np.random.choice(all_sets, 8, replace=False)
        test_data, test_data_info = data_extract_1obj.get_kitti_raw_tracklets(sets=test_sets)
        # train_sets = np.setdiff1d(all_sets, test_sets)
    else:
        raise Exception("`dataset` parameter must be one of: ['kitti_tracking', 'kitti_raw_tracklets']")

    gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name, test_data, test_data_info, dataset)



if __name__ == '__main__':
    ## TESTING ##
    data_cols = []
    for fNum in range(1, 12):
        for char in ['L', 'T', 'W', 'H']:
            data_cols.append(char + str(fNum))

    # LOAD MODEL
    model_name = 'maxGAN_CAR-ONLY_6-fold_1s-pred_G6-64_D3-32_w-adv0.0_adam-lr0.0005-b10.9-b20.999_bs4096_kd0_kg1_epochs700\\joint'
    epoch = '413'
    generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\gen_weights_epoch-' + epoch + '.h5'
    discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\discrim_weights_epoch-' + epoch + '.h5'
    optimizer = {
                'name': 'adam',
                'lr': .0005,        # default: .001
                'beta_1': .9,       # default: .9
                'beta_2': .999,     # default: .999
                'decay': 0       # default: 0
                }

    generator_model, discriminator_model, combined_model = gan_1obj.get_model(data_cols, generator_model_path=generator_model_path,
        discriminator_model_path=discriminator_model_path, optimizer=optimizer, w_adv=1.0)

    run_tests(generator_model, discriminator_model, combined_model, model_name, seed=10, dataset='kitti_raw_tracklets')

    # Deprecated
    # step = '13700'
    # generator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\gen_weights_step_' + step + '.h5'
    # discriminator_model_path = 'C:\\Users\\Max\\Research\\maxGAN\\models\\' + model_name + '\\weights\\discrim_weights_step_' + step + '.h5'

    # gan_1obj.test_model(generator_model, discriminator_model, combined_model, model_name)
    # gan_1obj.test_model_multiple(generator_model, discriminator_model, combined_model, model_name)
    # gan_1obj.test_model_IOU(generator_model, discriminator_model, combined_model, model_name)
