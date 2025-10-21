import tensorflow as tf
import numpy as np
import h5py
import os
# from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from . import utils
# import utils_GAN


def load_data(outer_file_path, rows, fc, snr, batch_size=32):
    H_true = np.empty((0, 2, 612, 14))       # true channel
    H_equal = np.empty((0, 2, 612, 14))      # LS channel
    H_linear = np.empty((0, 2, 612, 14))     # LS + linear
    H_practical = np.empty((0, 2, 612, 14))  # Practical channel
    has_practical = False

    for i in range(len(rows)):
        file_path_partial = f'Gan_{snr}_dBOutdoor1_{fc}_1ant_612subcs_Row_{rows[i][0]}_{rows[i][1]}.mat'
        file_path = os.path.normpath(os.path.join(outer_file_path, file_path_partial))
        file = h5py.File(file_path, 'r')

        H_true = np.concatenate((H_true, np.array(file['H_data'])), axis=0)
        H_equal = np.concatenate((H_equal, np.array(file['H_equalized_data'])), axis=0)
        H_linear = np.concatenate((H_linear, np.array(file['H_linear_data'])), axis=0)

        if 'H_practical_data' in file:
            H_practical = np.concatenate((H_practical, np.array(file['H_practical_data'])), axis=0)
            has_practical = True

    # Shuffle all datasets with the same permutation
    shuffle_order = np.random.permutation(H_true.shape[0])
    H_true = tf.convert_to_tensor(H_true[shuffle_order], dtype=tf.float32)
    H_equal = tf.convert_to_tensor(H_equal[shuffle_order], dtype=tf.float32)
    H_linear = tf.convert_to_tensor(H_linear[shuffle_order], dtype=tf.float32)
    if has_practical:
        H_practical = tf.convert_to_tensor(H_practical[shuffle_order], dtype=tf.float32)

    train_size = int(np.floor(H_true.shape[0] * 0.9) // batch_size * batch_size)

    trainLabels = H_true[:train_size]
    valLabels = H_true[train_size:]

    H_equal_train = H_equal[:train_size]
    H_equal_val   = H_equal[train_size:]

    H_linear_train = H_linear[:train_size]
    H_linear_val   = H_linear[train_size:]

    if has_practical:
        H_practical_train = H_practical[:train_size]
        H_practical_val   = H_practical[train_size:]
    else:
        H_practical_train = tf.zeros_like(H_linear_train)
        H_practical_val   = tf.zeros_like(H_linear_val)

    return [trainLabels, valLabels], [H_equal_train, H_linear_train, H_practical_train], [H_equal_val, H_linear_val, H_practical_val]

def load_map_data(outer_file_path, snr, train_rate=0.9, batch_size=32):
    H_true = np.empty((0, 2, 612, 14))       # true channel
    H_equal = np.empty((0, 2, 612, 14))      # LS channel
    H_linear = np.empty((0, 2, 612, 14))     # LS + linear
    H_practical = np.empty((0, 2, 612, 14))  # Practical channel
    has_practical = False

    file_path_partial = f'{snr}dB/1_mapBaseData.mat'
    file_path = os.path.normpath(os.path.join(outer_file_path, file_path_partial))
    file = h5py.File(file_path, 'r')

    H_true = np.concatenate((H_true, np.array(file['H_data'])), axis=0)
    H_equal = np.concatenate((H_equal, np.array(file['H_equalized_data'])), axis=0)
    H_linear = np.concatenate((H_linear, np.array(file['H_linear_data'])), axis=0)

    if 'H_practical_data' in file:
        H_practical = np.concatenate((H_practical, np.array(file['H_practical_data'])), axis=0)
        has_practical = True

    # Shuffle
    shuffle_order = np.random.permutation(H_true.shape[0])
    H_true = tf.convert_to_tensor(H_true[shuffle_order], dtype=tf.float32)
    H_equal = tf.convert_to_tensor(H_equal[shuffle_order], dtype=tf.float32)
    H_linear = tf.convert_to_tensor(H_linear[shuffle_order], dtype=tf.float32)
    if has_practical:
        H_practical = tf.convert_to_tensor(H_practical[shuffle_order], dtype=tf.float32)

    # Split
    train_size = int(np.floor(H_true.shape[0] * train_rate) // batch_size * batch_size)

    trainLabels = H_true[:train_size]
    valLabels = H_true[train_size:]

    H_equal_train = H_equal[:train_size]
    H_equal_val   = H_equal[train_size:]

    H_linear_train = H_linear[:train_size]
    H_linear_val   = H_linear[train_size:]

    if has_practical:
        H_practical_train = H_practical[:train_size]
        H_practical_val = H_practical[train_size:]
    else:
        H_practical_train = tf.zeros_like(H_linear_train)
        H_practical_val = tf.zeros_like(H_linear_val)

    return [trainLabels, valLabels], [H_equal_train, H_linear_train, H_practical_train], [H_equal_val, H_linear_val, H_practical_val]


def loader_dataset(H_linear_train_normd, H_true_train_normd, H_linear_val_normd, H_true_val_normd, batch_size=32):
    # Split real and imaginary parts and concatenate across the batch dimension
    train_real_imag = tf.concat([H_linear_train_normd[:, 0, :, :], H_linear_train_normd[:, 1, :, :]], axis=0)
    label_real_imag = tf.concat([H_true_train_normd[:, 0, :, :], H_true_train_normd[:, 1, :, :]], axis=0)

    # Add channel dimension: [2*N, 612, 14] -> [2*N, 612, 14, 1]
    train_real_imag = tf.expand_dims(train_real_imag, axis=-1)
    label_real_imag = tf.expand_dims(label_real_imag, axis=-1)

    # Create tf.data.Dataset for training
    train_dataset = tf.data.Dataset.from_tensor_slices((train_real_imag, label_real_imag))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation set remains in [N, 2, 612, 14] format
    val_dataset = tf.data.Dataset.from_tensor_slices((H_linear_val_normd, H_true_val_normd))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

# def load_model(params, file_path):
#     generator_li = utils_GAN.Generator(in_channel=params['in_channel'])   # in_channel=1 to estimate real and imag parts separately,
#                                                                 # default: in_channel=2 to estimate real and imag parts at the same time
#     discriminator_li = utils_GAN.Discriminator(params['in_channel'])
#     generator_ls = utils_GAN.Generator(in_channel=params['in_channel'])   # in_channel=1 to estimate real and imag parts separately,
#                                                                 # default: in_channel=2 to estimate real and imag parts at the same time
#     discriminator_ls = utils_GAN.Discriminator(params['in_channel'])
#     if params['load_saved_model']:
#         # modify the directory
#         params['epoc'] = params['epoc_saved_model']
        
#         variable_load_path = os.path.join(file_path, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_variable_'+params['rowss']+'.pth')
#         var_state = torch.load(variable_load_path)
                
#         # load for (LS+LI) model
#         generator_li_load_path = os.path.join(file_path, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_generator_'+params['rowss']+'.pth')
#         discriminator_li_load_path = os.path.join(file_path, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_discriminator_'+params['rowss']+'.pth')
#         # load the models
#         generator_li.load_state_dict(torch.load(generator_li_load_path))
#         discriminator_li.load_state_dict(torch.load(discriminator_li_load_path))
        
        
#         # load for LS model
#         generator_ls_load_path = os.path.join(file_path, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_generator_'+params['rowss']+'.pth')
#         discriminator_ls_load_path = os.path.join(file_path, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_discriminator_'+params['rowss']+'.pth')
#         # load the models
#         generator_ls.load_state_dict(torch.load(generator_ls_load_path))
#         discriminator_ls.load_state_dict(torch.load(discriminator_ls_load_path))

#     if params['load_saved_model']:
#         gen_li_loss_track = var_state['gen_li_loss_track']
#         disc_li_loss_track = var_state['disc_li_loss_track']
#         gen_li_val_loss_track = var_state['gen_li_val_loss_track']
#         gen_ls_loss_track = var_state['gen_ls_loss_track']
#         disc_ls_loss_track = var_state['disc_ls_loss_track']
#         gen_ls_val_loss_track = var_state['gen_ls_val_loss_track']
#     else: 
#         gen_li_loss_track  = []    # BCE loss in training
#         disc_li_loss_track = []    # BCE loss in training
#         gen_li_val_loss_track = [] # MSE _ compare estimated and true channels
#         gen_ls_loss_track  = []    # BCE loss in training
#         disc_ls_loss_track = []    # BCE loss in training
#         gen_ls_val_loss_track = [] # MSE _ compare estimated and true channels
#     return [generator_li, discriminator_li], [generator_ls, discriminator_ls], [gen_li_loss_track, disc_li_loss_track, gen_li_val_loss_track], [gen_ls_loss_track, disc_ls_loss_track, gen_ls_val_loss_track]

# save to .mat file
def find_incremental_filename(directory, prefix_name, postfix_name, extension='.mat'):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out files that match the pattern prefix_name + number + postfix_name + extension
    existing_files = [f for f in files if f.startswith(prefix_name) and f.endswith(postfix_name + extension)]
    
    # Extract the numbers from the filenames
    numbers = []
    for f in existing_files:
        # Strip the prefix and postfix, then extract the number in between
        try:
            number_part = f[len(prefix_name):-len(postfix_name + extension)]
            if number_part.isdigit():
                numbers.append(int(number_part))
        except ValueError:
            pass  # Skip any files that don't match the expected pattern
    
    # Determine the next number
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1  # Start numbering from 1 if no existing files  
    return next_number

def genLoader(data, target, BATCH_SIZE, device, mode, shuff, approach, lower_range=-1, random_repeat=False, size_after_repeat=0):
        # mode = 'train' or 'valid'
        # approach = 'minmax', 'std', or 'no'
        #   in 'minmax' case: x-min;  y-max
        #                       lower_range = -1  -- scale to [-1 1] range
        #                       lower_range =  0  -- scale to  [0 1] range
        #   in    'std' case: x-mean; y-var 

    # 1.2 Normalization Min-Max scaler
    if approach == 'minmax':
        data_normd,   data_x, data_y  = utils.minmaxScaler(data, lower_range)
        label_normd, label_x, label_y = utils.minmaxScaler(target, lower_range)
                    # x: min
                    # y: max
    elif approach == 'std':
        data_normd,   data_x, data_y  = utils.standardize(data)
        label_normd, label_x, label_y = utils.standardize(target)
                    # x: mean
                    # y: var
        # data_normd  == torch.tensor N_samples(data) x 2 x 612 x 14
        # label_normd == torch.tensor N_samples(label) x 2 x 612 x 14
        # label/data x, y == torch.tensor  N_samples (data/target) x 2 - min/max/mean/var of real/imag of each sample
    elif approach == 'no':
        data_normd  = data
        label_normd = target
        label_x = tf.zeros((target.shape[0], 2), dtype=tf.float32)
        label_y = tf.zeros((target.shape[0], 2), dtype=tf.float32)
    
    if mode == 'train':
        # Split real and imaginary grids into 2 image sets, then concatenate
        data_real_imag = tf.concat([data_normd[:, 0, :, :], data_normd[:, 1, :, :]], axis=0)
        label_real_imag = tf.concat([label_normd[:, 0, :, :], label_normd[:, 1, :, :]], axis=0)
        data_normd = tf.expand_dims(data_real_imag, axis=-1)    # shape: [2N, 612, 14, 1]
        label_normd = tf.expand_dims(label_real_imag, axis=-1)  # shape: [2N, 612, 14, 1]
        label_x = tf.concat([label_x[:, 0], label_x[:, 1]], axis=0)  # shape: [2N]
        label_y = tf.concat([label_y[:, 0], label_y[:, 1]], axis=0)
        # label x, y == torch.tensor  N_samples (data/target)*2 x 1 - min/max/mean/var of concatenated real-imag of each sample
        
    # data_normd  = data_normd.to(device, dtype=torch.float)
    # label_normd = label_normd.to(device, dtype=torch.float)
    # label_x     = label_x.to(device, dtype=torch.float)
    # label_y     = label_y.to(device, dtype=torch.float)

    # 1.3 Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((data_normd, label_normd, label_x, label_y))  # [4224, 1, n_subcs, n_symbs]
    if random_repeat:
        indices = tf.random.uniform(shape=(size_after_repeat,), minval=0, maxval=tf.shape(data_normd)[0], dtype=tf.int32)
        dataset = dataset.enumerate()
        dataset = dataset.filter(lambda i, _: tf.reduce_any(tf.equal(i, indices)))
        dataset = dataset.map(lambda _, val: val)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    else:
        if shuff:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, label_x, label_y    # note: x, y (min, max, var,... here have been shuffled, don't use)
