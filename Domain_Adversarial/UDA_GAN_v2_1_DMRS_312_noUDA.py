import tensorflow as tf
print(tf.__version__)

# Robust GPU configuration to prevent CUDA errors
try:
    # Set memory growth to prevent CUDA allocation issues
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Available GPUs: {len(gpus)}")
    
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for GPU: {gpu}")
        
        # Optional: Limit GPU memory if needed (uncomment if memory issues persist)
        # tf.config.experimental.set_memory_limit(gpus[0], 8192)  # 8GB limit
        
        # Set visible devices explicitly
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    else:
        print("No GPUs found, using CPU")
        
except RuntimeError as e:
    print(f"GPU configuration error: {e}")
    print("Falling back to CPU execution")
except Exception as e:
    print(f"Unexpected GPU error: {e}")
    print("Continuing with default GPU settings")

import os
import sys
import numpy as np
import pickle
from scipy.io import savemat, loadmat
from sklearn.linear_model import SGDClassifier

# Uncomment next line to force CPU usage for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces CPU usage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'helper')))
# import utils
# import loader
import utils_GAN_copy as utils_GAN
import PAD, utils_GAN_FiLM

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
print(os.path.abspath(os.path.join(script_dir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..')))
import Est_btween_CSIRS.helper.utils as utils_CNN
import Est_btween_CSIRS.helper.loader as loader
import Est_btween_CSIRS.helper.plotfig as plotfig

source_data_file_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'Generate_Data', 'CDL_Channel', 'generatedChannel', 'ver9_', '0dB', 'mapBaseData.mat'))
target_data_file_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'Generate_Data', 'Sionna', 'generatedChannel', 'ver3_', 'sionnaTrue.mat'))

norm_approach = 'minmax' # can be set to 'std'
lower_range = -1 
    # if norm_approach = 'minmax': 
        # =  0 for scaling to  [0 1]
        # = -1 for scaling to [-1 1]
    # if norm_approach = 'std': can be any value, but need to be defined
adv_weight=0.005
est_weight=1
domain_weight=0 # 0.5 for Domain Discriminator, 0 for no Domain Discriminator

# snr_start = -25
# snr_step = 5
# snr_end = 25
# SNR = np.arange(snr_start, snr_end+1, snr_step)

# SNR = np.array([0])

# if len(SNR) >1:
#     SNR_txt = f'{snr_start}:{snr_step}:{snr_end}'
# else:
#     SNR_txt = f'{SNR[0]}'
    
# ============ CNN settings ==============
if norm_approach == 'minmax':
    if lower_range == 0:
        norm_txt = 'Using min-max [0 1]'
    elif lower_range ==-1:
        norm_txt = 'Using min-max [-1 1]'
elif norm_approach == 'no':
    norm_txt = 'No'
    
CNN_activation = 'Tanh'
CNN_DropOut = 0.2
if CNN_DropOut != 0:
    dropOut_txt = f'Add p={CNN_DropOut} DropOut'


# Paths to save
idx_save_path = loader.find_incremental_filename(script_dir + '/model/GAN_calcu','ver', '_', '')

save_model = 1

load_checkpoint = True  # True if continue training
if load_checkpoint:
    model_path = script_dir + '/model/GAN_calcu/ver' + str(idx_save_path-1) + '_' # or replace idx_save_path-1 by the desired folder index
else:
    model_path = script_dir + '/model/GAN_calcu/ver' + str(idx_save_path) + '_'
if load_checkpoint:
    start_epoch = 41  # This is the epoch we want to CONTINUE FROM (not load from)
else:
    start_epoch = 0    

# figure_path = notebook_dir + '/model/GAN/ver' + str(idx_save_path) + '_/figure'
model_readme = model_path + '/readme.txt'

# Generate a (16, 792, 14, 2) matrix with random values
random_matrix = np.random.randn(16, 312, 14, 2)
random_matrix.shape

GAN_model = utils_GAN.GAN(n_subc=312)
out_put = GAN_model(random_matrix)
print(out_put.gen_out.shape)
print(out_put.disc_out.shape)
print(out_put.extracted_features.shape)

source_data_file_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'Generate_Data', 'CDL_Channel', 'generatedChannel', 'ver9_', '0dB', 'mapBaseData.mat'))
target_data_file_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'Generate_Data', 'Sionna', 'generatedChannel', 'ver3_', 'sionnaTrue.mat'))

import h5py

batch_size=16

# ============ Source data ==============
source_file = h5py.File(source_data_file_path, 'r')
H_true_source = source_file['H_true']
N_samp_source = H_true_source.shape[0]
print('N_samp_source = ', N_samp_source)

# ============ Target data ==============
target_file = h5py.File(target_data_file_path, 'r')
H_true_target = target_file['H_true']
N_samp_target = H_true_target.shape[0]
print('N_samp_target = ', N_samp_target)

indices_source = np.arange(N_samp_source)
np.random.shuffle(indices_source)
indices_target = np.arange(N_samp_target)
np.random.shuffle(indices_target)
#
train_size = int(np.floor(N_samp_source * 0.9) // batch_size * batch_size)
val_size = N_samp_source - train_size

# Repeat the indices to match the maximum number of samples
N_samp = max(N_samp_source, N_samp_target) 
indices_source = np.resize(indices_source, N_samp)
indices_target = np.resize(indices_target, N_samp)

# =======================================================
## Divide the indices into training and validation sets
indices_train_source = indices_source[:train_size]
indices_val_source   = indices_source[train_size:train_size + val_size]

indices_train_target = indices_target[:train_size]
indices_val_target   = indices_target[train_size:train_size + val_size]

# to test code
# indices_train_source = indices_source[:96]
# indices_val_source = indices_source[2032:]
# indices_train_target = indices_target[:96]
# indices_val_target = indices_target[2032:]

print('train_size = ', indices_train_source.shape[0])
print('val_size = ', indices_val_source.shape[0])

# =========== Source dataset ==============
loader_H_true_train_source = utils_CNN.H5BatchLoader(source_file, dataset_name='H_true', batch_size=batch_size, shuffled_indices=indices_train_source)
loader_H_practical_train_source = utils_CNN.H5BatchLoader(source_file, 'H_practical_save', batch_size=batch_size, shuffled_indices=indices_train_source)
loader_H_linear_train_source = utils_CNN.H5BatchLoader(source_file, 'H_linear_save', batch_size=batch_size, shuffled_indices=indices_train_source)

loader_H_true_val_source = utils_CNN.H5BatchLoader(source_file, dataset_name='H_true', batch_size=batch_size, shuffled_indices=indices_val_source)
loader_H_practical_val_source = utils_CNN.H5BatchLoader(source_file, 'H_practical_save', batch_size=batch_size, shuffled_indices=indices_val_source)
loader_H_linear_val_source = utils_CNN.H5BatchLoader(source_file, 'H_linear_save', batch_size=batch_size, shuffled_indices=indices_val_source)


# =========== Target dataset ==============
# replace source_file by target_file
loader_H_true_train_target = utils_CNN.H5BatchLoader(source_file, dataset_name='H_true', batch_size=batch_size, shuffled_indices=indices_train_target)
    # actually at target domain, we don't have true channels, just use this for evaluating the model
loader_H_practical_train_target = utils_CNN.H5BatchLoader(source_file, 'H_practical_save', batch_size=batch_size, shuffled_indices=indices_train_target)
    # channel at symbol 2 of slots 1,6,11 (channel corresponding to CSI-RS 1, 2)
loader_H_true_val_target = utils_CNN.H5BatchLoader(source_file, dataset_name='H_true', batch_size=batch_size, shuffled_indices=indices_val_target)
loader_H_practical_val_target = utils_CNN.H5BatchLoader(source_file, 'H_practical_save', batch_size=batch_size, shuffled_indices=indices_val_target)

print('size loader_H_true_train = ', loader_H_true_train_target.total_batches)
print('size loader_H_true_val = ', loader_H_true_val_target.total_batches)

class DataLoaders:
    def __init__(self, file, indices_train, indices_val, tag='practical', batch_size=32):
        self.true_train = utils_CNN.H5BatchLoader(file, dataset_name='H_true', batch_size=batch_size, shuffled_indices=indices_train)
        self.true_val = utils_CNN.H5BatchLoader(file, dataset_name='H_true', batch_size=batch_size, shuffled_indices=indices_val)

        self.input_train = utils_CNN.H5BatchLoader(file, f'H_{tag}_save', batch_size=batch_size, shuffled_indices=indices_train)
        self.input_val = utils_CNN.H5BatchLoader(file, f'H_{tag}_save', batch_size=batch_size, shuffled_indices=indices_val)

# Source domain
class_dict_source = {
    'GAN_practical': DataLoaders(source_file, indices_train_source, indices_val_source, tag='practical', batch_size=batch_size),
    'GAN_linear': DataLoaders(source_file, indices_train_source, indices_val_source, tag='linear', batch_size=batch_size)
}

# Target domain
# replace source_file by target_file when run UDA
class_dict_target = {
    'GAN_practical': DataLoaders(source_file, indices_train_target, indices_val_target, tag='practical', batch_size=batch_size),
    'GAN_linear': DataLoaders(source_file, indices_train_target, indices_val_target, tag='linear', batch_size=batch_size)
}

loss_fn_ce = tf.keras.losses.MeanSquaredError()  # Channel estimation loss (generator loss)
loss_fn_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False) # Binary cross-entropy loss for discriminator
loss_fn_domain = tf.keras.losses.BinaryCrossentropy()  # Domain classification loss

import time
start = time.perf_counter()

n_epochs= 300
epoch_min = 10
epoch_step = 10
# n_epochs= 3
# epoch_min = 0
# epoch_step = 1

sub_folder_ = ['GAN_linear'] # , 'GAN_practical']

for sub_folder in sub_folder_:
    print(f"Processing: {sub_folder}")
    linear_interp = False
    if sub_folder == 'GAN_linear':
        linear_interp =True # flag to clip values that go beyond the estimated pilot (min, max)
    ##
    loader_H_true_train_source = class_dict_source[sub_folder].true_train
    loader_H_input_train_source = class_dict_source[sub_folder].input_train
    loader_H_true_val_source = class_dict_source[sub_folder].true_val
    loader_H_input_val_source = class_dict_source[sub_folder].input_val
    
    loader_H_true_train_target = class_dict_target[sub_folder].true_train
    loader_H_input_train_target = class_dict_target[sub_folder].input_train
    loader_H_true_val_target = class_dict_target[sub_folder].true_val
    loader_H_input_val_target = class_dict_target[sub_folder].input_val
    ##
    
    if not os.path.exists(os.path.dirname(model_path + '/' + sub_folder +'/')):
        os.makedirs(os.path.dirname(model_path + '/' + sub_folder + '/'))   # Domain_Adversarial/model/_/ver_/{sub_folder}

    
    flag = 1 # flag to plot and save H_true
    H_to_save = {}          # list to save to .mat file for H
    if load_checkpoint==False:
        train_loss          = [] # (epoch,1)
        train_est_loss      = [] 
        train_disc_loss     = [] 
        train_domain_loss   = []
        train_est_loss_target = []
        #    
        val_loss, val_gan_disc_loss, val_domain_disc_loss,\
        val_est_loss_source, val_est_loss_target, val_est_loss,\
        source_acc, target_acc, acc,\
        nmse_val_source, nmse_val_target, nmse_val = [[] for _ in range(12)]
        #

        model = utils_GAN.GAN(n_subc=312, gen_l2=None, disc_l2=1e-5)  # l2 regularization for generator and discriminator
        model_domain = utils_GAN_FiLM.DomainDiscriminator3()
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)  # WGAN-GP uses Adam optimizer with beta_1=0.5
        domain_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        ####
        optimizer = [gen_optimizer, disc_optimizer, domain_optimizer]
        ####
    
        pad_observe = [] # pad observation over epochs
        epoc_pad = []    # epochs that calculating pad (return_features == True)
    else:   # load from check_point
        model = utils_GAN.GAN(n_subc=312, gen_l2=None, disc_l2=1e-5)
        # model.build(input_shape=(None, 16, 312, 14, 2))
        dummy_input = tf.random.normal((16, 312, 14, 2))
        _ = model(dummy_input)  # This builds the model with proper weights initialization
        # Build domain discriminator
        model_domain = utils_GAN_FiLM.DomainDiscriminator3()
        # 
        # Load checkpoint from the epoch we want to continue from (start_epoch-1 because we want to continue FROM start_epoch)
        epoch_load = start_epoch - 1  # Load the checkpoint from the previous epoch
        
        print(f"Loading checkpoint from epoch {epoch_load+1} to continue training from epoch {start_epoch}...")

        # Load checkpoint (this will also restore optimizers automatically)
        gen_optimizer, disc_optimizer, domain_optimizer = utils_GAN.load_checkpoint(
            model,
            model_path,
            sub_folder,
            epoch_load,
            domain_model=model_domain,
            domain_weight=domain_weight  # Use the same domain_weight as current training
        )
        optimizer = [gen_optimizer, disc_optimizer, domain_optimizer]    
        
        # Load performance history UP TO start_epoch (not including it)
        loadmat_params = loadmat(f"{model_path}/{sub_folder}/performance/performance.mat")
        train_loss          = loadmat_params['train_loss'].flatten().tolist()[:start_epoch]
        train_est_loss      = loadmat_params['train_est_loss'].flatten().tolist()[:start_epoch]
        train_disc_loss     = loadmat_params['train_disc_loss'].flatten().tolist()[:start_epoch]
        train_domain_loss   = loadmat_params['train_domain_loss'].flatten().tolist()[:start_epoch]
        train_est_loss_target = loadmat_params['train_est_loss_target'].flatten().tolist()[:start_epoch]
        #    
        val_loss             = loadmat_params['val_loss'].flatten().tolist()[:start_epoch]
        val_gan_disc_loss    = loadmat_params['val_gan_disc_loss'].flatten().tolist()[:start_epoch]
        val_domain_disc_loss = loadmat_params['val_domain_disc_loss'].flatten().tolist()[:start_epoch]
        val_est_loss_source  = loadmat_params['val_est_loss_source'].flatten().tolist()[:start_epoch]
        val_est_loss_target  = loadmat_params['val_est_loss_target'].flatten().tolist()[:start_epoch]
        val_est_loss         = loadmat_params['val_est_loss'].flatten().tolist()[:start_epoch]
        source_acc           = loadmat_params['source_acc'].flatten().tolist()[:start_epoch]
        target_acc           = loadmat_params['target_acc'].flatten().tolist()[:start_epoch]
        acc                  = loadmat_params['acc'].flatten().tolist()[:start_epoch]
        nmse_val_source      = loadmat_params['nmse_val_source'].flatten().tolist()[:start_epoch]
        nmse_val_target      = loadmat_params['nmse_val_target'].flatten().tolist()[:start_epoch]
        nmse_val             = loadmat_params['nmse_val'].flatten().tolist()[:start_epoch]
        pad_observe          = loadmat_params['pad_observe'].flatten().tolist()[:start_epoch]
        epoc_pad             = loadmat_params['epoc_pad'].flatten().tolist()[:start_epoch]
        
        print(f"Loaded {len(train_loss)} epochs of training history.")
        print(f"Last loaded training loss: {train_loss[-1] if train_loss else 'No history'}")
        
    for epoch in range(start_epoch, n_epochs):
        # ===================== Training =====================
        loader_H_true_train_source.reset()
        # loader_H_practical_train_source.reset()
        loader_H_input_train_source.reset()
        loader_H_true_train_target.reset()
        # loader_H_practical_train_target.reset()
        loader_H_input_train_target.reset()
                
        # loader_H = [loader_H_practical_train_source, loader_H_true_train_source, loader_H_practical_train_target, loader_H_true_train_target]
        loader_H = [loader_H_input_train_source, loader_H_true_train_source, loader_H_input_train_target, loader_H_true_train_target]
        
        loss_fn = [loss_fn_ce, loss_fn_bce, loss_fn_domain]
    
        ##########################
        if epoch in [int(n_epochs * r) for r in [0, 0.25, 0.5, 0.75, 1.0]]:
            # return_features == return features to calculate PAD
            return_features = True
            epoc_pad.append(epoch)
        else:
            return_features = False

        ##########################
        train_step_output = utils_GAN.train_step_wgan_gp(model, model_domain, loader_H, loss_fn, optimizer, lower_range=-1,
                                adv_weight=adv_weight, est_weight=est_weight, domain_weight=domain_weight, return_features=return_features, linear_interp=linear_interp)
        
        train_epoc_loss_est        = train_step_output.avg_epoc_loss_est
        train_epoc_loss_d          = train_step_output.avg_epoc_loss_d
        train_epoc_loss_domain     = train_step_output.avg_epoc_loss_domain
        train_epoc_loss            = train_step_output.avg_epoc_loss
        train_epoc_loss_est_target = train_step_output.avg_epoc_loss_est_target
                # train_epoc_loss        = total train loss = loss_est + lambda_domain * domain_loss
                # train_epoc_loss_est    = loss in estimation network in source domain (labels available)
                # train_epoc_loss_domain = loss in domain discrimination network
                # train_epoc_loss_est_target - just to monitor - the machine can not calculate because no label available in source domain
                # All are already calculated in average over training dataset (source/target - respectively)
        print("Time", time.perf_counter() - start, "seconds")
        # Calculate PAD for the extracted features
        if return_features and (domain_weight!=0):
            features_source_file = "features_source.h5"
            features_target_file = "features_target.h5"
            print(f"epoch {epoch+1}/{n_epochs}")
            # pad_epoc_sgd  = PAD.cal_PAD_SGD(features_source_file, features_target_file)
            pad_epoc  = PAD.cal_PAD2(features_source_file, features_target_file, pca_components=100, batch_size=128)
            pad_observe.append(pad_epoc)
            if os.path.exists(features_source_file):
                os.remove(features_source_file)
            if os.path.exists(features_target_file):
                os.remove(features_target_file)
            print("Time", time.perf_counter() - start, "seconds")
            
        
        # Average loss for the epoch
        train_loss.append(train_epoc_loss)
        print(f"epoch {epoch+1}/{n_epochs} Average Training Loss: {train_epoc_loss:.6f}")
        #
        train_est_loss.append(train_epoc_loss_est)
        print(f"epoch {epoch+1}/{n_epochs} Average Estimation Loss (in Source domain): {train_epoc_loss_est:.6f}")
        #
        train_disc_loss.append(train_epoc_loss_d)
        print(f"epoch {epoch+1}/{n_epochs} Average Disc Loss (in Source domain): {train_epoc_loss_d:.6f}")
        #
        train_domain_loss.append(train_epoc_loss_domain)
        print(f"epoch {epoch+1}/{n_epochs} Average Domain Discrimination Loss: {train_epoc_loss_domain:.6f}")
        #
        train_est_loss_target.append(train_epoc_loss_est_target)
        print(f"epoch {epoch+1}/{n_epochs} For observation only - Average Estimation Loss in Target domain: {train_epoc_loss_est_target:.6f}")
        
        
        # ===================== Evaluation =====================
        loader_H_true_val_source.reset()
        loader_H_input_val_source.reset()
        loader_H_true_val_target.reset()
        loader_H_input_val_target.reset()
        loader_H_eval = [loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target]

        loss_fn = [loss_fn_ce, loss_fn_bce, loss_fn_domain]
        
        # eval_func = utils_UDA_FiLM.val_step
        if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) or epoch==n_epochs-1:
            H_sample, epoc_val_return = utils_GAN.val_step_wgan_gp(model, model_domain, loader_H_eval, loss_fn, lower_range, 
                                            adv_weight=adv_weight, est_weight=est_weight, domain_weight=domain_weight, linear_interp=linear_interp)
            utils_GAN.visualize_H(H_sample, H_to_save, epoch, plotfig.figChan, flag, model_path, sub_folder, domain_weight=domain_weight)
            flag = 0  # after the first epoch, no need to save H_true anymore
            
        else:
            _, epoc_val_return = utils_GAN.val_step_wgan_gp(model, model_domain, loader_H_eval, loss_fn, lower_range, 
                                            adv_weight=adv_weight, est_weight=est_weight, domain_weight=domain_weight, linear_interp=linear_interp)
        
        utils_GAN.post_val(epoc_val_return, epoch, n_epochs, val_est_loss, val_est_loss_source, val_loss, val_est_loss_target,
            val_gan_disc_loss, val_domain_disc_loss, nmse_val_source, nmse_val_target, nmse_val, source_acc, target_acc, acc, domain_weight=domain_weight)
        
        if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) or epoch==n_epochs-1:
            utils_GAN.save_checkpoint(model, save_model, model_path, sub_folder, epoch, plotfig.figLoss, savemat, train_loss, train_est_loss, train_domain_loss, train_est_loss_target,
                    val_est_loss, val_est_loss_source, val_loss, val_est_loss_target, val_gan_disc_loss, val_domain_disc_loss,
                    source_acc, target_acc, acc, nmse_val_source, nmse_val_target, nmse_val, pad_observe, epoc_pad, train_disc_loss, domain_weight=domain_weight, optimizer=optimizer)
        else:
            os.makedirs(f"{model_path}/{sub_folder}/model/", exist_ok=True)
            # model.save(f"{model_path}/{sub_folder}/model/epoch_.keras")
            content = "Model at epoch " + str(epoch+1)
            txt_file = os.path.join(model_path, sub_folder, "model", "readme.txt")
            with open(txt_file, "w") as f:
                f.write(f"Model at epoch {epoch+1}\n")
        
    # end of epoch loop
    # =====================            
    # Save performances
    # Save H matrix
    savemat(model_path + '/' + sub_folder + '/H_visualize/H_trix.mat', H_to_save)

# end of trainmode   
    

