import tensorflow as tf

import os
import sys
import numpy as np
from scipy.io import savemat
import h5py

# Add the root project directory
try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(code_dir, '..', '..', '..'))
except NameError:
    # Running in Jupyter Notebook
    code_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(code_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(code_dir)
print(project_root) # Hest_NTN_UDA/

from Domain_Adversarial.helper import loader, plotfig, PAD
from Domain_Adversarial.helper.utils import H5BatchLoader
from Domain_Adversarial.helper.utils_GAN import visualize_H
from JMMD.helper.utils_GAN import save_checkpoint_jmmd as save_checkpoint
from JMMD.helper.utils_GAN import WeightScheduler

SNR = -5
# source_data_file_path_label = os.path.abspath(os.path.join(code_dir, '..', 'generatedChan', 'OpenNTN','H_perfect.mat'))
source_data_file_path = os.path.abspath(os.path.join(code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_D_30_sim', f'SNR_{SNR}dB', 'matlabNTN.mat'))
target_data_file_path = os.path.abspath(os.path.join(code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_A_300_sim', f'SNR_{SNR}dB', 'matlabNTN.mat'))
norm_approach = 'minmax' # can be set to 'std'
lower_range = -1 
    # if norm_approach = 'minmax': 
        # =  0 for scaling to  [0 1]
        # = -1 for scaling to [-1 1]
    # if norm_approach = 'std': can be any value, but need to be defined

# Weight scheduler for DANN
scheduler = WeightScheduler(strategy='reconstruction_first', start_domain_weight=0.01, end_domain_weight=0.05,
                            start_est_weight=1.5, end_est_weight=0.8, warmup_epochs=80) 
                            # adv_weight will be used for adversarial training
                            # domain_weight not used in DANN (only for CORAL/JMMD)
                            # warmup_epochs=150 default
                            # schedule_type = 'linear' default

if norm_approach == 'minmax':
    if lower_range == 0:
        norm_txt = 'Using min-max [0 1]'
    elif lower_range ==-1:
        norm_txt = 'Using min-max [-1 1]'
elif norm_approach == 'no':
    norm_txt = 'No'
    
# Paths to save
path_temp = code_dir + f'/results_dann/'
os.makedirs(os.path.dirname(path_temp), exist_ok=True)
idx_save_path = loader.find_incremental_filename(path_temp,'ver', '_', '')

save_model = False
model_path = code_dir + f'/results_dann/ver' + str(idx_save_path) + '_'
model_readme = model_path + '/readme.txt'

batch_size= 8 # 16

# ============ Source data ==============
source_file = h5py.File(source_data_file_path, 'r')
H_true_source = source_file['H_perfect']
N_samp_source = H_true_source.shape[0]
print('N_samp_source = ', N_samp_source)

# ============ Target data ==============
target_file = h5py.File(target_data_file_path, 'r')
H_true_target = target_file['H_perfect']
N_samp_target = H_true_target.shape[0]
print('N_samp_target = ', N_samp_target)

# Store random state 
rng_state = np.random.get_state()

# --- Set a temporary seed for reproducible split ---
np.random.seed(1234)   # any fixed integer seed
# Random but repeatable split
indices_source = np.arange(N_samp_source)
np.random.shuffle(indices_source)
indices_target = np.arange(N_samp_target)
np.random.shuffle(indices_target)
# Restore previous random state (so other code stays random)
np.random.set_state(rng_state)
#
train_size = int(np.floor(N_samp_source * 0.9) // batch_size * batch_size)
val_size = N_samp_source - train_size

# Repeat the indices to match the maximum number of samples
N_samp = max(N_samp_source, N_samp_target) 
indices_source = np.resize(indices_source, N_samp)
indices_target = np.resize(indices_target, N_samp)

# =======================================================
## Divide the indices into training and validation sets
# indices_train_source = indices_source[:train_size]
# indices_val_source   = indices_source[train_size:train_size + val_size]

# indices_train_target = indices_target[:train_size]
# indices_val_target   = indices_target[train_size:train_size + val_size]

# to test code
indices_train_source = indices_source[:96]
indices_val_source = indices_source[2032:]
indices_train_target = indices_target[:96]
indices_val_target = indices_target[2032:]

print('train_size = ', indices_train_source.shape[0])
print('val_size = ', indices_val_source.shape[0])

class DataLoaders:
    def __init__(self, file, indices_train, indices_val, tag='prac', batch_size=32): 
        # tag = 'prac' or 'li' or 'ls'
        self.true_train = H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_train)
        self.true_val = H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_val)

        self.input_train = H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_train)
        self.input_val = H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_val)

# Source domain
class_dict_source = {
    'GAN_practical': DataLoaders(source_file, indices_train_source, indices_val_source, tag='prac', batch_size=batch_size),
    'GAN_linear': DataLoaders(source_file, indices_train_source, indices_val_source, tag='li', batch_size=batch_size),
    'GAN_ls': DataLoaders(source_file, indices_train_source, indices_val_source, tag='ls', batch_size=batch_size)
}

# Target domain
class_dict_target = {
    'GAN_practical': DataLoaders(target_file, indices_train_target, indices_val_target, tag='prac', batch_size=batch_size),
    'GAN_linear': DataLoaders(target_file, indices_train_target, indices_val_target, tag='li', batch_size=batch_size),
    'GAN_ls': DataLoaders(target_file, indices_train_target, indices_val_target, tag='ls', batch_size=batch_size)
}

loss_fn_ce = tf.keras.losses.MeanSquaredError()  # Channel estimation loss (generator loss)
loss_fn_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False) # Binary cross-entropy loss for discriminator

# Import DANN-specific functions
from JMMD.helper.utils_GAN import CNNGenerator
from Domain_Adversarial.helper.utils_CNN import DomainDiscForCNN, train_step_cnn_residual_dann, val_step_cnn_residual_dann
from JMMD.helper.utils_GAN import post_val

import time
start = time.perf_counter()

# n_epochs= 300 # 300
# epoch_min = 100
# epoch_step = 20
n_epochs= 5
epoch_min = 0
epoch_step = 1

sub_folder_ = ['GAN_linear']  # ['GAN_linear', 'GAN_practical', 'GAN_ls']

for sub_folder in sub_folder_:
    print(f"Processing: {sub_folder}")
    pad_metrics = {
        'pad_pca_lda': {},      # Dictionary to store LDA PAD values by epoch
        'pad_pca_logreg': {},   # Dictionary to store LogReg PAD values by epoch
        'pad_pca_svm': {},      # Dictionary to store SVM PAD values by epoch
        'w_dist': {}            # Dictionary to store Wasserstein distances by epoch
    }
    linear_interp = False
    # if sub_folder == 'GAN_linear':
    #     linear_interp =True # flag to clip values that go beyond the estimated pilot (min, max)
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

    #
    train_metrics = {
        'train_loss': [],           # total training loss 
        'train_est_loss': [],       # estimation loss
        'train_disc_loss': [],      # discriminator loss (domain discriminator)
        'train_domain_loss': [],    # adversarial loss (CNN fooling discriminator)
        'train_est_loss_target': [] # target estimation loss (monitoring)
    }
    
    # 
    val_metrics = {
        'val_loss': [],                 # total validation loss
        'val_domain_loss': [],     # domain discriminator loss (same as above)
        'val_est_loss_source': [],      # source estimation loss
        'val_est_loss_target': [],      # target estimation loss  
        'val_est_loss': [],             # average estimation loss
        'source_acc': [],               # source domain accuracy
        'target_acc': [],               # target domain accuracy
        'acc': [],                      # average domain accuracy
        'nmse_val_source': [],          # source NMSE
        'nmse_val_target': [],          # target NMSE
        'nmse_val': [],                 # average NMSE
        'val_smoothness_loss': []
    }
    #
    H_to_save = {}          # list to save to .mat file for H
    perform_to_save = {}    # list to save to .mat file for nmse, losses,...

    # Initialize CNN Generator and Domain Discriminator
    model = CNNGenerator(n_blocks=4, extract_layers=['block_2'])
    domain_disc = DomainDiscForCNN(l2_reg=1e-5)
    
    # Separate optimizers for CNN and domain discriminator
    cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    domain_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    
    flag = 1 # flag to plot and save H_true
    epoc_pad = []    # epochs that calculating pad (return_features == True)
    
    for epoch in range(n_epochs):
        # get weights 
        weights = scheduler.get_weights_domain_first_smooth(epoch, n_epochs)
        print(f"Epoch {epoch+1}/{n_epochs}, Weights: {weights}")
        
        # ===================== Training =====================
        loader_H_true_train_source.reset()
        loader_H_input_train_source.reset()
        loader_H_true_train_target.reset()
        loader_H_input_train_target.reset()
                
        loader_H = [loader_H_input_train_source, loader_H_true_train_source, 
                    loader_H_input_train_target, loader_H_true_train_target]

        # Loss functions for DANN: estimation loss + binary cross-entropy
        loss_fn = [loss_fn_ce, loss_fn_bce]
        
        # Optimizers: CNN optimizer and domain discriminator optimizer
        optimizers = [cnn_optimizer, domain_optimizer]
    
        ##########################
        # Optional: Save features for PAD calculation
        # if epoch==0 or epoch == n_epochs-1:
        #     save_features = True
        #     epoc_pad.append(epoch)
        # else:
        #     save_features = False
        save_features = False  # Set to True if you want to calculate PAD

        ##########################
        # Train step with DANN
        train_step_output = train_step_cnn_residual_dann(
            model, domain_disc, loader_H, loss_fn, optimizers, 
            lower_range=-1, save_features=save_features, 
            weights=weights, linear_interp=linear_interp
        )

        train_epoc_loss_est        = train_step_output.avg_epoc_loss_est
        train_epoc_loss_d          = train_step_output.avg_epoc_loss_d  # Domain disc loss
        train_epoc_loss_domain     = train_step_output.avg_epoc_loss_domain  # Adversarial loss
        train_epoc_loss            = train_step_output.avg_epoc_loss
        train_epoc_loss_est_target = train_step_output.avg_epoc_loss_est_target
                # train_epoc_loss        = total train loss = loss_est + adv_weight * adv_loss
                # train_epoc_loss_est    = loss in estimation network in source domain (labels available)
                # train_epoc_loss_domain = adversarial loss (CNN fooling discriminator)
                # train_epoc_loss_d      = domain discriminator loss
                # train_epoc_loss_est_target - just to monitor - no labels in target domain
        print("Time", time.perf_counter() - start, "seconds")
        
        # Calculate PAD if features were saved
        # if save_features and (weights['adv_weight']!=0) and (epoch==0 or epoch == n_epochs-1):
        #     features_source_file = "features_source_dann.h5"
        #     features_target_file = "features_target_dann.h5"
        #     print(f"epoch {epoch+1}/{n_epochs}")
        #     ## Calculate PCA_PAD for extracted features with PCA_SVM, PCA_LDA, PCA_LogReg
        #     X_features, y_features = PAD.extract_features_with_pca(features_source_file, features_target_file, pca_components=100)
        #     pad_svm_epoc = PAD.calc_pad_svm(X_features, y_features)
        #     pad_lda_epoc = PAD.calc_pad_lda(X_features, y_features)
        #     pad_logreg_epoc = PAD.calc_pad_logreg(X_features, y_features)
        #     pad_metrics['pad_pca_svm'][f'epoch_{epoch+1}'] = pad_svm_epoc
        #     pad_metrics['pad_pca_lda'][f'epoch_{epoch+1}'] = pad_lda_epoc
        #     pad_metrics['pad_pca_logreg'][f'epoch_{epoch+1}'] = pad_logreg_epoc
            
        #     ## Distribution of extracted features
        #     plotfig.plotHist(features_source_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'source_epoch_{epoch+1}', percent=99)
        #     plotfig.plotHist(features_target_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'target_epoch_{epoch+1}', percent=99)
            
        #     if os.path.exists(features_source_file):
        #         os.remove(features_source_file)
        #     if os.path.exists(features_target_file):
        #         os.remove(features_target_file)
        #     print("Time", time.perf_counter() - start, "seconds")
        
        # Average loss for the epoch
        train_metrics['train_loss'].append(train_epoc_loss)
        print(f"epoch {epoch+1}/{n_epochs} Average Training Loss: {train_epoc_loss:.6f}")
        
        train_metrics['train_est_loss'].append(train_epoc_loss_est)
        print(f"epoch {epoch+1}/{n_epochs} Average Estimation Loss (in Source domain): {train_epoc_loss_est:.6f}")
        
        train_metrics['train_disc_loss'].append(train_epoc_loss_d)
        print(f"epoch {epoch+1}/{n_epochs} Average Domain Disc Loss: {train_epoc_loss_d:.6f}")
        
        train_metrics['train_domain_loss'].append(train_epoc_loss_domain)
        print(f"epoch {epoch+1}/{n_epochs} Average Adversarial Loss (CNN vs Disc): {train_epoc_loss_domain:.6f}")
        
        train_metrics['train_est_loss_target'].append(train_epoc_loss_est_target)
        print(f"epoch {epoch+1}/{n_epochs} For observation only - Average Estimation Loss in Target domain: {train_epoc_loss_est_target:.6f}")
        
        
        # ===================== Evaluation =====================
        loader_H_true_val_source.reset()
        loader_H_input_val_source.reset()
        loader_H_true_val_target.reset()
        loader_H_input_val_target.reset()
        loader_H_eval = [loader_H_input_val_source, loader_H_true_val_source, 
                        loader_H_input_val_target, loader_H_true_val_target]

        # Loss functions for validation
        loss_fn = [loss_fn_ce, loss_fn_bce]
        
        # Validation step with DANN
        if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) and epoch!=n_epochs-1:
            H_sample, epoc_val_return = val_step_cnn_residual_dann(
                model, domain_disc, loader_H_eval, loss_fn, lower_range, 
                weights=weights, linear_interp=linear_interp
            )
            visualize_H(H_sample, H_to_save, epoch, plotfig.figChan, flag, model_path, sub_folder, domain_weight=weights['adv_weight'])
            flag = 0  # after the first epoch, no need to save H_true anymore
        elif epoch==n_epochs-1:
            _, epoc_val_return, H_val_gen = val_step_cnn_residual_dann(
                model, domain_disc, loader_H_eval, loss_fn, lower_range, 
                weights=weights, linear_interp=linear_interp, return_H_gen=True
            )    
        else:
            _, epoc_val_return = val_step_cnn_residual_dann(
                model, domain_disc, loader_H_eval, loss_fn, lower_range, 
                weights=weights, linear_interp=linear_interp
            )
        
        post_val(epoc_val_return, epoch, n_epochs, val_metrics, domain_weight=weights['adv_weight'])
        
        if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) or epoch==n_epochs-1:
            # 
            all_metrics = {
                'figLoss': plotfig.figLoss, 
                'savemat': savemat,
                # 'pad_metrics': pad_metrics, 
                # 'epoc_pad': epoc_pad,
                'weights': weights, 
                'optimizer': [cnn_optimizer, domain_optimizer]
            }
            # Combine all metrics
            all_metrics.update(train_metrics)  # Add training metrics
            all_metrics.update(val_metrics)    # Add validation metrics

            # Save models
            if save_model:
                model.save_weights(f"{model_path}/{sub_folder}/cnn_epoch_{epoch+1}.h5")
                domain_disc.save_weights(f"{model_path}/{sub_folder}/domain_disc_epoch_{epoch+1}.h5")
            
            save_checkpoint(model, save_model, model_path, sub_folder, epoch, all_metrics)
    
    # end of epoch loop
    # =====================            
    # Save performances
    # Save H matrix
    savemat(model_path + '/' + sub_folder + '/H_visualize/H_trix.mat', H_to_save)
    savemat(model_path + '/' + sub_folder + '/H_visualize/H_val_generated.mat', 
        {'H_val_gen': H_val_gen,
        'indices_val_source': indices_val_source,
        'indices_val_target': indices_val_target})
# end of trainmode