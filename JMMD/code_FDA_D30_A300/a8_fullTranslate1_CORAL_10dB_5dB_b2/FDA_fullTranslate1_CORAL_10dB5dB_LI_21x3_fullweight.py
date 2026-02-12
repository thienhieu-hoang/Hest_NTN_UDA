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

SNR = 5
# source_data_file_path_label = os.path.abspath(os.path.join(code_dir, '..', 'generatedChan', 'OpenNTN','H_perfect.mat'))
source_data_file_path = os.path.abspath(os.path.join(code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_D_30_sim', f'SNR_10dB', 'matlabNTN.mat'))
target_data_file_path = os.path.abspath(os.path.join(code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_A_300_sim', f'SNR_{SNR}dB', 'matlabNTN.mat'))
norm_approach = 'minmax' # can be set to 'std'
lower_range = -1 
    # if norm_approach = 'minmax': 
        # =  0 for scaling to  [0 1]
        # = -1 for scaling to [-1 1]
    # if norm_approach = 'std': can be any value, but need to be defined
# weights = {
#     # Core loss weights
#     'adv_weight': 0.05,        # GAN adversarial loss weight
#     'est_weight': 0.6,          # Estimation loss weight (main task)
#     'domain_weight': 1.5,       # CORAL loss weight (domain adaptation)
    
#     # Smoothness regularization weights
#     'temporal_weight': 0.02,    # Temporal smoothness penalty
#     'frequency_weight': 0.1,    # Frequency smoothness penalty
# }
# print('adv_weight = ', weights['adv_weight'], ', est_weight = ', weights['est_weight'], ', domain_weight = ', weights['domain_weight'])

print("CORAL, reconstruction training first then domain adaptation training, domain weight 0.01 → 1.5, est weight 1.5 → 0.8, warmup epochs 80")
scheduler = WeightScheduler(strategy='reconstruction_first', start_domain_weight=0.01, end_domain_weight=1.5,
                            start_est_weight=1.5, end_est_weight=0.8, warmup_epochs=80) 
                            # adv_weight = 0.005 default
                            # warmup_epochs=150 default
                            # schedule_type = 'linear' default

fda_win_h=21 
fda_win_w=3 
fda_weight= 1.0 # 0.8 # 1.0
print(f"FDA window size: {fda_win_h}x{fda_win_w}, FDA weight: {fda_weight}")

if norm_approach == 'minmax':
    if lower_range == 0:
        norm_txt = 'Using min-max [0 1]'
    elif lower_range ==-1:
        norm_txt = 'Using min-max [-1 1]'
elif norm_approach == 'no':
    norm_txt = 'No'
    
# Paths to save
path_temp = code_dir + f'/results/'
os.makedirs(os.path.dirname(path_temp), exist_ok=True)
idx_save_path = loader.find_incremental_filename(path_temp,'ver', '_', '')

save_model = False
model_path = code_dir + f'/results/ver' + str(idx_save_path) + '_'
# figure_path = code_dir + '/model/GAN/ver' + str(idx_save_path) + '_/figure'
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

from JMMD.helper.utils_GAN import CNNGenerator
from JMMD.helper.utils_GAN import post_val, train_step_cnn_residual_FDAfullTranslation1_coral, val_step_cnn_residual_coral

import time
start = time.perf_counter()

n_epochs= 300 # 300
epoch_min = 100
epoch_step = 20
# n_epochs= 5
# epoch_min = 0
# epoch_step = 1

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
    
    # Distribution of original input training datasets (or before training)    
    # plotfig.plotHist(loader_H_input_train_source, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=100)
    # plotfig.plotHist(loader_H_input_train_target, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=100)
    
    # plotfig.plotHist(loader_H_input_train_source, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=99)
    # plotfig.plotHist(loader_H_input_train_target, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=99)
    
    # plotfig.plotHist(loader_H_input_train_source, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=95)
    # plotfig.plotHist(loader_H_input_train_target, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=95)

    # plotfig.plotHist(loader_H_input_train_source, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=90)
    # plotfig.plotHist(loader_H_input_train_target, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=90)

    # Calculate Wasserstein-1 distance for original input training datasets (before training)
    # print("Calculating Wasserstein-1 distance for original input training datasets (before training)...")
    # w_dist_epoc = plotfig.wasserstein_approximate(loader_H_input_train_source, loader_H_input_train_target)
    # pad_metrics['w_dist']['before_training'] = w_dist_epoc
    
    # # Calculate     PAD for original input training datasets with SVM
    # pad_svm = PAD.original_PAD(loader_H_input_train_source, loader_H_input_train_target)
    # print(f"PAD = {pad_svm:.4f}")
    
    # # Calculate PCA_PAD for original input training datasets with PCA_SVM, PCA_LDA, PCA_LogReg
    # X_features_, y_features_ = PAD.extract_features_with_pca(loader_H_input_train_source, loader_H_input_train_target, pca_components=100)
    # pad_pca_svm_epoc = PAD.calc_pad_svm(X_features_, y_features_)
    # pad_pca_lda_epoc = PAD.calc_pad_lda(X_features_, y_features_)
    # pad_pca_logreg_epoc = PAD.calc_pad_logreg(X_features_, y_features_)
    
    # pad_metrics['pad_pca_lda']['before_training'] = pad_pca_lda_epoc
    # pad_metrics['pad_pca_logreg']['before_training'] = pad_pca_logreg_epoc  
    # pad_metrics['pad_pca_svm']['before_training'] = pad_pca_svm_epoc
    ## 
    
    if not os.path.exists(os.path.dirname(model_path + '/' + sub_folder +'/')):
        os.makedirs(os.path.dirname(model_path + '/' + sub_folder + '/'))   # Domain_Adversarial/model/_/ver_/{sub_folder}

    #
    train_metrics = {
        'train_loss': [],           # total training loss 
        'train_est_loss': [],       # estimation loss
        'train_disc_loss': [],      # discriminator loss
        'train_domain_loss': [],    # CORAL loss (replaces domain loss)
        'train_est_loss_target': [] # target estimation loss (monitoring)
    }
    
    # 
    val_metrics = {
        'val_loss': [],                 # total validation loss
        'val_gan_disc_loss': [],        # GAN discriminator loss
        'val_domain_disc_loss': [],     # CORAL loss (replaces domain discriminator)
        'val_est_loss_source': [],      # source estimation loss
        'val_est_loss_target': [],      # target estimation loss  
        'val_est_loss': [],             # average estimation loss
        'source_acc': [],               # source domain accuracy (placeholder for CORAL)
        'target_acc': [],               # target domain accuracy (placeholder for CORAL)
        'acc': [],                      # average accuracy (placeholder for CORAL)
        'nmse_val_source': [],          # source NMSE
        'nmse_val_target': [],          # target NMSE
        'nmse_val': [],                  # average NMSE
        'val_smoothness_loss': []
    }
    #
    H_to_save = {}          # list to save to .mat file for H
    perform_to_save = {}    # list to save to .mat file for nmse, losses,...

    # 
    model = CNNGenerator(n_blocks=4, extract_layers=['block_2'])
    print("4 blocks, CORAL, extract layers = block 2")
    # 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    # 
    
    flag = 1 # flag to plot and save H_true
    epoc_pad = []    # epochs that calculating pad (return_features == True)
    for epoch in range(n_epochs):
        # get weights 
        weights = scheduler.get_weights_domain_first_smooth(epoch, n_epochs)
        print(f"Epoch {epoch+1}/{n_epochs}, Weights: {weights}")
        
        # ===================== Training =====================
        loader_H_true_train_source.reset()
        # loader_H_practical_train_source.reset()
        loader_H_input_train_source.reset()
        loader_H_true_train_target.reset()
        # loader_H_practical_train_target.reset()
        loader_H_input_train_target.reset()
                
        # loader_H = [loader_H_practical_train_source, loader_H_true_train_source, loader_H_practical_train_target, loader_H_true_train_target]
        loader_H = [loader_H_input_train_source, loader_H_true_train_source, loader_H_input_train_target, loader_H_true_train_target]

        # Only 2 loss functions
        loss_fn = [loss_fn_ce, loss_fn_bce]
    
        ##########################
        # if epoch==0 or epoch == n_epochs-1:
        #     # return_features == return features to calculate PAD
        #     return_features = True
        #     epoc_pad.append(epoch)
        # else:
        #     return_features = False

        ##########################
        # 
        train_step_output = train_step_cnn_residual_FDAfullTranslation1_coral(model, loader_H, loss_fn, optimizer, lower_range=-1, 
                        save_features=False, weights=weights, linear_interp=linear_interp,
                        fda_win_h=fda_win_h, fda_win_w=fda_win_w, fda_weight=fda_weight)

        train_epoc_loss_est        = train_step_output.avg_epoc_loss_est
        train_epoc_loss_d          = train_step_output.avg_epoc_loss_d
        train_epoc_loss_domain     = train_step_output.avg_epoc_loss_domain  # Now contains CORAL loss
        train_epoc_loss            = train_step_output.avg_epoc_loss
        train_epoc_loss_est_target = train_step_output.avg_epoc_loss_est_target
                # train_epoc_loss        = total train loss = loss_est + lambda_coral * coral_loss
                # train_epoc_loss_est    = loss in estimation network in source domain (labels available)
                # train_epoc_loss_domain = JMMD loss (statistical distribution matching)
                # train_epoc_loss_est_target - just to monitor - the machine can not calculate because no label available in source domain
                # All are already calculated in average over training dataset (source/target - respectively)
        print("Time", time.perf_counter() - start, "seconds")
        # Calculate PAD for the extracted features
        # if return_features and (weights['domain_weight']!=0) and (epoch==0 or epoch == n_epochs-1):
        #     features_source_file = "features_source.h5"
        #     features_target_file = "features_target.h5"
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
        #     #
        #     plotfig.plotHist(features_source_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'source_epoch_{epoch+1}', percent=100)
        #     plotfig.plotHist(features_target_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'target_epoch_{epoch+1}', percent=100)
        #     #
        #     # Calculate Wasserstein-1 distance for extracted features
        #     # print("Calculating Wasserstein-1 distance for extracted features ...")
        #     # w_dist_epoc = plotfig.wasserstein_approximate(features_source_file, features_target_file)
        #     # w_dist.append(w_dist_epoc)
            

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
        print(f"epoch {epoch+1}/{n_epochs} Average Disc Loss (in Source domain): {train_epoc_loss_d:.6f}")
        
        train_metrics['train_domain_loss'].append(train_epoc_loss_domain)
        print(f"epoch {epoch+1}/{n_epochs} Average CORAL Loss: {train_epoc_loss_domain:.6f}")  # Updated print message
        
        train_metrics['train_est_loss_target'].append(train_epoc_loss_est_target)
        print(f"epoch {epoch+1}/{n_epochs} For observation only - Average Estimation Loss in Target domain: {train_epoc_loss_est_target:.6f}")
        
        
        # ===================== Evaluation =====================
        loader_H_true_val_source.reset()
        loader_H_input_val_source.reset()
        loader_H_true_val_target.reset()
        loader_H_input_val_target.reset()
        loader_H_eval = [loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target]

        # 
        loss_fn = [loss_fn_ce, loss_fn_bce]
        
        # eval_func = utils_UDA_FiLM.val_step
        if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) and epoch!=n_epochs-1:
            # 
            H_sample, epoc_val_return = val_step_cnn_residual_coral(model, loader_H_eval, loss_fn, lower_range, 
                                            weights=weights, linear_interp=linear_interp)
            
            visualize_H(H_sample, H_to_save, epoch, plotfig.figChan, flag, model_path, sub_folder, domain_weight=weights['domain_weight'])
            flag = 0  # after the first epoch, no need to save H_true anymore
        elif epoch==n_epochs-1:
            _, epoc_val_return, H_val_gen = val_step_cnn_residual_coral(model, loader_H_eval, loss_fn, lower_range, 
                                            weights=weights, linear_interp=linear_interp, return_H_gen=True)
        else:
            # 
            _, epoc_val_return = val_step_cnn_residual_coral(model, loader_H_eval, loss_fn, lower_range, 
                                        weights=weights, linear_interp=linear_interp)
        
        post_val(epoc_val_return, epoch, n_epochs, val_metrics, domain_weight=weights['domain_weight'])
        
        if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) or epoch==n_epochs-1:
            # 
            all_metrics = {
                'figLoss': plotfig.figLoss, 
                'savemat': savemat,
                # 'pad_metrics': pad_metrics, 
                # 'epoc_pad': epoc_pad,
                # 'pad_svm': pad_svm, 
                'weights': weights, 
                'optimizer': optimizer
            }
            # Combine all metrics
            all_metrics.update(train_metrics)  # Add training metrics
            all_metrics.update(val_metrics)    # Add validation metrics

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

