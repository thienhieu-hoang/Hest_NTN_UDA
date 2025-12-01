import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import os
import sys
import numpy as np

import scipy
print(f"scipy version: {scipy.__version__}")

# import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "POT"])

import ot
print(f"POT version: {ot.__version__}")

from scipy.io import savemat, loadmat
from sklearn.linear_model import SGDClassifier

script_dir = os.path.dirname(__file__)
print(script_dir)
notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..'))
print(notebook_dir) # need to be in Domain_Adversarial/
notebook_dir

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..', 'helper')))
# import utils
# import loader
print('Append path', os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..', 'helper')))
import utils_GAN, PAD


import utils
import loader
import plotfig


SNR = -5
# source_data_file_path_label = os.path.abspath(os.path.join(notebook_dir, '..', 'generatedChan', 'OpenNTN','H_perfect.mat'))
# target_data_file_path = os.path.abspath(os.path.join(notebook_dir, '..', 'generatedChan', 'OpenNTN', f'SNR_{SNR}dB','sionnaNTN.mat'))
target_data_file_path = os.path.abspath(os.path.join(notebook_dir, '..', 'generatedChan', 'MATLAB', 'TDL_B_100_300_sim', f'SNR_{SNR}dB','matlabNTN.mat'))
source_data_file_path = os.path.abspath(os.path.join(notebook_dir, '..', 'generatedChan', 'MATLAB', 'TDL_D_30_sim', f'SNR_{SNR}dB','matlabNTN.mat'))

norm_approach = 'minmax' # can be set to 'std'
lower_range = -1 
    # if norm_approach = 'minmax': 
        # =  0 for scaling to  [0 1]
        # = -1 for scaling to [-1 1]
    # if norm_approach = 'std': can be any value, but need to be defined

gen_lr=1e-4
disc_lr=1e-5
domain_lr=5e-6

scheduler = utils_GAN.WeightScheduler(strategy='reconstruction_first', start_domain_weight=0.005, end_domain_weight=0.05,
                            start_est_weight=1.2, end_est_weight=0.8) 
                            # adv_weight = 0.005 default

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
path_temp = script_dir + f'/results/'
os.makedirs(os.path.dirname(path_temp), exist_ok=True)
idx_save_path = loader.find_incremental_filename(path_temp,'ver', '_', '')

save_model = False
model_path = script_dir + f'/results/ver' + str(idx_save_path) + '_'
# figure_path = notebook_dir + '/model/GAN/ver' + str(idx_save_path) + '_/figure'
model_readme = model_path + '/readme.txt'

import h5py
import scipy.io

batch_size=16

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

class DataLoaders:
    def __init__(self, file, indices_train, indices_val, tag='prac', batch_size=32): 
        # tag = 'prac' or 'li' or 'ls'
        self.true_train = utils.H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_train)
        self.true_val = utils.H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_val)

        self.input_train = utils.H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_train)
        self.input_val = utils.H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_val)

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
loss_fn_domain = tf.keras.losses.BinaryCrossentropy()  # Domain classification loss

load_checkpoint = False  # True if continue training
if load_checkpoint:
    # model_path = notebook_dir + '/model/GAN_cal_A300_B100_300/ver' + str(idx_save_path-1) + '_' # or replace idx_save_path-1 by the desired folder index
    model_path = notebook_dir + f'/model/GAN_cal_A300_B100_300/{SNR}_dB/ver' + str(idx_save_path-1) + '_'
if load_checkpoint:
    start_epoch = 3  # This is the epoch we want to CONTINUE FROM (not load from)
else:
    start_epoch = 0    

import time
start = time.perf_counter()

n_epochs= 300
epoch_min = 20
epoch_step = 20
# n_epochs= 5
# epoch_min = 0
# epoch_step = 1

sub_folder = 'GAN_practical'  # 'GAN_linear', 'GAN_practical', 'GAN_ls'
print(f"Processing: {sub_folder}")

w_dist = []
pad_pca_lda = []
pad_pca_logreg = []
pad_pca_svm = []
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
plotfig.plotHist(loader_H_input_train_source, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=99)
plotfig.plotHist(loader_H_input_train_target, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=99)

plotfig.plotHist(loader_H_input_train_source, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=95)
plotfig.plotHist(loader_H_input_train_target, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=95)

plotfig.plotHist(loader_H_input_train_source, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=90)
plotfig.plotHist(loader_H_input_train_target, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=90)

# Calculate Wasserstein-1 distance for original input training datasets (before training)
if load_checkpoint==False:
    print("Calculating Wasserstein-1 distance for original input training datasets (before training)...")
    w_dist_epoc = plotfig.wasserstein_approximate(loader_H_input_train_source, loader_H_input_train_target)
    w_dist.append(w_dist_epoc)

    # Calculate     PAD for original input training datasets with SVM
    pad_svm = PAD.original_PAD(loader_H_input_train_source, loader_H_input_train_target)
    print(f"PAD = {pad_svm:.4f}")

    # Calculate PCA_PAD for original input training datasets with PCA_SVM, PCA_LDA, PCA_LogReg
    X_features_, y_features_ = PAD.extract_features_with_pca(loader_H_input_train_source, loader_H_input_train_target, pca_components=100)
    pad_pca_svm_epoc = PAD.calc_pad_svm(X_features_, y_features_)
    pad_pca_lda_epoc = PAD.calc_pad_lda(X_features_, y_features_)
    pad_pca_logreg_epoc = PAD.calc_pad_logreg(X_features_, y_features_)

    pad_pca_lda.append(pad_pca_lda_epoc)
    pad_pca_logreg.append(pad_pca_logreg_epoc)
    pad_pca_svm.append(pad_pca_svm_epoc)
## 

if not os.path.exists(os.path.dirname(model_path + '/' + sub_folder +'/')):
    os.makedirs(os.path.dirname(model_path + '/' + sub_folder + '/'))   # Domain_Adversarial/model/_/ver_/{sub_folder}

flag = 1 # flag to plot and save H_true
H_to_save = {} 

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
    model_domain = utils_GAN.DomainDisc()
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=0.5, beta_2=0.9)  # WGAN-GP uses Adam optimizer with beta_1=0.5
    domain_optimizer = tf.keras.optimizers.Adam(learning_rate=domain_lr)
    ####
    optimizer = [gen_optimizer, disc_optimizer, domain_optimizer]
    ####

    epoc_pad = []    # epochs in which we calculate pad (return_features == True)
    pad_pca_lda = []
    pad_pca_logreg = []
    pad_pca_svm = []
    pad_svm = 0
else:   # load from check_point
    model = utils_GAN.GAN(n_subc=132, gen_l2=None, disc_l2=1e-5)
    # model.build(input_shape=(None, 16, 312, 14, 2))
    dummy_input = tf.random.normal((batch_size, 132, 14, 2))
    _ = model(dummy_input)  # This builds the model with proper weights initialization
    # Build domain discriminator
    model_domain = utils_GAN.DomainDisc()
    dummy_input = tf.random.normal((batch_size, 7, 14, 256))
    _ = model_domain(dummy_input)
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
    
    print("=== Optimizer State Verification ===")
    print(f"Generator optimizer learning rate: {gen_optimizer.learning_rate.numpy()}")
    print(f"Discriminator optimizer learning rate: {disc_optimizer.learning_rate.numpy()}")
    print(f"Domain optimizer learning rate: {domain_optimizer.learning_rate.numpy()}")

    # Check if optimizers have momentum/state from previous training
    print(f"Gen optimizer iterations: {gen_optimizer.iterations.numpy()}")
    print(f"Disc optimizer iterations: {disc_optimizer.iterations.numpy()}")
    print(f"Domain optimizer iterations: {domain_optimizer.iterations.numpy()}")
    
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
    #
    epoc_pad             = loadmat_params['epoc_pad'].flatten().tolist()
    pad_pca_lda          = loadmat_params['pad_pca_lda'].flatten().tolist()
    pad_pca_logreg       = loadmat_params['pad_pca_logreg'].flatten().tolist()
    pad_pca_svm          = loadmat_params['pad_pca_svm'].flatten().tolist()
    pad_svm              = loadmat_params['pad_svm']

    print(f"Loaded {len(train_loss)} epochs of training history.")
    print(f"Last loaded training loss: {train_loss[-1] if train_loss else 'No history'}")
####

for epoch in range(n_epochs):
    # get weights 
    weights = scheduler.get_weights_domain_first_smooth(epoch, n_epochs)
    print(f"Epoch {epoch+1}/{n_epochs}, Weights: {weights}")
    adv_weight = weights['adv_weight']
    est_weight = weights['est_weight'] 
    domain_weight = weights['domain_weight']
    
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
    if epoch in [int(n_epochs * r) for r in [0, 0.25, 0.5, 0.75]] or epoch == n_epochs-1:
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
        ## Calculate PCA_PAD for extracted features with PCA_SVM, PCA_LDA, PCA_LogReg
        X_features, y_features = PAD.extract_features_with_pca(features_source_file, features_target_file, pca_components=100)
        pad_svm_epoc = PAD.calc_pad_svm(X_features, y_features)
        pad_pca_svm.append(pad_svm_epoc)
        #
        pad_lda_epoc = PAD.calc_pad_lda(X_features, y_features)
        pad_pca_lda.append(pad_lda_epoc)
        #
        pad_logreg_epoc = PAD.calc_pad_logreg(X_features, y_features)
        pad_pca_logreg.append(pad_logreg_epoc)
        
        ## Distribution of extracted features
        plotfig.plotHist(features_source_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'source_epoch_{epoch+1}', percent=99)
        plotfig.plotHist(features_target_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'target_epoch_{epoch+1}', percent=99)
        #
        plotfig.plotHist(features_source_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'source_epoch_{epoch+1}', percent=95)
        plotfig.plotHist(features_target_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'target_epoch_{epoch+1}', percent=95)
        #
        plotfig.plotHist(features_source_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'source_epoch_{epoch+1}', percent=90)
        plotfig.plotHist(features_target_file, fig_show = False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'target_epoch_{epoch+1}', percent=90)
        # Calculate Wasserstein-1 distance for extracted features
        # print("Calculating Wasserstein-1 distance for extracted features ...")
        # w_dist_epoc = plotfig.wasserstein_approximate(features_source_file, features_target_file)
        # w_dist.append(w_dist_epoc)
        

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
    if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) and epoch!=n_epochs-1:
        H_sample, epoc_val_return = utils_GAN.val_step_wgan_gp(model, model_domain, loader_H_eval, loss_fn, lower_range, 
                                        adv_weight=adv_weight, est_weight=est_weight, domain_weight=domain_weight, linear_interp=linear_interp)
        utils_GAN.visualize_H(H_sample, H_to_save, epoch, plotfig.figChan, flag, model_path, sub_folder, domain_weight=domain_weight)
        flag = 0  # after the first epoch, no need to save H_true anymore

    elif epoch==n_epochs-1: # last epoch   
        _, epoc_val_return, H_val_gen = utils_GAN.val_step_wgan_gp(model, model_domain, loader_H_eval, loss_fn, lower_range, 
                                        adv_weight=adv_weight, est_weight=est_weight, domain_weight=domain_weight, 
                                        linear_interp=linear_interp, return_H_gen=True)
        
    else:
        _, epoc_val_return = utils_GAN.val_step_wgan_gp(model, model_domain, loader_H_eval, loss_fn, lower_range, 
                                        adv_weight=adv_weight, est_weight=est_weight, domain_weight=domain_weight, linear_interp=linear_interp)
    
    utils_GAN.post_val(epoc_val_return, epoch, n_epochs, val_est_loss, val_est_loss_source, val_loss, val_est_loss_target,
        val_gan_disc_loss, val_domain_disc_loss, nmse_val_source, nmse_val_target, nmse_val, source_acc, target_acc, acc, domain_weight=domain_weight)
    
    
    if (epoch==epoch_min) or (epoch+1>epoch_min and (epoch-epoch_min)%epoch_step==0) or epoch==n_epochs-1:
        utils_GAN.save_checkpoint(model, save_model, model_path, sub_folder, epoch, plotfig.figLoss, savemat, train_loss, train_est_loss, train_domain_loss, train_est_loss_target,
                val_est_loss, val_est_loss_source, val_loss, val_est_loss_target, val_gan_disc_loss, val_domain_disc_loss,
                source_acc, target_acc, acc, nmse_val_source, nmse_val_target, nmse_val, pad_pca_svm, pad_pca_lda, pad_pca_logreg, epoc_pad, pad_svm, train_disc_loss, 
                domain_weight=domain_weight, optimizer=optimizer, domain_model=model_domain)
        
# end of epoch loop
# =====================            
# Save performances
# Save H matrix
savemat(model_path + '/' + sub_folder + '/H_visualize/H_trix.mat', H_to_save)
savemat(model_path + '/' + sub_folder + '/H_visualize/H_val_generated.mat', 
        {'H_val_gen': H_val_gen,
        'indices_val_source': indices_val_source,
        'indices_val_target': indices_val_target})

    