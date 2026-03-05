import tensorflow as tf
import numpy as np
import h5py
import os
import sys 
from dataclasses import dataclass
from sklearn.decomposition import IncrementalPCA

from tensorflow import keras

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..'))
except NameError:
    # Running in Jupyter Notebook
    notebook_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(notebook_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Domain_Adversarial.helper.utils import minmaxScaler, complx2real, deMinMax
from Domain_Adversarial.helper.utils_GAN import gradient_penalty
from JMMD.helper.utils_GAN import GlobalPoolingCORALLoss, compute_total_smoothness_loss, train_step_Output, save_features_with_incremental_pca


def train_step_cnn_residual_pseudo(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                        save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    CNN-only residual training step on pseudo domain
    Identical to source-only training, but uses pseudo domain data
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src,
                        loader_H_input_train_tgt, loader_H_true_train_tgt,
                        loader_H_input_train_pseudo, loader_H_true_train_pseudo)
                - src/tgt not used during training, only pseudo
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - only first one used
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
    """
    # Unpack loaders - only use pseudo for training
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt, \
        loader_H_input_train_pseudo, loader_H_true_train_pseudo = loader_H
    
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_residual_norm = 0.0
    N_train = 0
    
    # Train on PSEUDO domain only
    for batch_idx in range(loader_H_true_train_pseudo.total_batches):
        x_pseudo = loader_H_input_train_pseudo.next_batch()
        y_pseudo = loader_H_true_train_pseudo.next_batch()
        N_train += x_pseudo.shape[0]

        # Preprocess pseudo data
        x_pseudo = complx2real(x_pseudo)
        y_pseudo = complx2real(y_pseudo)
        x_pseudo = np.transpose(x_pseudo, (0, 2, 3, 1))
        y_pseudo = np.transpose(y_pseudo, (0, 2, 3, 1))
        x_scaled_pseudo, x_min_pseudo, x_max_pseudo = minmaxScaler(x_pseudo, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_pseudo, _, _ = minmaxScaler(y_pseudo, min_pre=x_min_pseudo, max_pre=x_max_pseudo, lower_range=lower_range)

        # Train CNN with RESIDUAL learning on pseudo domain
        with tf.GradientTape() as tape:
            # RESIDUAL LEARNING: Predict correction
            residual_pseudo, features_pseudo = model_cnn(x_scaled_pseudo, training=True, return_features=True)
            x_corrected_pseudo = x_scaled_pseudo + residual_pseudo
            
            # Estimation loss
            est_loss = loss_fn_est(y_scaled_pseudo, x_corrected_pseudo)
            
            # Residual regularization
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_pseudo))
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss = compute_total_smoothness_loss(x_corrected_pseudo, 
                                                            temporal_weight=temporal_weight, 
                                                            frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # Total loss
            total_loss = est_weight * est_loss + residual_reg + smoothness_loss
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)
        
        # Optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        epoc_loss_total += total_loss.numpy() * x_pseudo.shape[0]
        epoc_loss_est += est_loss.numpy() * x_pseudo.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_pseudo)).numpy() * x_pseudo.shape[0]

    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    
    print(f"    Pseudo residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    Loss (pseudo): {avg_loss_est:.6f}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=0.0,
        features_source=None,
        film_features_source=None,
        avg_epoc_loss_d=0.0
    )

def val_step_cnn_residual_pseudo(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                                    linear_interp=False, return_H_gen=False):
    """
    Validation step for CNN-only residual learning with CORAL (no discriminator)
    Validates on source, target, AND pseudo domains
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of validation loaders (src_input, src_label, tgt_input, tgt_label, pseudo_input, pseudo_label)
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
    """
    # Unpack 6 loaders if pseudo included, otherwise 4
    if len(loader_H) == 6:
        loader_H_input_val_source, loader_H_true_val_source, \
        loader_H_input_val_target, loader_H_true_val_target, \
        loader_H_input_val_pseudo, loader_H_true_val_pseudo = loader_H
        validate_pseudo = True
    else:
        loader_H_input_val_source, loader_H_true_val_source, \
        loader_H_input_val_target, loader_H_true_val_target = loader_H
        validate_pseudo = False
    
    loss_fn_est = loss_fn[0]
    
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    N_val_source = 0
    N_val_target = 0
    N_val_pseudo = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_est_pseudo = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_nmse_val_pseudo = 0.0
    epoc_coral_loss = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []
        all_H_gen_pseudo = []

    for idx in range(loader_H_true_val_source.total_batches):
        # === SOURCE AND TARGET DOMAINS ===
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        #
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        #
        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        #
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src
        
        if hasattr(preds_src, 'numpy'):
            preds_src_numpy = preds_src.numpy()
        else:
            preds_src_numpy = preds_src
        
        preds_src_descaled = deMinMax(preds_src_numpy, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt
        
        if hasattr(preds_tgt, 'numpy'):
            preds_tgt_numpy = preds_tgt.numpy()
        else:
            preds_tgt_numpy = preds_tgt
            
        preds_tgt_descaled = deMinMax(preds_tgt_numpy, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === PSEUDO DOMAIN VALIDATION (NEW) ===
        if validate_pseudo and idx < loader_H_true_val_pseudo.total_batches:
            x_pseudo = loader_H_input_val_pseudo.next_batch()
            y_pseudo = loader_H_true_val_pseudo.next_batch()
            N_val_pseudo += x_pseudo.shape[0]
            
            # Preprocessing
            x_pseudo_real = complx2real(x_pseudo)
            y_pseudo_real = complx2real(y_pseudo)
            x_pseudo_real = np.transpose(x_pseudo_real, (0, 2, 3, 1))
            y_pseudo_real = np.transpose(y_pseudo_real, (0, 2, 3, 1))
            x_scaled_pseudo, x_min_pseudo, x_max_pseudo = minmaxScaler(x_pseudo_real, lower_range=lower_range, linear_interp=linear_interp)
            y_scaled_pseudo, _, _ = minmaxScaler(y_pseudo_real, min_pre=x_min_pseudo, max_pre=x_max_pseudo, lower_range=lower_range)
            
            # Prediction
            residual_pseudo, _ = model_cnn(x_scaled_pseudo, training=False, return_features=True)
            preds_pseudo = x_scaled_pseudo + residual_pseudo
            
            if hasattr(preds_pseudo, 'numpy'):
                preds_pseudo_numpy = preds_pseudo.numpy()
            else:
                preds_pseudo_numpy = preds_pseudo
                
            preds_pseudo_descaled = deMinMax(preds_pseudo_numpy, x_min_pseudo, x_max_pseudo, lower_range=lower_range)
            batch_est_loss_pseudo = loss_fn_est(y_scaled_pseudo, preds_pseudo).numpy()
            epoc_loss_est_pseudo += batch_est_loss_pseudo * x_pseudo.shape[0]
            mse_val_pseudo = np.mean((preds_pseudo_descaled - y_pseudo_real) ** 2, axis=(1, 2, 3))
            power_pseudo = np.mean(y_pseudo_real ** 2, axis=(1, 2, 3))
            epoc_nmse_val_pseudo += np.mean(mse_val_pseudo / (power_pseudo + 1e-30)) * x_pseudo.shape[0]
            
            if return_H_gen:
                H_gen_pseudo_batch = preds_pseudo_descaled.numpy().copy() if hasattr(preds_pseudo_descaled, 'numpy') else preds_pseudo_descaled.copy()
                all_H_gen_pseudo.append(H_gen_pseudo_batch)

        #
        # if domain_weight > 0:
        #     coral_loss = coral_loss_fn(features_src, features_tgt)
        #     epoc_coral_loss += coral_loss.numpy() * x_src.shape[0]
        
        if temporal_weight != 0 or frequency_weight != 0:
            preds_src_tensor = tf.convert_to_tensor(preds_src_numpy) if not tf.is_tensor(preds_src) else preds_src
            preds_tgt_tensor = tf.convert_to_tensor(preds_tgt_numpy) if not tf.is_tensor(preds_tgt) else preds_tgt
            
            smoothness_loss_src = compute_total_smoothness_loss(
                preds_src_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
            )
            smoothness_loss_tgt = compute_total_smoothness_loss(
                preds_tgt_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
            )
            batch_smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            epoc_smoothness_loss += batch_smoothness_loss.numpy() * x_src.shape[0]

        #
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            if hasattr(preds_src_descaled, 'numpy'):
                H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample = preds_src_descaled[:n_samples].copy()
            
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            if hasattr(preds_tgt_descaled, 'numpy'):
                H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample_target = preds_tgt_descaled[:n_samples].copy()
            
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            
            H_sample = [H_true_sample, H_input_sample, H_est_sample, nmse_input_source, nmse_est_source,
                        H_true_sample_target, H_input_sample_target, H_est_sample_target, nmse_input_target, nmse_est_target]
            
        if return_H_gen:
            H_gen_src_batch = preds_src_descaled.numpy().copy() if hasattr(preds_src_descaled, 'numpy') else preds_src_descaled.copy()
            H_gen_tgt_batch = preds_tgt_descaled.numpy().copy() if hasattr(preds_tgt_descaled, 'numpy') else preds_tgt_descaled.copy()
            
            all_H_gen_src.append(H_gen_src_batch)
            all_H_gen_tgt.append(H_gen_tgt_batch)
    
    if return_H_gen:
        H_gen_src_all = np.concatenate(all_H_gen_src, axis=0)
        H_gen_tgt_all = np.concatenate(all_H_gen_tgt, axis=0)
        H_gen = {
            'H_gen_src': H_gen_src_all,
            'H_gen_tgt': H_gen_tgt_all
        }
        if validate_pseudo:
            H_gen_pseudo_all = np.concatenate(all_H_gen_pseudo, axis=0)
            H_gen['H_gen_pseudo'] = H_gen_pseudo_all

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    
    if validate_pseudo and N_val_pseudo > 0:
        avg_loss_est_pseudo = epoc_loss_est_pseudo / N_val_pseudo
        avg_nmse_pseudo = epoc_nmse_val_pseudo / N_val_pseudo
        print(f"    Validation Loss (pseudo): {avg_loss_est_pseudo:.6f}, NMSE (pseudo): {avg_nmse_pseudo:.6f}")
    
    avg_coral_loss = epoc_coral_loss / N_val_source if epoc_coral_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    avg_total_loss = est_weight * avg_loss_est + domain_weight * avg_coral_loss + avg_smoothness_loss

    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est_pseudo': avg_loss_est_pseudo,
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,
        'avg_domain_loss': avg_coral_loss,
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse_pseudo': avg_nmse_pseudo,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss
    }
    
    if validate_pseudo and N_val_pseudo > 0:
        epoc_eval_return['avg_loss_est_pseudo'] = avg_loss_est_pseudo
        epoc_eval_return['avg_nmse_pseudo'] = avg_nmse_pseudo

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return

def save_checkpoint_pseudo(model, save_model, model_path, sub_folder, epoch, metrics):
    exclude_keys = {'figLoss', 'savemat', 'optimizer'}  # keys to exclude
    
    # Create performance dictionary by excluding unwanted keys
    perform_to_save = {k: v for k, v in metrics.items() if k not in exclude_keys}
    
    # Extract needed items for other operations
    figLoss = metrics['figLoss']
    savemat = metrics['savemat'] 
    optimizer = metrics['optimizer']
    domain_weight = metrics['weights']['domain_weight']

    # Save model
    os.makedirs(f"{model_path}/{sub_folder}/model/", exist_ok=True)
    if save_model:
        # Create checkpoint with all model components and optimizers
        gen_optimizer, disc_optimizer = optimizer[:2]
        
        # Create checkpoint object
        ckpt = tf.train.Checkpoint(
            generator=model.generator,
            discriminator=model.discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer
        )
        
        
        # Create checkpoint manager - save only current epoch
        checkpoint_dir = f"{model_path}/{sub_folder}/model/"
        
        # Save checkpoint
        milestone_path = ckpt.save(f"{checkpoint_dir}/epoch_{epoch+1}")
        print(f"Checkpoint saved at epoch {epoch+1}: {milestone_path}")

        # Save optimizer configs
        optimizer_configs = {
            'gen_optimizer_config': gen_optimizer.get_config(),
            'disc_optimizer_config': disc_optimizer.get_config()
        }
        config_path = f"{checkpoint_dir}/optimizer_configs.json"  # No epoch number
        # Only save if file doesn't exist (to avoid overwriting)
        if not os.path.exists(config_path):
            import json
            with open(config_path, 'w') as f:
                json.dump(optimizer_configs, f, indent=2)
            print(f"Optimizer configs saved to: {config_path}")

    # save
    os.makedirs(f"{model_path}/{sub_folder}/performance/", exist_ok=True)
    savemat(model_path + '/' + sub_folder + '/performance/performance.mat', perform_to_save)
    
    # Plot figures === save and overwrite at checkpoints
    figLoss(line_list=[(metrics['nmse_val_source'], 'Source Domain'), (metrics['nmse_val_target'], 'Target Domain'), (metrics['nmse_val_pseudo'], 'Pseudo Domain')], xlabel='Epoch', ylabel='NMSE',
                title='NMSE in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='NMSE_val')

    figLoss(line_list=[(metrics['train_est_loss'], 'Train Loss - Pseudo Domain'), (metrics['val_est_loss_pseudo'], 'Val Loss - Pseudo Domain'), 
                    (metrics['val_est_loss_source'], 'Val Loss - Source Domain'), (metrics['val_est_loss_target'], 'Val Loss - Target Domain')], 
                xlabel='Epoch', ylabel='Loss',
                title='Estimation Losses', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Estimation_loss')
    
    # figLoss(line_list=[(metrics['train_est_loss'], 'GAN Generate Loss'), (metrics['train_disc_loss'], 'GAN Discriminator Loss')], xlabel='Epoch', ylabel='Loss',
    #             title='Training GAN Losses', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='GAN_train')
    
    # to plot from epoch 30 if length > 35:
    train_loss_data = metrics['train_loss'][30:] if len(metrics['train_loss']) > 35 else metrics['train_loss']
    val_loss_data = metrics['val_loss'][30:] if len(metrics['val_loss']) > 35 else metrics['val_loss']
    figLoss(line_list=[(train_loss_data, 'Training'), (val_loss_data, 'Validating')], xlabel='Epoch', ylabel='Total Loss',
            title='Training and Validating Total Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_total')
    ##
    
    if domain_weight!=0:
        if 'val_domain_loss' in metrics:
            figLoss(line_list=[(metrics['train_domain_loss'], 'Training'), (metrics['val_domain_loss'], 'Validating')], xlabel='Epoch', ylabel='Domain Loss',
                    title='Training and Validating Domain Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_domain')
        if 'val_domain_disc_loss' in metrics:
            figLoss(line_list=[(metrics['train_domain_loss'], 'Training'), (metrics['val_domain_disc_loss'], 'Validating')], xlabel='Epoch', ylabel='Domain Loss',
                    title='Training and Validating Domain Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_domain')
