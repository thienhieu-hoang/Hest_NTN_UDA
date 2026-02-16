import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
import h5py
import sys
import os 

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..'))
except NameError:
    # Running in Jupyter Notebook
    notebook_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(notebook_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from JMMD.helper.utils_GAN import CNNGenerator
from Domain_Adversarial.helper.utils_GAN import train_step_Output

from Domain_Adversarial.helper.utils import minmaxScaler, complx2real, deMinMax
from Domain_Adversarial.helper.utils_GAN import gradient_penalty

from JMMD.helper.utils_GAN import compute_total_smoothness_loss


# Domain labels
def make_domain_labels(batch_size, domain):
    return tf.ones((batch_size, 1)) if domain == 'source' else tf.zeros((batch_size, 1))


class DomainDiscForCNN(tf.keras.Model):
    """
    Domain Discriminator for CNN residual features from JMMD generator.
    
    Input: (batch, 132, 14, C) - CNN-extracted features
    Output: (batch, 1) - domain probability (1=source, 0=target)
    
    Architecture:
    - Layer 1: kernel=(4,3), stride=(2,1) → (65, 12)
    - Layer 2: kernel=(3,3), stride=(2,1) → (32, 10)
    - Layer 3: kernel=(4,3), stride=(2,1) → (15, 9)
    - Layer 4: kernel=(3,3), stride=(2,1) → (7, 6)
    - Global Average Pooling → (C_out,)
    - Dense layers for binary classification
    """
    def __init__(self, l2_reg=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg is not None else None
        
        # Layer 1: (132, 14) → (65, 12)
        self.conv1 = tf.keras.layers.Conv2D(
            256, kernel_size=(4,3), strides=(2,1), padding='valid',
            activation='relu', kernel_regularizer=kernel_regularizer
        )
        
        # Layer 2: (65, 12) → (32, 10)
        self.conv2 = tf.keras.layers.Conv2D(
            256, kernel_size=(3,3), strides=(2,1), padding='valid',
            activation='relu', kernel_regularizer=kernel_regularizer
        )
        
        # Layer 3: (32, 10) → (15, 9)
        self.conv3 = tf.keras.layers.Conv2D(
            128, kernel_size=(4,3), strides=(2,1), padding='valid',
            activation='relu', kernel_regularizer=kernel_regularizer
        )
        
        # Layer 4: (15, 9) → (7, 6)
        self.conv4 = tf.keras.layers.Conv2D(
            64, kernel_size=(3,3), strides=(2,1), padding='valid',
            activation='relu', kernel_regularizer=kernel_regularizer
        )
        
        # Global Average Pooling → (64,) - pools over 7*6=42 spatial positions
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # Dense classification layers
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        """
        Forward pass for domain discriminator
        
        Args:
            x: Input features tensor
            training: Boolean flag for training/inference mode
            
        Returns:
            Domain probability (batch, 1)
        """
        x = self.conv1(x)  # (batch, 65, 12, 256)
        x = self.conv2(x)  # (batch, 32, 10, 256)
        x = self.conv3(x)  # (batch, 15, 9, 128)
        x = self.conv4(x)  # (batch, 7, 6, 64)
        x = self.pool(x)   # (batch, 64) - spatial dimensions collapsed
        x = self.fc1(x)    # (batch, 64)
        if training:
            x = self.dropout(x, training=training)
        return self.out(x) # (batch, 1) - domain probability            

def train_step_cnn_residual_dann(model_cnn, domain_disc, loader_H, loss_fn, optimizers, lower_range=-1, 
                                save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    CNN residual training step with DANN (Domain Adversarial Neural Networks)
    
    Model predicts residual correction + domain discriminator for adversarial adaptation
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        domain_disc: Domain discriminator (DomainDiscForCNN instance)
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss)
        optimizers: tuple of (cnn_optimizer, domain_optimizer)
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for analysis
        nsymb: number of symbols
        weights: weight dictionary with keys: est_weight, adv_weight, domain_weight, temporal_weight, frequency_weight
        linear_interp: linear interpolation flag
    
    Returns:
        train_step_Output: Dataclass with loss metrics and feature information
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn
    cnn_optimizer, domain_optimizer = optimizers
    
    # Extract weights with defaults
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    adv_weight = weights.get('adv_weight', 0.01) if weights else 0.01
    domain_weight = weights.get('domain_weight', 0.5) if weights else 0.5
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Metrics tracking
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_adv_gen = 0.0
    epoc_loss_adv_disc = 0.0
    epoc_residual_norm = 0.0
    epoc_domain_acc = 0.0
    N_train = 0
    
    # Feature saving setup
    if save_features and (adv_weight != 0):
        features_h5_path_source = 'features_source_dann.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target_dann.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    # ============ BATCH LOOP ============
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get and preprocess source data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        N_train += x_src.shape[0]

        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Get and preprocess target data
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()

        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        batch_size = x_src.shape[0]
        domain_labels_src = make_domain_labels(batch_size, 'source')  # 1 for source
        domain_labels_tgt = make_domain_labels(batch_size, 'target')  # 0 for target
        
        # ============ STEP 1: Train Domain Discriminator ============
        with tf.GradientTape() as tape_disc:
            # Get CNN features
            residual_src, features_src = model_cnn(x_scaled_src, training=True, return_features=True)
            residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=True, return_features=True)
            
            # Extract single layer from list (DANN uses only one layer)
            if isinstance(features_src, list):
                features_src = features_src[-1]  # Use last layer: (8, 132, 14, 128)
            if isinstance(features_tgt, list):
                features_tgt = features_tgt[-1]  # Use last layer: (8, 132, 14, 128)
            
            # Domain discriminator predictions
            domain_pred_src = domain_disc(features_src, training=True)
            domain_pred_tgt = domain_disc(features_tgt, training=True)
            
            # Domain discrimination loss (cross-entropy)
            disc_loss_src = loss_fn_bce(domain_labels_src, domain_pred_src)
            disc_loss_tgt = loss_fn_bce(domain_labels_tgt, domain_pred_tgt)
            disc_loss = disc_loss_src + disc_loss_tgt
            
            # Add L2 regularization from discriminator
            if domain_disc.losses:
                disc_loss += tf.add_n(domain_disc.losses)
        
        # Update discriminator
        grads_disc = tape_disc.gradient(disc_loss, domain_disc.trainable_variables)
        domain_optimizer.apply_gradients(zip(grads_disc, domain_disc.trainable_variables))
        
        # ============ STEP 2: Train CNN with Adversarial Loss ============
        with tf.GradientTape() as tape_cnn:
            # Get CNN features
            residual_src_cnn, features_src_cnn = model_cnn(x_scaled_src, training=True, return_features=True)
            x_corrected_src = x_scaled_src + residual_src_cnn  # Apply residual correction
            
            residual_tgt_cnn, features_tgt_cnn = model_cnn(x_scaled_tgt, training=True, return_features=True)
            x_corrected_tgt = x_scaled_tgt + residual_tgt_cnn  # Apply residual correction
            
            # Extract single layer
            if isinstance(features_src_cnn, list):
                features_src_cnn = features_src_cnn[-1]
            if isinstance(features_tgt_cnn, list):
                features_tgt_cnn = features_tgt_cnn[-1]
            
            # --- Estimation Loss (on source domain) ---
            est_loss = loss_fn_est(y_scaled_src, x_corrected_src)
            
            # --- Adversarial Loss (fool discriminator with target features) ---
            # For DANN: We want domain_disc to predict source (1) for target features
            # This forces CNN to make target features look like source features
            domain_pred_tgt_adv = domain_disc(features_tgt_cnn, training=True)
            adv_loss = loss_fn_bce(
                domain_labels_src,  # Pretend target is source (label=1)
                domain_pred_tgt_adv  # But discriminator sees it's target
            )  # High loss when discriminator correctly identifies target
            
            # --- Residual Regularization ---
            residual_reg = 0.001 * (tf.reduce_mean(tf.square(residual_src_cnn)) + 
                                    tf.reduce_mean(tf.square(residual_tgt_cnn)))
            
            # --- Smoothness Loss ---
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss_src = compute_total_smoothness_loss(x_corrected_src, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                smoothness_loss_tgt = compute_total_smoothness_loss(x_corrected_tgt, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            else:
                smoothness_loss = 0.0
            
            # --- Total CNN Loss ---
            total_loss_cnn = (est_weight * est_loss + 
                            adv_weight * adv_loss + 
                            residual_reg +
                            smoothness_loss)
            
            # Add L2 regularization from CNN
            if model_cnn.losses:
                total_loss_cnn += tf.add_n(model_cnn.losses)
        
        # Update CNN
        grads_cnn = tape_cnn.gradient(total_loss_cnn, model_cnn.trainable_variables)
        cnn_optimizer.apply_gradients(zip(grads_cnn, model_cnn.trainable_variables))
        
        # ============ Save Features ============
        if save_features and (adv_weight != 0):
            features_np_source = features_src.numpy() if isinstance(features_src, list) else features_src.numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features',
                    data=features_np_source,
                    maxshape=(None,) + features_np_source.shape[1:],
                    chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_source.shape[0], axis=0)
                features_dataset_source[-features_np_source.shape[0]:] = features_np_source
                
            features_np_target = features_tgt.numpy() if isinstance(features_tgt, list) else features_tgt.numpy()
            if features_dataset_target is None:
                features_dataset_target = features_h5_target.create_dataset(
                    'features',
                    data=features_np_target,
                    maxshape=(None,) + features_np_target.shape[1:],
                    chunks=True
                )
            else:
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
                features_dataset_target[-features_np_target.shape[0]:] = features_np_target
        
        # ============ Track Metrics ============
        epoc_loss_total += total_loss_cnn.numpy() * batch_size
        epoc_loss_est += est_loss.numpy() * batch_size
        epoc_loss_adv_gen += adv_loss.numpy() * batch_size
        epoc_loss_adv_disc += disc_loss.numpy() * batch_size
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_src)).numpy() * batch_size
        
        # Domain accuracy: percentage of correct domain predictions
        pred_binary_src = (domain_pred_src.numpy() > 0.5).astype(np.float32)
        pred_binary_tgt = (domain_pred_tgt.numpy() > 0.5).astype(np.float32)
        acc_src = accuracy_score(domain_labels_src.numpy(), pred_binary_src)
        acc_tgt = accuracy_score(domain_labels_tgt.numpy(), pred_binary_tgt)
        epoc_domain_acc += (acc_src + acc_tgt) / 2 * batch_size
    
    # ============ End Batch Loop ============
    if save_features and (adv_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate epoch averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_adv_gen = epoc_loss_adv_gen / N_train
    avg_loss_adv_disc = epoc_loss_adv_disc / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_domain_acc = epoc_domain_acc / N_train
    
    # Print metrics
    print(f"    Est Loss: {avg_loss_est:.6f} | Adv Gen Loss: {avg_loss_adv_gen:.6f} | " +
            f"Adv Disc Loss: {avg_loss_adv_disc:.6f} | Residual Norm: {avg_residual_norm:.6f} | " +
            f"Domain Acc: {avg_domain_acc:.4f}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_adv_gen,
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_adv_disc,
        features_source= None,
        film_features_source= None,
        avg_epoc_loss_d=avg_loss_adv_disc
    )
    
def val_step_cnn_residual_dann(model_cnn, domain_disc, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                            linear_interp=False, return_H_gen=False):
    """
    Validation step for CNN residual learning with DANN (WITH domain discriminator evaluation)
    
    During validation, we evaluate:
    - Estimation loss (MSE) on source and target
    - Domain discriminator loss (how well it distinguishes domains)
    - Domain classification accuracy
    - Residual magnitude
    - Optional: smoothness of corrected channels
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        domain_disc: Domain discriminator (DomainDiscForCNN instance)
        loader_H: tuple of validation loaders (src_input, src_true, tgt_input, tgt_true)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
    
    Returns:
        H_sample: Sample predictions for visualization
        epoc_eval_return: Dictionary with validation metrics
        H_gen (optional): All generated H matrices if return_H_gen=True
    """
    loader_H_input_val_source, loader_H_true_val_source, \
        loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn
    
    # Extract weights
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    adv_weight = weights.get('adv_weight', 0.01) if weights else 0.01
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Metrics tracking
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_domain_disc_loss = 0.0
    epoc_domain_adv_loss = 0.0
    epoc_domain_acc_source = 0.0
    epoc_domain_acc_target = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm_src = 0.0
    epoc_residual_norm_tgt = 0.0
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    # ============ VALIDATION LOOP ============
    for idx in range(loader_H_true_val_source.total_batches):
        # === Get and preprocess SOURCE data ===
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        N_val_source += x_src.shape[0]

        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # === Get and preprocess TARGET data ===
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_target += x_tgt.shape[0]

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # Domain labels
        batch_size = x_src.shape[0]
        domain_labels_src = make_domain_labels(batch_size, 'source')  # 1 for source
        domain_labels_tgt = make_domain_labels(batch_size, 'target')  # 0 for target

        # === SOURCE domain prediction (residual learning with features) ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Convert to numpy safely
        preds_src_numpy = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = deMinMax(preds_src_numpy, x_min_src, x_max_src, lower_range=lower_range)
        
        # Estimation loss and NMSE for source
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]
        
        # Track residual magnitude for source
        residual_src_norm = tf.reduce_mean(tf.abs(residual_src)).numpy()
        epoc_residual_norm_src += residual_src_norm * x_src.shape[0]

        # === TARGET domain prediction (residual learning with features) ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Convert to numpy safely
        preds_tgt_numpy = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt_numpy, x_min_tgt, x_max_tgt, lower_range=lower_range)
        
        # Estimation loss and NMSE for target
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]
        
        # Track residual magnitude for target
        residual_tgt_norm = tf.reduce_mean(tf.abs(residual_tgt)).numpy()
        epoc_residual_norm_tgt += residual_tgt_norm * x_tgt.shape[0]

        # === DOMAIN DISCRIMINATOR EVALUATION ===
        # Domain discriminator predictions
        # Extract single layer
        if isinstance(features_src, list):
            features_src = features_src[-1]
        if isinstance(features_tgt, list):
            features_tgt = features_tgt[-1]
            
        domain_pred_src = domain_disc(features_src, training=False)
        domain_pred_tgt = domain_disc(features_tgt, training=False)
        
        # Domain discrimination loss (how well discriminator distinguishes domains)
        disc_loss_src = loss_fn_bce(domain_labels_src, domain_pred_src).numpy()
        disc_loss_tgt = loss_fn_bce(domain_labels_tgt, domain_pred_tgt).numpy()
        batch_disc_loss = disc_loss_src + disc_loss_tgt
        epoc_domain_disc_loss += batch_disc_loss * x_src.shape[0]
        
        # Adversarial loss (how well CNN fools the discriminator)
        # For target: CNN wants discriminator to predict source (1) but it's actually target (0)
        adv_loss_tgt = loss_fn_bce(domain_labels_src, domain_pred_tgt).numpy()
        epoc_domain_adv_loss += adv_loss_tgt * x_tgt.shape[0]
        
        # Domain classification accuracy
        pred_binary_src = (domain_pred_src.numpy() > 0.5).astype(np.float32)
        pred_binary_tgt = (domain_pred_tgt.numpy() > 0.5).astype(np.float32)
        acc_src = accuracy_score(domain_labels_src.numpy(), pred_binary_src)
        acc_tgt = accuracy_score(domain_labels_tgt.numpy(), pred_binary_tgt)
        epoc_domain_acc_source += acc_src * x_src.shape[0]
        epoc_domain_acc_target += acc_tgt * x_tgt.shape[0]

        # === Smoothness loss (optional, for monitoring only) ===
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

        # === Save sample predictions (first batch only) ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            
            # Source samples
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            
            # Target samples
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
            
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            
            H_sample = [H_true_sample, H_input_sample, H_est_sample, nmse_input_source, nmse_est_source,
                    H_true_sample_target, H_input_sample_target, H_est_sample_target, 
                    nmse_input_target, nmse_est_target]
        
        # === Save all generated H matrices (optional) ===
        if return_H_gen:
            all_H_gen_src.append(preds_src_descaled.numpy().copy())
            all_H_gen_tgt.append(preds_tgt_descaled.numpy().copy())
    
    # ============ END VALIDATION LOOP ============
    
    if return_H_gen:
        H_gen_src_all = np.concatenate(all_H_gen_src, axis=0)
        H_gen_tgt_all = np.concatenate(all_H_gen_tgt, axis=0)
        H_gen = {
            'H_gen_src': H_gen_src_all,
            'H_gen_tgt': H_gen_tgt_all
        }

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    
    avg_domain_disc_loss = epoc_domain_disc_loss / (N_val_source + N_val_target)  # disc loss when training Domain Disc
    avg_domain_adv_loss = epoc_domain_adv_loss / N_val_target # disc loss when training Estimator 
    
    avg_domain_acc_source = epoc_domain_acc_source / N_val_source
    avg_domain_acc_target = epoc_domain_acc_target / N_val_target
    avg_domain_acc = (avg_domain_acc_source + avg_domain_acc_target) / 2
    
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm_src = epoc_residual_norm_src / N_val_source
    avg_residual_norm_tgt = epoc_residual_norm_tgt / N_val_target
    avg_residual_norm = (avg_residual_norm_src + avg_residual_norm_tgt) / 2
    
    # Print validation statistics
    print(f"    Val Est Loss - Source: {avg_loss_est_source:.6f} | Target: {avg_loss_est_target:.6f}")
    print(f"    Val NMSE - Source: {avg_nmse_source:.6f} | Target: {avg_nmse_target:.6f}")
    print(f"    Val Domain Disc Loss: {avg_domain_disc_loss:.6f} | Domain Adv Loss: {avg_domain_adv_loss:.6f}")
    print(f"    Val Domain Acc - Source: {avg_domain_acc_source:.4f} | Target: {avg_domain_acc_target:.4f} | Overall: {avg_domain_acc:.4f}")
    print(f"    Val Residual Norm - Source: {avg_residual_norm_src:.6f} | Target: {avg_residual_norm_tgt:.6f}")

    # Total validation loss (includes all components for monitoring)
    avg_total_loss = (est_weight * avg_loss_est + 
                    adv_weight * avg_domain_adv_loss + 
                    avg_smoothness_loss)

    # Return comprehensive validation metrics
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_domain_disc_loss': avg_domain_disc_loss,  # How well discriminator works
        'avg_domain_adv_loss': avg_domain_adv_loss,    # How well CNN fools discriminator
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss,
        'avg_residual_norm': avg_residual_norm
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return