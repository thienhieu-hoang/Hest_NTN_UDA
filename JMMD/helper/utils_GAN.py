""" helper functions and classes for GAN model 
"""
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

@dataclass
class train_step_Output:
    """Dataclass to hold the output of the train_step function."""
    avg_epoc_loss_est: float  # Average loss for generator 
    avg_epoc_loss_d: float  # Average loss for discriminator 
    avg_epoc_loss_domain: float
    avg_epoc_loss: float
    avg_epoc_loss_est_target: float  # Average loss for channel estimation on target domain
    features_source: tf.Tensor = None  # Features from the source domain, if return_features is True
    film_features_source: tf.Tensor = None  # Film features from the source domain, if return_features is True
    features_target: tf.Tensor = None  # Features from the target domain, if return_features is True
    film_features_target: tf.Tensor = None  # Film features from the target domain, if return_features is True
    pad: float = 0

# try:
#     # Works in .py scripts
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..'))
# except NameError:
#     # __file__ is not defined in Jupyter Notebook
#     notebook_dir = os.getcwd()
#     project_root = os.path.abspath(os.path.join(notebook_dir, '..', '..'))
# sys.path.append(project_root)

    
# Domain labels
def make_domain_labels(batch_size, domain):
    return tf.ones((batch_size, 1)) if domain == 'source' else tf.zeros((batch_size, 1))
    
        
def reflect_padding_2d(x, pad_h, pad_w):
    return tf.pad(x, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='SYMMETRIC')

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # gamma and beta for each channel
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="beta"
        )
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta
    
def reflect_padding_2d(x, pad_h, pad_w):
    return tf.pad(x, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='SYMMETRIC')
    
class UNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, apply_dropout=False, kernel_size=(4,3), strides=(2,1), pad_h=0, pad_w=1, gen_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides=strides, padding='valid',
                                            kernel_regularizer=kernel_regularizer)
        self.norm = InstanceNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None

    def call(self, x, training):
        if self.pad_h > 0 or self.pad_w > 0:
            x = reflect_padding_2d(x, pad_h=self.pad_h, pad_w=self.pad_w)  # symmetric padding
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = tf.nn.leaky_relu(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        return x
    
class UNetUpBlock(tf.keras.layers.Layer):
    def __init__(self, filters, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.deconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,
                                                        padding='valid', kernel_regularizer=kernel_regularizer)
        self.norm = InstanceNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None

    def call(self, x, skip, training):
        x = self.deconv(x)
        if x.shape[2] >14: 
            x = x[:, :, 1:15, :]
        x = self.norm(x, training=training)
        x = tf.nn.relu(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        x = tf.concat([x, skip], axis=-1)
        return x

class Pix2PixGenerator(tf.keras.Model):
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, extract_layers=['d2', 'd3', 'd4']):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.extract_layers = extract_layers
        # Encoder
        self.down1 = UNetBlock(32, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down2 = UNetBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.down3 = UNetBlock(128, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down4 = UNetBlock(256, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        # Decoder
        self.up1 = UNetUpBlock(128, apply_dropout=True, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.up2 = UNetUpBlock(64, apply_dropout=True, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.up3 = UNetUpBlock(32, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.last = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=(4,3), strides=(2,1), padding='valid',
                                                    activation='tanh', kernel_regularizer=kernel_regularizer)
            
    def call(self, x, training=False, return_features=False): 
                # always return list of features no matter return_features=True or False 
        # Encoder
        d1 = self.down1(x, training=training)      # (batch, 65, 14, C_out)
        d2 = self.down2(d1, training=training)     # (batch, 32, 14, C_out)
        d3 = self.down3(d2, training=training)     # (batch,  15, 14, C_out)
        d4 = self.down4(d3, training=training)     # (batch,  7, 14, C_out)
        # Decoder with skip connections
        u1 = self.up1(d4, d3, training=training)   # (batch,  15, 14, C_out)
        u2 = self.up2(u1, d2, training=training)   # (batch, 32, 14, C_out)
        u3 = self.up3(u2, d1, training=training)   # (batch, 65, 14, C_out)
        u4 = self.last(u3)  # (batch, 132, 14, C_out)
        
        if u4.shape[2] > 14:
            u4 = u4[:, :, 1:15, :]
            
        # Return multiple feature layers for JMMD
        features = []
        layer_map = {
            'd1': d1,
            'd2': d2, 
            'd3': d3,
            'd4': d4,
            'u1': u1,
            'u2': u2,
            'u3': u3
        }
        
        for layer_name in self.extract_layers:
            if layer_name in layer_map:
                layer_tensor = layer_map[layer_name]
                features.append(tf.reshape(layer_tensor, [tf.shape(layer_tensor)[0], -1]))
                    # features is already flattened
        return u4, features

class PatchGANDiscriminator(tf.keras.Model):
    """
    PatchGAN Discriminator for Pix2Pix GAN.
    Input: (batch, H, W, C)
    Output: (batch, H_out, W_out, 1) patch-level real/fake probabilities
    """
    def __init__(self, filters=[32, 64, 128, 256], n_subc=132, disc_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(disc_l2) if disc_l2 is not None else None
        
        self.conv1 = tf.keras.layers.Conv2D(filters[0], kernel_size=(4,3), strides=(2,1), padding='valid',
                                            kernel_regularizer=kernel_regularizer)
        self.conv2 = tf.keras.layers.Conv2D(filters[1], kernel_size=(3,3), strides=(2,1), padding='valid',
                                            kernel_regularizer=kernel_regularizer)
        self.norm2 = InstanceNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters[2], kernel_size=(4,3), strides=(2,1), padding='valid',
                                            kernel_regularizer=kernel_regularizer)
        self.norm3 = InstanceNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters[3], kernel_size=(3,3), strides=(2,1), padding='valid',
                                            kernel_regularizer=kernel_regularizer)
        self.norm4 = InstanceNormalization()
        self.last = tf.keras.layers.Conv2D(1, kernel_size=(3,3), strides=(2,1), padding='valid',
                                            kernel_regularizer=kernel_regularizer)  # Output: patch map

    def call(self, x, training=False):
        x = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)  # (batch, 395, 12, C_out)
        x = tf.nn.leaky_relu(self.norm2(self.conv2(x), training=training), alpha=0.2)  # (batch, 196, 10, C_out)
        x = tf.nn.leaky_relu(self.norm3(self.conv3(x), training=training), alpha=0.2)  # (batch, 97, 8, C_out)
        x = tf.nn.leaky_relu(self.norm4(self.conv4(x), training=training), alpha=0.2)  # (batch, 48, 6, C_out)
        return self.last(x)  # (batch, 23, 3, 1) - patch-level real/fake probabilities

class GAN_Output:
    """Dataclass to hold the output of the GAN model."""
    def __init__(self, gen_out, disc_out, extracted_features):
        self.gen_out = gen_out  # Generator output
        self.disc_out = disc_out  # Discriminator output
        self.extracted_features = extracted_features  # Extracted features
        
class GAN(tf.keras.Model):
    def __init__(self, n_subc=132, generator=Pix2PixGenerator, discriminator=PatchGANDiscriminator, gen_l2=None, disc_l2=None, extract_layers=['d2', 'd3','d4']):
        super().__init__()
        self.generator = generator(n_subc=n_subc, gen_l2=gen_l2, extract_layers=extract_layers)
        self.discriminator = discriminator(n_subc=n_subc, disc_l2=disc_l2)

    def call(self, inputs, training=False):
        # Optionally implement a forward pass if needed
        x = inputs
        gen_out, features = self.generator(x, training=training)
        disc_out = self.discriminator(gen_out, training=training)
        return GAN_Output(
            gen_out=gen_out,
            disc_out=disc_out,
            extracted_features=features
        )

class WeightScheduler:
    def __init__(self, strategy='domain_first_smooth'):
        self.strategy = strategy
    
    def get_weights_domain_first_smooth(self, epoch, n_epochs):
        """
        Domain-first with smooth sigmoid transitions (combines both approaches)
        """
        import math
        progress = epoch / n_epochs
        
        # Domain weight: High early, smooth decay
        domain_factor = 1.0 / (1 + math.exp(15 * (progress - 0.4)))  # Smooth decline
        domain_weight = 1.5 + 3.0 * domain_factor  # Range: 4.5 → 1.5
        
        # Estimation weight: Low early, smooth rise
        est_factor = 1.0 / (1 + math.exp(-12 * (progress - 0.5)))  # Smooth rise
        est_weight = 0.1 + 0.5 * est_factor  # Range: 0.1 → 0.6
        
        # Adversarial: Moderate throughout
        adv_weight = 0.03 + 0.05 * (1 - domain_factor)  # 0.03 → 0.08
        
        return {
            'adv_weight': adv_weight,
            'est_weight': est_weight,
            'domain_weight': domain_weight,
            'temporal_weight': 0.02,
            'frequency_weight': 0.1,
        }
    
    def get_weights_co_evolution(self, epoch, n_epochs):
        """
        Agent's co-evolution approach (for comparison)
        """
        import math
        progress = epoch / n_epochs
        
        # Generation: Quick rise then stable
        est_max = 0.6
        est_weight = est_max / (1 + math.exp(-10 * (progress - 0.1)))
        
        # Domain: Slower rise, peaks mid-training
        domain_max = 3.0
        domain_weight = domain_max / (1 + math.exp(-10 * (progress - 0.3)))
        
        # Decay domain weight after peak
        if progress > 0.7:
            domain_weight *= math.exp(-5 * (progress - 0.7))
        
        return {
            'adv_weight': 0.03,
            'est_weight': est_weight,
            'jmmd_weight': domain_weight,
            'temporal_weight': 0.02,
            'frequency_weight': 0.1,
        }

class JMMDLoss(keras.layers.Layer):
    """
    Joint Maximum Mean Discrepancy Loss in TensorFlow
    Computes JMMD across multiple feature layers
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(JMMDLoss, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        Compute Gaussian kernel matrix
        """
        n_samples = tf.shape(source)[0] + tf.shape(target)[0]
        total = tf.concat([source, target], axis=0)
        
        # Compute pairwise distances
        total_expanded_1 = tf.expand_dims(total, 1)  # [n, 1, d]
        total_expanded_2 = tf.expand_dims(total, 0)  # [1, n, d]
        
        L2_distance = tf.reduce_sum(tf.square(total_expanded_1 - total_expanded_2), axis=2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = tf.reduce_sum(L2_distance) / tf.cast(n_samples ** 2 - n_samples, tf.float32)
        
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return tf.reduce_sum(kernel_val, axis=0)
    
    def mmd(self, source, target):
        """
        Compute MMD between source and target
        """
        batch_size = tf.shape(source)[0]
        kernels = self.gaussian_kernel(source, target, 
                                    kernel_mul=self.kernel_mul, 
                                    kernel_num=self.kernel_num, 
                                    fix_sigma=self.fix_sigma)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = tf.reduce_mean(XX + YY - XY - YX)
        return loss
    
    def call(self, source_list, target_list):
        """
        Compute JMMD across multiple layers
        source_list: list of source features from different layers
        target_list: list of target features from different layers
        """
        jmmd_loss = 0.0
        for source_feat, target_feat in zip(source_list, target_list):
            # Flatten features if needed
            if len(source_feat.shape) > 2:
                source_feat = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
                target_feat = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
            
            jmmd_loss += self.mmd(source_feat, target_feat)
        
        return jmmd_loss / len(source_list)


# JMMD Loss with Feature Normalization
class JMMDLossNormalized(keras.layers.Layer):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, 
                 normalize_features=True, layer_weights=None, **kwargs):
        super(JMMDLossNormalized, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.normalize_features = normalize_features
        self.layer_weights = layer_weights  # Optional: weight different layers differently
        
    def normalize_feature_layer(self, features):
        """
        Normalize features for better JMMD performance
        """
        # L2 normalization (unit length vectors)
        features_l2 = tf.nn.l2_normalize(features, axis=-1)
        
        # Standardization (zero mean, unit variance)
        mean = tf.reduce_mean(features_l2, axis=0, keepdims=True)
        std = tf.math.reduce_std(features_l2, axis=0, keepdims=True) + 1e-8
        features_norm = (features_l2 - mean) / std
        
        return features_norm
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        Compute Gaussian kernel matrix (same as original but with normalized features)
        """
        n_samples = int(source.shape[0]) + int(target.shape[0])
        total = tf.concat([source, target], axis=0)
        
        total_expanded_1 = tf.expand_dims(total, axis=1)
        total_expanded_2 = tf.expand_dims(total, axis=0)
        
        L2_distance = tf.reduce_sum(tf.square(total_expanded_1 - total_expanded_2), axis=2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = tf.reduce_sum(L2_distance) / tf.cast(n_samples ** 2 - n_samples, tf.float32)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        
        return sum(kernel_val)
    
    def mmd(self, source, target):
        """
        Compute Maximum Mean Discrepancy (same as original)
        """
        batch_size = int(source.shape[0])
        kernels = self.gaussian_kernel(source, target,
                                      kernel_mul=self.kernel_mul, 
                                      kernel_num=self.kernel_num, 
                                      fix_sigma=self.fix_sigma)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = tf.reduce_mean(XX + YY - XY - YX)
        return loss
    
    def call(self, source_list, target_list):
        """
        Compute JMMD across multiple layers with normalization and optional layer weighting
        """
        jmmd_loss = 0.0
        total_weight = 0.0
        
        for i, (source_feat, target_feat) in enumerate(zip(source_list, target_list)):
            # Flatten features if needed
            if len(source_feat.shape) > 2:
                source_feat = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
                target_feat = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
            
            # Apply normalization for better JMMD performance
            if self.normalize_features:
                source_feat = self.normalize_feature_layer(source_feat)
                target_feat = self.normalize_feature_layer(target_feat)
            
            # Compute MMD for this layer
            layer_mmd = self.mmd(source_feat, target_feat)
            
            # Apply layer weighting (earlier layers get higher weight by default)
            if self.layer_weights is not None:
                layer_weight = self.layer_weights[i] if i < len(self.layer_weights) else 1.0
            else:
                layer_weight = 1.0 / (i + 1)  # Earlier layers: higher weight (less specialized)
            
            jmmd_loss += layer_weight * layer_mmd
            total_weight += layer_weight
        
        return jmmd_loss / total_weight


def compute_total_smoothness_loss(x, temporal_weight=0.02, frequency_weight=0.1):
    """
        x: Generated channel matrix of shape (batch, freq, time, channels)
    """
    # Temporal smoothness (along time axis=2)
    temporal_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
    temporal_loss = tf.reduce_mean(tf.square(temporal_diff))
    
    # Frequency smoothness (along frequency axis=1) 
    frequency_diff = x[:, 1:, :, :] - x[:, :-1, :, :]
    frequency_loss = tf.reduce_mean(tf.square(frequency_diff))
    
    loss_smoothness = temporal_weight * temporal_loss + frequency_weight * frequency_loss
    return loss_smoothness

def save_compressed_batch_to_h5(batch_pca, h5_file, dataset, is_source=True):
    """
    Helper function to save PCA-compressed batch to HDF5 file
    """
    if dataset is None:
        # First batch - create dataset
        dataset = h5_file.create_dataset(
            'features',
            data=batch_pca,
            maxshape=(None,) + batch_pca.shape[1:],
            chunks=True
        )
        # Initialize domain labels
        domain_labels = np.ones(batch_pca.shape[0]) if is_source else np.zeros(batch_pca.shape[0])
        h5_file.create_dataset('domain_labels', data=domain_labels, maxshape=(None,), chunks=True)
    else:
        # Append to existing dataset
        old_size = dataset.shape[0]
        dataset.resize(old_size + batch_pca.shape[0], axis=0)
        dataset[-batch_pca.shape[0]:] = batch_pca
        
        # Append domain labels
        domain_labels = np.ones(batch_pca.shape[0]) if is_source else np.zeros(batch_pca.shape[0])
        domain_dataset = h5_file['domain_labels']
        domain_dataset.resize(old_size + batch_pca.shape[0], axis=0)
        domain_dataset[-batch_pca.shape[0]:] = domain_labels
    
    return dataset

def train_step_wgan_gp_jmmd(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                            save_features = False, nsymb=14, weights=None, linear_interp=False):
    """
    Modified WGAN-GP training step using JMMD instead of domain discriminator
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - no domain loss needed
        optimizers: tuple of (gen_optimizer, disc_optimizer) - no domain optimizer needed
        domain_weight: weight for JMMD loss (replaces domain_weight)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]  # Only need first two loss functions
    gen_optimizer, disc_optimizer = optimizers[:2]  # No domain optimizer needed
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    jmmd_weight = weights.get('domain_weight')
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_jmmd = 0.0
    N_train = 0
    
    if save_features==True and (jmmd_weight != 0):
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)  # Remove if exists to start fresh
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None  # Will be created after first batch

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)  # Remove if exists to start fresh   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None  # Will be created after first batch 
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # --- Get data ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess (source)
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === 1. Train Discriminator (WGAN-GP) ===
        # Only considering source domain for discriminator training
        with tf.GradientTape() as tape_d:
            x_fake_src, _ = model.generator(x_scaled_src, training=True)
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            
            # WGAN-GP gradient penalty
            gp = gradient_penalty(model.discriminator, y_scaled_src, x_fake_src, batch_size=x_scaled_src.shape[0])
            lambda_gp = 10.0  # typical gradient penalty weight

            # WGAN-GP discriminator loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            # Add L2 regularization loss from discriminator
            if model.discriminator.losses:
                d_loss += tf.add_n(model.discriminator.losses)
                
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]

        # === 2. Train Generator with JMMD ===
        with tf.GradientTape() as tape_g:
            # Generate from source domain with features
            x_fake_src, features_src = model.generator(x_scaled_src, training=True)
            d_fake_src = model.discriminator(x_fake_src, training=False)
            
            # Generate from target domain with features
            x_fake_tgt, features_tgt = model.generator(x_scaled_tgt, training=True)
            
            # Generator losses
            g_adv_loss = -tf.reduce_mean(d_fake_src)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)  # Estimation loss (source)
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)  # Estimation loss (target, for monitoring)
            
            # JMMD loss between source and target features
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            
            # ADD TEMPORAL SMOOTHNESS LOSS
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss_src = compute_total_smoothness_loss(x_fake_src, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss_tgt = compute_total_smoothness_loss(x_fake_tgt, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            else:
                smoothness_loss = 0.0
        
            # Total generator loss
            g_loss = (est_weight * g_est_loss + 
                    adv_weight * g_adv_loss + 
                    jmmd_weight * jmmd_loss + 
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        
        # === 3. Save features (after the bottleneck layer) if required (to calcu PAD) ===
        if save_features and (jmmd_weight != 0):
            # save features in a temporary file instead of stacking them up, to avoid memory exploding
            features_np_source = features_src[-1].numpy()  # Convert to numpy if it's a tensor
            # print('Feature shape: ', features_np_source.shape)
            if features_dataset_source is None:
                # Create dataset with unlimited first dimension
                features_dataset_source = features_h5_source.create_dataset(
                    'features',
                    data=features_np_source,
                    maxshape=(None,) + features_np_source.shape[1:],
                    chunks=True
                )
            else:
                # Resize and append
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_source.shape[0], axis=0)
                features_dataset_source[-features_np_source.shape[0]:] = features_np_source
                
            features_np_target = features_tgt[-1].numpy()
            if features_dataset_target is None:
                # Create dataset with unlimited first dimension
                features_dataset_target = features_h5_target.create_dataset(
                    'features',
                    data=features_np_target,
                    maxshape=(None,) + features_np_target.shape[1:],
                    chunks=True
                )
            else:
                # Resize and append
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
                features_dataset_target[-features_np_target.shape[0]:] = features_np_target
                
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0]
    # end batch loop
    if save_features and (jmmd_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
    
    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train
    
    # Return compatible output structure (replacing domain loss with JMMD)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd,  # Replace domain loss with JMMD
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=features_src[-1] if features_src else None,  # Return bottleneck features
        film_features_source=features_src[-1] if features_src else None,
        avg_epoc_loss_d=avg_loss_d
    )

def train_step_wgan_gp_jmmd_new(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                            save_features = False, nsymb=14, weights=None, linear_interp=False,
                            pca_components=2000):
    """
    Modified WGAN-GP training step using JMMD instead of domain discriminator
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - no domain loss needed
        optimizers: tuple of (gen_optimizer, disc_optimizer) - no domain optimizer needed
        jmmd_weight: weight for JMMD loss (replaces domain_weight)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]  # Only need first two loss functions
    gen_optimizer, disc_optimizer = optimizers[:2]  # No domain optimizer needed
    
    batch_size = loader_H_input_train_src.batch_size
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    jmmd_weight = weights.get('domain_weight')
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_jmmd = 0.0
    N_train = 0
    
    if save_features==True and (jmmd_weight != 0):
        # Initialize incremental PCA variables
        pca_src = IncrementalPCA(n_components=pca_components, batch_size=64)
        pca_tgt = IncrementalPCA(n_components=pca_components, batch_size=64)
        pca_fitted = False
        fitting_batch_count = 0
        max_fitting_batches = 3 #128//batch_size  # Use first batches to fit PCA
        
        # Storage for fitting batches (temporary)
        fitting_batches_src = []
        fitting_batches_tgt = []
        
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)  # Remove if exists to start fresh
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None  # Will be created after first batch

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)  # Remove if exists to start fresh   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None  # Will be created after first batch 
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # --- Get data ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess (source)
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === 1. Train Discriminator (WGAN-GP) ===
        # Only considering source domain for discriminator training
        with tf.GradientTape() as tape_d:
            x_fake_src, _ = model.generator(x_scaled_src, training=True)
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            
            # WGAN-GP gradient penalty
            gp = gradient_penalty(model.discriminator, y_scaled_src, x_fake_src, batch_size=x_scaled_src.shape[0])
            lambda_gp = 10.0  # typical gradient penalty weight

            # WGAN-GP discriminator loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            # Add L2 regularization loss from discriminator
            if model.discriminator.losses:
                d_loss += tf.add_n(model.discriminator.losses)
                
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]

        # === 2. Train Generator with JMMD ===
        with tf.GradientTape() as tape_g:
            # Generate from source domain with features
            x_fake_src, features_src = model.generator(x_scaled_src, training=True)
            d_fake_src = model.discriminator(x_fake_src, training=False)
            
            # Generate from target domain with features
            x_fake_tgt, features_tgt = model.generator(x_scaled_tgt, training=True)
            
            # Generator losses
            g_adv_loss = -tf.reduce_mean(d_fake_src)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)  # Estimation loss (source)
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)  # Estimation loss (target, for monitoring)
            
            # JMMD loss between source and target features
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            
            # ADD TEMPORAL SMOOTHNESS LOSS
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss_src = compute_total_smoothness_loss(x_fake_src, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss_tgt = compute_total_smoothness_loss(x_fake_tgt, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            else:
                smoothness_loss = 0.0
        
            # Total generator loss
            g_loss = (est_weight * g_est_loss + 
                    adv_weight * g_adv_loss + 
                    jmmd_weight * jmmd_loss + 
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        
        # === 3. Save features (after the bottleneck layer) if required (to calcu PAD) ===
        if save_features and (jmmd_weight != 0):
            # save features in a temporary file instead of stacking them up, to avoid memory exploding
            features_np_source = features_src[-1].numpy()  # Convert to numpy if it's a tensor
            # print('Feature shape: ', features_np_source.shape)
            # Check if already flattened, if not, flatten
            if len(features_np_source.shape) > 2:
                features_np_source = features_np_source.reshape(features_np_source.shape[0], -1)            
            # if features_dataset_source is None:
            #     # Create dataset with unlimited first dimension
            #     features_dataset_source = features_h5_source.create_dataset(
            #         'features',
            #         data=features_np_source,
            #         maxshape=(None,) + features_np_source.shape[1:],
            #         chunks=True
            #     )
            # else:
            #     # Resize and append
            #     features_dataset_source.resize(features_dataset_source.shape[0] + features_np_source.shape[0], axis=0)
            #     features_dataset_source[-features_np_source.shape[0]:] = features_np_source
                
            features_np_target = features_tgt[-1].numpy()
            if len(features_np_target.shape) > 2:
                features_np_target = features_np_target.reshape(features_np_target.shape[0], -1)            
            # if features_dataset_target is None:
            #     # Create dataset with unlimited first dimension
            #     features_dataset_target = features_h5_target.create_dataset(
            #         'features',
            #         data=features_np_target,
            #         maxshape=(None,) + features_np_target.shape[1:],
            #         chunks=True
            #     )
            # else:
            #     # Resize and append
            #     features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
            #     features_dataset_target[-features_np_target.shape[0]:] = features_np_target
                
            print(f'Batch {batch_idx+1}: Original feature shape - Source: {features_np_source.shape}, Target: {features_np_target.shape}')
            
            if not pca_fitted and fitting_batch_count < max_fitting_batches:
                # Phase 1: Collect batches for PCA fitting
                fitting_batches_src.append(features_np_source)
                fitting_batches_tgt.append(features_np_target)
                fitting_batch_count += 1
                print(f"Collecting batch {fitting_batch_count}/{max_fitting_batches} for PCA fitting...")
                
                if fitting_batch_count == max_fitting_batches:
                    # Fit incremental PCA on collected batches
                    print("Fitting Incremental PCA on collected batches...")
                    fitting_data_src = np.vstack(fitting_batches_src)  # (.., 100352) 
                    fitting_data_tgt = np.vstack(fitting_batches_tgt) 
                    pca_src.partial_fit(fitting_data_src)  
                    pca_tgt.partial_fit(fitting_data_tgt) 
                    
                    # Transform and save the fitting batches
                    print("Transforming and saving fitting batches...")
                    for batch in fitting_batches_src:
                        batch_pca = pca_src.transform(batch)
                        
                        if features_dataset_source is None:
                            features_dataset_source = features_h5_source.create_dataset(
                                'features', data=batch_pca,
                                maxshape=(None,) + batch_pca.shape[1:], chunks=True
                            )
                        else:
                            features_dataset_source.resize(features_dataset_source.shape[0] + batch_pca.shape[0], axis=0)
                            features_dataset_source[-batch_pca.shape[0]:] = batch_pca
                    
                                        
                    for batch in fitting_batches_tgt:
                        batch_pca = pca_tgt.transform(batch)
                        
                        if features_dataset_target is None:
                            features_dataset_target = features_h5_target.create_dataset(
                                'features', data=batch_pca,
                                maxshape=(None,) + batch_pca.shape[1:], chunks=True
                            )
                        else:
                            features_dataset_target.resize(features_dataset_target.shape[0] + batch_pca.shape[0], axis=0)
                            features_dataset_target[-batch_pca.shape[0]:] = batch_pca
                    
                    # Clear fitting data and mark as fitted
                    del fitting_batches_src, fitting_batches_tgt
                    pca_fitted = True
                    
                    explained_var_src = np.sum(pca_src.explained_variance_ratio_)
                    explained_var_tgt = np.sum(pca_tgt.explained_variance_ratio_)
                    print("Phase 1 PCA fitting completed.")
                    print(f"PCA fitted! Explained variance - Source: {explained_var_src:.4f}, Target: {explained_var_tgt:.4f}")
                    print(f"Compression: {features_np_source.shape[1]} -> {pca_components} dimensions")
            
            elif pca_fitted:
                print("Begin Phase 2 PCA")
                # Phase 2: Transform current batch and save immediately
                features_src_pca = pca_src.transform(features_np_source)
                features_tgt_pca = pca_tgt.transform(features_np_target)
                
                # Update PCA incrementally with current batch
                pca_src.partial_fit(features_np_source)
                pca_tgt.partial_fit(features_np_target)
                
                # Save compressed features
                features_dataset_source.resize(features_dataset_source.shape[0] + features_src_pca.shape[0], axis=0)
                features_dataset_source[-features_src_pca.shape[0]:] = features_src_pca
                
                features_dataset_target.resize(features_dataset_target.shape[0] + features_tgt_pca.shape[0], axis=0)
                features_dataset_target[-features_tgt_pca.shape[0]:] = features_tgt_pca
    
                
                print(f"Batch {batch_idx+1}: Transformed and saved - "
                    f"Source: {features_np_source.shape} -> {features_src_pca.shape}, "
                    f"Target: {features_np_target.shape} -> {features_tgt_pca.shape}")
        
            
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0]
    # end batch loop
    if save_features and (jmmd_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
    
    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train
    
    # Return compatible output structure (replacing domain loss with JMMD)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd,  # Replace domain loss with JMMD
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=features_src[-1] if features_src else None,  # Return bottleneck features
        film_features_source=features_src[-1] if features_src else None,
        avg_epoc_loss_d=avg_loss_d
    )


# Training function with normalized JMMD
def train_step_wgan_gp_jmmd_normalized(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                                    save_features=False, nsymb=14, weights=None, linear_interp=False,
                                    normalize_features=True, layer_weights=None, debug_features=False):
    """
    Enhanced WGAN-GP training step with normalized JMMD for better domain adaptation
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss)
        optimizers: tuple of (gen_optimizer, disc_optimizer)
        normalize_features (bool): Enable feature normalization in JMMD
        layer_weights (list): Optional weights for different feature layers [w1, w2, w3]
        debug_features (bool): Print feature statistics for debugging
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]  # Only need first two loss functions
    gen_optimizer, disc_optimizer = optimizers[:2]  # No domain optimizer needed
    
    # Extract weights with better defaults for normalized JMMD
    adv_weight = weights.get('adv_weight', 0.02)
    temporal_weight = weights.get('temporal_weight', 0.02)
    frequency_weight = weights.get('frequency_weight', 0.1)
    est_weight = weights.get('est_weight', 0.4)  # Lower default for better adaptation
    jmmd_weight = weights.get('domain_weight')  # Higher default for stronger adaptation
    
    # Initialize NORMALIZED JMMD loss
    jmmd_loss_fn = JMMDLossNormalized(normalize_features=normalize_features, 
                                    layer_weights=layer_weights)
    
    # Training loop metrics
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0 
    epoc_loss_est = 0.0
    epoc_loss_jmmd = 0.0
    epoc_loss_est_tgt = 0.0
    N_train = 0
    
    if save_features==True and (jmmd_weight != 0):
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)  # Remove if exists to start fresh
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None  # Will be created after first batch

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)  # Remove if exists to start fresh   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None  # Will be created after first batch 
    
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # --- Get data ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess (source)
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        batch_size = tf.shape(x_scaled_src)[0]
        
        # === 1. Train Discriminator (WGAN-GP) ===
        with tf.GradientTape() as tape_d:
            x_fake_src, _ = model.generator(x_scaled_src, training=True)
            
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            
            # WGAN-GP gradient penalty
            gp = gradient_penalty(model.discriminator, y_scaled_src, x_fake_src, batch_size=batch_size)
            lambda_gp = 10.0
            
            # WGAN-GP discriminator loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            # Add L2 regularization loss from discriminator
            if model.discriminator.losses:
                d_loss += tf.add_n(model.discriminator.losses)
        
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]
        
        # === 2. Train Generator with Normalized JMMD ===
        with tf.GradientTape() as tape_g:
            # Generate with feature extraction for JMMD
            x_fake_src, features_src = model.generator(x_scaled_src, training=True)
            d_fake_src = model.discriminator(x_fake_src, training=True)
            
            # Generate from target domain with features
            x_fake_tgt, features_tgt = model.generator(x_scaled_tgt, training=True)
            
            # Generator losses
            g_adv_loss = -tf.reduce_mean(d_fake_src)
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt) 
            
            # NORMALIZED JMMD loss between source and target features
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            
            # Smoothness regularization (using existing function)
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss_src = compute_total_smoothness_loss(x_fake_src, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss_tgt = compute_total_smoothness_loss(x_fake_tgt, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            else:
                smoothness_loss = 0.0
                
            # Total generator loss
            g_loss = (est_weight * g_est_loss + 
                    adv_weight * g_adv_loss + 
                    jmmd_weight * jmmd_loss + 
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)

        # === 3. Save features (after the bottleneck layer) if required (to calcu PAD) ===
        if save_features and (jmmd_weight != 0):
            # save features in a temporary file instead of stacking them up, to avoid memory exploding
            features_np_source = features_src[-1].numpy()  # Convert to numpy if it's a tensor
            # print('Feature shape: ', features_np_source.shape)
            if features_dataset_source is None:
                # Create dataset with unlimited first dimension
                features_dataset_source = features_h5_source.create_dataset(
                    'features',
                    data=features_np_source,
                    maxshape=(None,) + features_np_source.shape[1:],
                    chunks=True
                )
            else:
                # Resize and append
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_source.shape[0], axis=0)
                features_dataset_source[-features_np_source.shape[0]:] = features_np_source
                
            features_np_target = features_tgt[-1].numpy()
            if features_dataset_target is None:
                # Create dataset with unlimited first dimension
                features_dataset_target = features_h5_target.create_dataset(
                    'features',
                    data=features_np_target,
                    maxshape=(None,) + features_np_target.shape[1:],
                    chunks=True
                )
            else:
                # Resize and append
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
                features_dataset_target[-features_np_target.shape[0]:] = features_np_target
                
        gradients_of_generator = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0]
        
            
    # end batch loop
    if save_features and (jmmd_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()

    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train
    
    # Return compatible output structure with normalized JMMD
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd,  # Normalized JMMD loss
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=features_src[-1] if features_src else None,
        film_features_source=features_src[-1] if features_src else None,
        avg_epoc_loss_d=avg_loss_d
    )


def train_step_wgan_gp_source_only(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                                  save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    WGAN-GP training step using only source domain (no domain adaptation)
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                          loader_H_input_train_tgt, loader_H_true_train_tgt) - tgt used only for testing
        loss_fn: tuple of loss functions (estimation_loss, bce_loss)
        optimizers: tuple of (gen_optimizer, disc_optimizer)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]
    gen_optimizer, disc_optimizer = optimizers[:2]
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0  # For monitoring target performance
    N_train = 0
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)  # Remove if exists to start fresh
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None  # Will be created after first batch

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)  # Remove if exists to start fresh   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None  # Will be created after first batch
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # --- Get SOURCE data for training ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        N_train += x_src.shape[0]

        # Preprocess source data
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # === 1. Train Discriminator (WGAN-GP) - Source only ===
        with tf.GradientTape() as tape_d:
            x_fake_src, _ = model.generator(x_scaled_src, training=True, return_features=False)
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            
            # WGAN-GP gradient penalty
            gp = gradient_penalty(model.discriminator, y_scaled_src, x_fake_src, batch_size=x_scaled_src.shape[0])
            lambda_gp = 10.0

            # WGAN-GP discriminator loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            # Add L2 regularization loss from discriminator
            if model.discriminator.losses:
                d_loss += tf.add_n(model.discriminator.losses)
                
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]

        # === 2. Train Generator - Source only ===
        with tf.GradientTape() as tape_g:
            # Generate from source domain
            x_fake_src, features_src = model.generator(x_scaled_src, training=True, return_features=True)
            d_fake_src = model.discriminator(x_fake_src, training=False)
            
            # Generator losses
            g_adv_loss = -tf.reduce_mean(d_fake_src)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)  # Estimation loss (source)
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss = compute_total_smoothness_loss(x_fake_src, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
        
            # Total generator loss (no domain adaptation)
            g_loss = est_weight * g_est_loss + adv_weight * g_adv_loss + smoothness_loss
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
                
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]

        # === 3. Optional: Monitor target performance (no training) ===
        if batch_idx < loader_H_true_train_tgt.total_batches:
            x_tgt = loader_H_input_train_tgt.next_batch()
            y_tgt = loader_H_true_train_tgt.next_batch()
            
            # Preprocess target data
            x_tgt = complx2real(x_tgt)
            y_tgt = complx2real(y_tgt)
            x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
            y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
            x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
            y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
            
            # Test on target (no gradients)
            x_fake_tgt, _ = model.generator(x_scaled_tgt, training=False, return_features=False)
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)
            epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]

    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train if epoc_loss_est_tgt > 0 else 0.0
    
    # Return compatible output structure (no domain adaptation)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,  # No domain adaptation
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=None,
        film_features_source=None,
        avg_epoc_loss_d=avg_loss_d
    )

def val_step_wgan_gp_source_only(model, loader_H, loss_fn, lower_range, nsymb=14, weights=None, linear_interp=False):
    """
    Validation step for source-only training.
    Validates on source domain, tests on target domain.
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0  # This is now testing on target
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0  # This is now testing on target
    epoc_gan_disc_loss = 0.0
    H_sample = []

    for idx in range(loader_H_true_val_source.total_batches):
        # --- Source domain validation ---
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        N_val_source += x_src.shape[0]

        # Preprocess source
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Source validation prediction
        preds_src, _ = model.generator(x_scaled_src, training=False, return_features=False)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        
        # Source NMSE
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # Discriminator loss (source only)
        d_real_src = model.discriminator(y_scaled_src, training=False)
        d_fake_src = model.discriminator(preds_src, training=False)
        gp_src = gradient_penalty(model.discriminator, y_scaled_src, preds_src, batch_size=x_scaled_src.shape[0])
        d_loss_src = tf.reduce_mean(d_fake_src) - tf.reduce_mean(d_real_src) + 10.0 * gp_src
        epoc_gan_disc_loss += d_loss_src.numpy() * x_src.shape[0]

    # --- Target domain testing (separate loop to handle different batch sizes) ---
    for idx in range(loader_H_true_val_target.total_batches):
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_target += x_tgt.shape[0]

        # Preprocess target
        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # Target testing prediction
        preds_tgt, _ = model.generator(x_scaled_tgt, training=False, return_features=False)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        
        # Target NMSE
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # Save samples from first batch
        if idx == 0:
            n_samples = min(3, x_tgt_real.shape[0])
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            H_est_sample_target = preds_tgt_descaled[:n_samples].copy()
            
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            
            H_sample = [H_true_sample_target, H_input_sample_target, H_est_sample_target, 
                       nmse_input_target, nmse_est_target]

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target  # This is testing performance
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target  # This is testing performance
    avg_gan_disc_loss = epoc_gan_disc_loss / N_val_source

    # Total loss (source validation only)
    avg_total_loss = est_weight * avg_loss_est_source + adv_weight * avg_gan_disc_loss

    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target,  # This is testing loss
        'avg_loss_est': avg_loss_est_source,  # Use source for validation
        'avg_gan_disc_loss': avg_gan_disc_loss,
        'avg_jmmd_loss': 0.0,  # No domain adaptation
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,  # This is testing NMSE
        'avg_nmse': avg_nmse_source,  # Use source for validation
        'avg_domain_acc_source': 0.5,
        'avg_domain_acc_target': 0.5,
        'avg_domain_acc': 0.5,
        'avg_smoothness_loss': 0.0
    }

    return H_sample, epoc_eval_return

def val_step_wgan_gp_jmmd(model, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                        linear_interp=False, return_H_gen=False):
    """
    Validation step for WGAN-GP model with JMMD. Returns H_sample and epoc_eval_return (summary metrics).
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (input_src, true_src, input_tgt, true_tgt) DataLoaders
        loss_fn: tuple of (estimation loss, binary cross-entropy loss) - no domain loss needed
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        adv_weight, est_weight, jmmd_weight: loss weights
        
    Returns:
        H_sample, epoc_eval_return
    """
    from sklearn.metrics import accuracy_score
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    jmmd_weight = weights.get('jmmd_weight', 0.5)
    
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]  # Only need first two loss functions
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_jmmd_loss = 0.0  # Replace domain loss with JMMD
    epoc_smoothness_loss = 0.0
    H_sample = []
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # --- Source domain ---
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        # --- Target domain ---
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocess (source)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === Source domain prediction ===
        preds_src, features_src = model.generator(x_scaled_src, training=False, return_features=True)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === Target domain prediction ===
        preds_tgt, features_tgt = model.generator(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # === WGAN Discriminator Scores (for monitoring only) ===
        # Only considering source domain
        d_real_src = model.discriminator(y_scaled_src, training=False)
        d_fake_src = model.discriminator(preds_src, training=False)
        gp_src = gradient_penalty(model.discriminator, y_scaled_src, preds_src, batch_size=x_scaled_src.shape[0])
        lambda_gp = 10.0  # typical gradient penalty weight
        
        # WGAN critic loss: mean(fake) - mean(real)
        d_loss_src = tf.reduce_mean(d_fake_src) - tf.reduce_mean(d_real_src) + lambda_gp * gp_src
        
        # only observe GAN disc loss on source dataset
        epoc_gan_disc_loss += d_loss_src.numpy() * x_src.shape[0]

        # === JMMD Loss (replaces Domain Discriminator) ===
        if jmmd_weight > 0:
            # Compute JMMD loss between source and target features
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # === ADD SMOOTHNESS LOSS COMPUTATION (INSERT HERE) ===
        if temporal_weight != 0 or frequency_weight != 0:
            # Convert back to tensors if needed for smoothness computation
            preds_src_tensor = tf.convert_to_tensor(preds_src) if not tf.is_tensor(preds_src) else preds_src
            preds_tgt_tensor = tf.convert_to_tensor(preds_tgt) if not tf.is_tensor(preds_tgt) else preds_tgt
            
            smoothness_loss_src = compute_total_smoothness_loss(
                preds_src_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
            )
            smoothness_loss_tgt = compute_total_smoothness_loss(
                preds_tgt_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
            )
            batch_smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            epoc_smoothness_loss += batch_smoothness_loss.numpy() * x_src.shape[0]

        # === Save H samples for visualization at first batch ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            #
            if hasattr(preds_src_descaled, 'numpy'):
                H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample = preds_src_descaled[:n_samples].copy()
            #
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            # Target
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            #
            if hasattr(preds_tgt_descaled, 'numpy'):
                H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample_target = preds_tgt_descaled[:n_samples].copy()
            #
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            H_sample = [H_true_sample, H_input_sample, H_est_sample, nmse_input_source, nmse_est_source,
                        H_true_sample_target, H_input_sample_target, H_est_sample_target, nmse_input_target, nmse_est_target]
        if return_H_gen:
            # Convert to numpy if needed and append to lists
            H_gen_src_batch = preds_src_descaled.numpy().copy() if hasattr(preds_src_descaled, 'numpy') else preds_src_descaled.copy()
            H_gen_tgt_batch = preds_tgt_descaled.numpy().copy() if hasattr(preds_tgt_descaled, 'numpy') else preds_tgt_descaled.copy()
            
            all_H_gen_src.append(H_gen_src_batch)
            all_H_gen_tgt.append(H_gen_tgt_batch)
    if return_H_gen:
        # Concatenate all batches along the batch dimension (axis=0)
        H_gen_src_all = np.concatenate(all_H_gen_src, axis=0)
        H_gen_tgt_all = np.concatenate(all_H_gen_tgt, axis=0)
        H_gen = {
            'H_gen_src': H_gen_src_all,
            'H_gen_tgt': H_gen_tgt_all
        }

    # Calculate averages
    N_val = N_val_source + N_val_target
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    # only observe GAN disc loss on source dataset
    avg_gan_disc_loss = epoc_gan_disc_loss / N_val_source 
    
    # JMMD loss average (replaces domain discriminator loss)
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    # smoothness loss average
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    
    # For compatibility with existing code, we'll set domain accuracy to 0.5 (random)
    # since JMMD doesn't have classification accuracy
    avg_domain_acc_source = 0.5  # Neutral value for JMMD (no classification)
    avg_domain_acc_target = 0.5  # Neutral value for JMMD (no classification)
    avg_domain_acc = 0.5         # Neutral value for JMMD (no classification)

    # Weighted total loss (for comparison with training)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss \
                     + jmmd_weight * avg_jmmd_loss + avg_smoothness_loss

    # Compose epoc_eval_return - Replace domain discriminator loss with JMMD loss
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': avg_gan_disc_loss,
        'avg_jmmd_loss': avg_jmmd_loss,
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return

def val_step_wgan_gp_jmmd_normalized(model, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                        linear_interp=False, return_H_gen=False):
    """
    Validation step for WGAN-GP model with JMMD. Returns H_sample and epoc_eval_return (summary metrics).
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (input_src, true_src, input_tgt, true_tgt) DataLoaders
        loss_fn: tuple of (estimation loss, binary cross-entropy loss) - no domain loss needed
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        adv_weight, est_weight, jmmd_weight: loss weights
        
    Returns:
        H_sample, epoc_eval_return
    """
    from sklearn.metrics import accuracy_score
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    jmmd_weight = weights.get('domain_weight')
    
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]  # Only need first two loss functions
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLossNormalized()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_jmmd_loss = 0.0  # Replace domain loss with JMMD
    epoc_smoothness_loss = 0.0
    H_sample = []
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # --- Source domain ---
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        # --- Target domain ---
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocess (source)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === Source domain prediction ===
        preds_src, features_src = model.generator(x_scaled_src, training=False, return_features=True)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === Target domain prediction ===
        preds_tgt, features_tgt = model.generator(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # === WGAN Discriminator Scores (for monitoring only) ===
        # Only considering source domain
        d_real_src = model.discriminator(y_scaled_src, training=False)
        d_fake_src = model.discriminator(preds_src, training=False)
        gp_src = gradient_penalty(model.discriminator, y_scaled_src, preds_src, batch_size=x_scaled_src.shape[0])
        lambda_gp = 10.0  # typical gradient penalty weight
        
        # WGAN critic loss: mean(fake) - mean(real)
        d_loss_src = tf.reduce_mean(d_fake_src) - tf.reduce_mean(d_real_src) + lambda_gp * gp_src
        
        # only observe GAN disc loss on source dataset
        epoc_gan_disc_loss += d_loss_src.numpy() * x_src.shape[0]

        # === JMMD Loss (replaces Domain Discriminator) ===
        if jmmd_weight > 0:
            # Compute JMMD loss between source and target features
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # === ADD SMOOTHNESS LOSS COMPUTATION (INSERT HERE) ===
        if temporal_weight != 0 or frequency_weight != 0:
            # Convert back to tensors if needed for smoothness computation
            preds_src_tensor = tf.convert_to_tensor(preds_src) if not tf.is_tensor(preds_src) else preds_src
            preds_tgt_tensor = tf.convert_to_tensor(preds_tgt) if not tf.is_tensor(preds_tgt) else preds_tgt
            
            smoothness_loss_src = compute_total_smoothness_loss(
                preds_src_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
            )
            smoothness_loss_tgt = compute_total_smoothness_loss(
                preds_tgt_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
            )
            batch_smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            epoc_smoothness_loss += batch_smoothness_loss.numpy() * x_src.shape[0]

        # === Save H samples for visualization at first batch ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            #
            if hasattr(preds_src_descaled, 'numpy'):
                H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample = preds_src_descaled[:n_samples].copy()
            #
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            # Target
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            #
            if hasattr(preds_tgt_descaled, 'numpy'):
                H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample_target = preds_tgt_descaled[:n_samples].copy()
            #
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            H_sample = [H_true_sample, H_input_sample, H_est_sample, nmse_input_source, nmse_est_source,
                        H_true_sample_target, H_input_sample_target, H_est_sample_target, nmse_input_target, nmse_est_target]
        if return_H_gen:
            # Convert to numpy if needed and append to lists
            H_gen_src_batch = preds_src_descaled.numpy().copy() if hasattr(preds_src_descaled, 'numpy') else preds_src_descaled.copy()
            H_gen_tgt_batch = preds_tgt_descaled.numpy().copy() if hasattr(preds_tgt_descaled, 'numpy') else preds_tgt_descaled.copy()
            
            all_H_gen_src.append(H_gen_src_batch)
            all_H_gen_tgt.append(H_gen_tgt_batch)
    if return_H_gen:
        # Concatenate all batches along the batch dimension (axis=0)
        H_gen_src_all = np.concatenate(all_H_gen_src, axis=0)
        H_gen_tgt_all = np.concatenate(all_H_gen_tgt, axis=0)
        H_gen = {
            'H_gen_src': H_gen_src_all,
            'H_gen_tgt': H_gen_tgt_all
        }

    # Calculate averages
    N_val = N_val_source + N_val_target
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    # only observe GAN disc loss on source dataset
    avg_gan_disc_loss = epoc_gan_disc_loss / N_val_source 
    
    # JMMD loss average (replaces domain discriminator loss)
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    # smoothness loss average
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    
    # For compatibility with existing code, we'll set domain accuracy to 0.5 (random)
    # since JMMD doesn't have classification accuracy
    avg_domain_acc_source = 0.5  # Neutral value for JMMD (no classification)
    avg_domain_acc_target = 0.5  # Neutral value for JMMD (no classification)
    avg_domain_acc = 0.5         # Neutral value for JMMD (no classification)

    # Weighted total loss (for comparison with training)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss \
                     + jmmd_weight * avg_jmmd_loss + avg_smoothness_loss

    # Compose epoc_eval_return - Replace domain discriminator loss with JMMD loss
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': avg_gan_disc_loss,
        'avg_jmmd_loss': avg_jmmd_loss,
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return


def train_step_wgan_gp_source_only(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                                save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    WGAN-GP training step using only source domain (no domain adaptation)
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt) - tgt used only for testing
        loss_fn: tuple of loss functions (estimation_loss, bce_loss)
        optimizers: tuple of (gen_optimizer, disc_optimizer)
        save_features: bool, whether to save features for PAD computation
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]
    gen_optimizer, disc_optimizer = optimizers[:2]
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight')
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0  # For monitoring target performance
    N_train = 0
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)  # Remove if exists to start fresh
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None  # Will be created after first batch

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)  # Remove if exists to start fresh   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None  # Will be created after first batch
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # --- Get SOURCE data for training ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        N_train += x_src.shape[0]

        # Preprocess source data
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # === 1. Train Discriminator (WGAN-GP) - Source only ===
        with tf.GradientTape() as tape_d:
            x_fake_src, _ = model.generator(x_scaled_src, training=True, return_features=False)
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            
            # WGAN-GP gradient penalty
            gp = gradient_penalty(model.discriminator, y_scaled_src, x_fake_src, batch_size=x_scaled_src.shape[0])
            lambda_gp = 10.0

            # WGAN-GP discriminator loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            # Add L2 regularization loss from discriminator
            if model.discriminator.losses:
                d_loss += tf.add_n(model.discriminator.losses)
        
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]

        # === 2. Train Generator - Source only ===
        with tf.GradientTape() as tape_g:
            # Generate from source domain
            x_fake_src, features_src = model.generator(x_scaled_src, training=True, return_features=False) 
            d_fake_src = model.discriminator(x_fake_src, training=False)
            
            # Generator losses
            g_adv_loss = -tf.reduce_mean(d_fake_src)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)  # Estimation loss (source)
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss = compute_total_smoothness_loss(x_fake_src, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
        
            # Total generator loss (no domain adaptation)
            g_loss = est_weight * g_est_loss + adv_weight * g_adv_loss + smoothness_loss
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
                
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]

        # === 3.1. Optional: Monitor target performance (no training) ===
        if batch_idx < loader_H_true_train_tgt.total_batches:
            x_tgt = loader_H_input_train_tgt.next_batch()
            y_tgt = loader_H_true_train_tgt.next_batch()
            
            # Preprocess target data
            x_tgt = complx2real(x_tgt)
            y_tgt = complx2real(y_tgt)
            x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
            y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
            x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
            y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
            
            # Test on target (no gradients)
            x_fake_tgt, features_tgt = model.generator(x_scaled_tgt, training=False, return_features=False)
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)
            epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === 3.2 Save features (after the bottleneck layer) if required (to calcu PAD) ===
        if save_features:
            # save features in a temporary file instead of stacking them up, to avoid memory exploding
            features_np_source = features_src[-1].numpy()  # Convert to numpy if it's a tensor
            # print('Feature shape: ', features_np_source.shape)
            if features_dataset_source is None:
                # Create dataset with unlimited first dimension
                features_dataset_source = features_h5_source.create_dataset(
                    'features',
                    data=features_np_source,
                    maxshape=(None,) + features_np_source.shape[1:],
                    chunks=True
                )
            else:
                # Resize and append
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_source.shape[0], axis=0)
                features_dataset_source[-features_np_source.shape[0]:] = features_np_source
                
            features_np_target = features_tgt[-1].numpy()
            if features_dataset_target is None:
                # Create dataset with unlimited first dimension
                features_dataset_target = features_h5_target.create_dataset(
                    'features',
                    data=features_np_target,
                    maxshape=(None,) + features_np_target.shape[1:],
                    chunks=True
                )
            else:
                # Resize and append
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
                features_dataset_target[-features_np_target.shape[0]:] = features_np_target
                        

    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train if epoc_loss_est_tgt > 0 else 0.0
    
    # Return compatible output structure (no domain adaptation)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,  # No domain adaptation
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=None,
        film_features_source=None,
        avg_epoc_loss_d=avg_loss_d
    )

def val_step_wgan_gp_source_only(model, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                                    linear_interp=False, return_H_gen=False):
    """
    Validation step for source-only training.
    Validates on source domain, tests on target domain.
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0  # This is now testing on target
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0  # This is now testing on target
    epoc_gan_disc_loss = 0.0
    H_sample = []
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # --- Source domain validation ---
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        N_val_source += x_src.shape[0]

        # Preprocess source
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Source validation prediction
        preds_src, _ = model.generator(x_scaled_src, training=False, return_features=False)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        
        # Source NMSE
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # Discriminator loss (source only)
        d_real_src = model.discriminator(y_scaled_src, training=False)
        d_fake_src = model.discriminator(preds_src, training=False)
        gp_src = gradient_penalty(model.discriminator, y_scaled_src, preds_src, batch_size=x_scaled_src.shape[0])
        d_loss_src = tf.reduce_mean(d_fake_src) - tf.reduce_mean(d_real_src) + 10.0 * gp_src
        epoc_gan_disc_loss += d_loss_src.numpy() * x_src.shape[0]

    # --- Target domain testing (separate loop to handle different batch sizes) ---
    for idx in range(loader_H_true_val_target.total_batches):
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_target += x_tgt.shape[0]

        # Preprocess target
        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # Target testing prediction
        preds_tgt, _ = model.generator(x_scaled_tgt, training=False, return_features=False)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        
        # Target NMSE
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # Save samples from first batch
        if idx == 0:
            n_samples = min(3, x_tgt_real.shape[0])
            #source 
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            # target
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
            
            H_sample = [H_true_sample_target, H_input_sample_target, H_est_sample_target, 
                        nmse_input_target, nmse_est_target]
            H_sample = [H_true_sample,  H_input_sample, H_est_sample, 
                            nmse_input_source, nmse_est_source,
                            H_true_sample_target, H_input_sample_target, H_est_sample_target,
                            nmse_input_target, nmse_est_target]
        if return_H_gen:
            # Convert to numpy if needed and append to lists
            H_gen_src_batch = preds_src_descaled.numpy().copy() if hasattr(preds_src_descaled, 'numpy') else preds_src_descaled.copy()
            H_gen_tgt_batch = preds_tgt_descaled.numpy().copy() if hasattr(preds_tgt_descaled, 'numpy') else preds_tgt_descaled.copy()
            
            all_H_gen_src.append(H_gen_src_batch)
            all_H_gen_tgt.append(H_gen_tgt_batch)
    if return_H_gen:
        # Concatenate all batches along the batch dimension (axis=0)
        H_gen_src_all = np.concatenate(all_H_gen_src, axis=0)
        H_gen_tgt_all = np.concatenate(all_H_gen_tgt, axis=0)
        H_gen = {
            'H_gen_src': H_gen_src_all,
            'H_gen_tgt': H_gen_tgt_all
        }
            
    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target  # This is testing performance
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target  # This is testing performance
    avg_gan_disc_loss = epoc_gan_disc_loss / N_val_source

    # Total loss (source validation only)
    avg_total_loss = est_weight * avg_loss_est_source + adv_weight * avg_gan_disc_loss

    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target,  # This is testing loss
        'avg_loss_est': avg_loss_est_source,  # Use source for validation
        'avg_gan_disc_loss': avg_gan_disc_loss,
        'avg_jmmd_loss': 0.0,  # No domain adaptation
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,  # This is testing NMSE
        'avg_nmse': avg_nmse_source,  # Use source for validation
        'avg_domain_acc_source': 0.5,
        'avg_domain_acc_target': 0.5,
        'avg_domain_acc': 0.5,
        'avg_smoothness_loss': 0.0
    }
    
    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen

    return H_sample, epoc_eval_return


# ================= DEEPER GAN ARCHITECTURE =====================

class ResidualUNetBlock(tf.keras.layers.Layer):
    """Enhanced UNet block with residual connections, using reflect padding like original UNetBlock"""
    
    def __init__(self, filters, apply_dropout=False, kernel_size=(4,3), strides=(2,1), pad_h=0, pad_w=1, gen_l2=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.apply_dropout = apply_dropout
        self.strides = strides
        self.pad_h = pad_h
        self.pad_w = pad_w
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        
        # Main convolution path (same as original UNetBlock)
        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides, padding='valid',
            kernel_regularizer=kernel_regularizer
        )
        self.norm1 = InstanceNormalization()
        
        # Residual processing (same shape after main conv)
        self.conv2 = tf.keras.layers.Conv2D(
            filters, (3,3), strides=(1,1), padding='valid',  # Use 'valid' for reflect padding
            kernel_regularizer=kernel_regularizer
        )
        self.norm2 = InstanceNormalization()
        
        self.conv3 = tf.keras.layers.Conv2D(
            filters, (3,3), strides=(1,1), padding='valid',  # Use 'valid' for reflect padding
            kernel_regularizer=kernel_regularizer
        )
        self.norm3 = InstanceNormalization()
        
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None
        
        # Skip connection adjustment if needed (for residual connection)
        self.residual_conv = None
        
    def build(self, input_shape):
        super().build(input_shape)
        # Create residual connection adjustment if channel dimensions don't match after main conv
        # We need to account for the spatial change from main conv
        if input_shape[-1] != self.filters:
            self.residual_conv = tf.keras.layers.Conv2D(
                self.filters, (1,1), strides=self.strides, padding='valid'
            )
    
    def call(self, x, training=False):
        # Main convolution with spatial reduction (same as original UNetBlock)
        if self.pad_h > 0 or self.pad_w > 0:
            x_padded = reflect_padding_2d(x, pad_h=self.pad_h, pad_w=self.pad_w)
        else:
            x_padded = x
            
        out = self.conv1(x_padded)
        out = self.norm1(out, training=training)
        out = tf.nn.leaky_relu(out)
        
        # Store for residual connection (after spatial reduction)
        residual = out
        
        # Residual processing (same spatial dimensions as 'out')
        # For 3x3 kernel, we need 1 pixel padding on each side
        out_padded = reflect_padding_2d(out, pad_h=1, pad_w=1)
        out = self.conv2(out_padded)
        out = self.norm2(out, training=training)
        out = tf.nn.leaky_relu(out)
        
        out_padded = reflect_padding_2d(out, pad_h=1, pad_w=1)
        out = self.conv3(out_padded)
        out = self.norm3(out, training=training)
        
        # Add residual connection
        out = out + residual
        out = tf.nn.leaky_relu(out)
        
        if self.dropout:
            out = self.dropout(out, training=training)
            
        return out

class SameShapeBlock(tf.keras.layers.Layer):
    """Same-shape processing block for adding depth without spatial reduction using reflect padding"""
    
    def __init__(self, filters, gen_l2=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        
        # Same-shape processing with 'valid' padding (we'll use reflect_padding_2d manually)
        self.conv1 = tf.keras.layers.Conv2D(
            filters, (3,3), strides=(1,1), padding='valid',  # Changed to 'valid'
            kernel_regularizer=kernel_regularizer
        )
        self.norm1 = InstanceNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters, (3,3), strides=(1,1), padding='valid',  # Changed to 'valid'
            kernel_regularizer=kernel_regularizer
        )
        self.norm2 = InstanceNormalization()
        
        # Channel adjustment for residual connection if needed
        self.channel_adjust = None
        
    def build(self, input_shape):
        super().build(input_shape)
        if input_shape[-1] != self.filters:
            self.channel_adjust = tf.keras.layers.Conv2D(
                self.filters, (1,1), strides=(1,1), padding='same'  # 1x1 conv doesn't need reflect padding
            )
    
    def call(self, x, training=False):
        residual = x
        
        # Same-shape processing with reflect padding
        # For 3x3 kernel, we need 1 pixel padding on each side
        out = reflect_padding_2d(x, pad_h=1, pad_w=1)  # Reflect padding
        out = self.conv1(out)
        out = self.norm1(out, training=training)
        out = tf.nn.leaky_relu(out)
        
        out = reflect_padding_2d(out, pad_h=1, pad_w=1)  # Reflect padding
        out = self.conv2(out)
        out = self.norm2(out, training=training)
        
        # Adjust channels for residual if needed
        if self.channel_adjust:
            residual = self.channel_adjust(residual)
        
        # Add residual connection
        out = out + residual
        out = tf.nn.leaky_relu(out)
        
        return out

class ResidualUNetUpBlock(tf.keras.layers.Layer):
    """Enhanced UNet upsampling block with residual connections, similar to original UNetUpBlock"""
    
    def __init__(self, filters, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.apply_dropout = apply_dropout
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        
        # Upsampling (same as original UNetUpBlock)
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size=kernel_size, strides=strides,
            padding='valid', kernel_regularizer=kernel_regularizer
        )
        self.norm1 = InstanceNormalization()
        
        # Residual processing after concatenation with skip connection
        self.conv1 = tf.keras.layers.Conv2D(
            filters, (3,3), strides=(1,1), padding='valid',  # Use 'valid' for reflect padding
            kernel_regularizer=kernel_regularizer
        )
        self.norm2 = InstanceNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters, (3,3), strides=(1,1), padding='valid',  # Use 'valid' for reflect padding
            kernel_regularizer=kernel_regularizer
        )
        self.norm3 = InstanceNormalization()
        
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None
        
        # Channel adjustment for residual connection after concatenation
        self.channel_adjust = tf.keras.layers.Dense(
            filters, use_bias=False,  # Optional: no bias for residual
            kernel_regularizer=kernel_regularizer
        )
    
    def build(self, input_shape):
        super().build(input_shape)
        # We'll create channel_adjust dynamically in call() since we need to know
        # the concatenated tensor shape
        pass
    
    def call(self, x, skip, training=False):
        # Upsampling (same as original UNetUpBlock)
        x = self.deconv(x)
        
        # Handle width adjustment if needed (same as original)
        if x.shape[2] > 14: 
            x = x[:, :, 1:15, :]
            
        x = self.norm1(x, training=training)
        x = tf.nn.relu(x)
        
        # Concatenate with skip connection (same as original)
        concat = tf.concat([x, skip], axis=-1)
        
        if self.dropout:
            concat = self.dropout(concat, training=training)
        
        # Residual processing on concatenated features
        residual = concat
        
        # Process concatenated features with reflect padding
        # For 3x3 kernel, we need 1 pixel padding on each side
        out = reflect_padding_2d(concat, pad_h=1, pad_w=1)
        out = self.conv1(out)
        out = self.norm2(out, training=training)
        out = tf.nn.leaky_relu(out)
        
        out = reflect_padding_2d(out, pad_h=1, pad_w=1)
        out = self.conv2(out)
        out = self.norm3(out, training=training)
        
        # Channel adjustment for residual connection if needed
        if concat.shape[-1] != out.shape[-1]: # or self.filters
            residual = self.channel_adjust(concat)
        else:
            residual = concat
        
        # Add residual connection
        out = out + residual
        out = tf.nn.leaky_relu(out)
            
        return out

class DeeperPix2PixGenerator(tf.keras.Model):
    """Enhanced Pix2Pix generator with residual blocks and same-shape layers"""
    
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, extract_layers=['d4_deep1', 'd4_deep2', 'd4_deep3']):
        # extract_layers needs to be the correct order (the last one is the deepest layer)
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        
        # Configure extraction layers
        self.extract_layers = extract_layers
        
        # Enhanced Encoder with residual blocks and same-shape layers
        # Level 1: (132, 14) -> (65, 14)
        self.down1 = ResidualUNetBlock(32, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down1_same = SameShapeBlock(64, gen_l2=gen_l2)  # Same-shape processing
        
        # Level 2: (65, 14) -> (32, 14)  
        self.down2 = ResidualUNetBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.down2_same = SameShapeBlock(128, gen_l2=gen_l2)  # Same-shape processing
        
        # Level 3: (32, 14) -> (15, 14)
        self.down3 = ResidualUNetBlock(128, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down3_same = SameShapeBlock(256, gen_l2=gen_l2)  # Same-shape processing
        
        # Level 4: (15, 14) -> (7, 14) - Bottleneck with multiple same-shape layers
        self.down4 = ResidualUNetBlock(256, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.down4_same1 = SameShapeBlock(512, gen_l2=gen_l2)  # Deep bottleneck processing
        self.down4_same2 = SameShapeBlock(512, gen_l2=gen_l2)  # Even deeper processing
        self.down4_same3 = SameShapeBlock(1024, gen_l2=gen_l2) # Deepest features
        
        # Enhanced Decoder with residual blocks
        # Bottleneck processing: (7, 14, 1024) -> (7, 14, 256)
        self.up0_same3 = SameShapeBlock(512, gen_l2=gen_l2)  # Process deep features
        self.up0_same2 = SameShapeBlock(512, gen_l2=gen_l2)  # Continue processing
        self.up0_same1 = SameShapeBlock(256, gen_l2=gen_l2)  # Prepare for upsampling
        self.up0 = ResidualUNetUpBlock(256, apply_dropout=True, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        
        # Level 3: (15, 14, 256) -> (32, 14, 128)
        self.up1_same = SameShapeBlock(128, gen_l2=gen_l2)   # Process concatenated features
        self.up1 = ResidualUNetUpBlock(128, apply_dropout=True, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        
        # Level 2: (32, 14, 128) -> (65, 14, 64)
        self.up2_same = SameShapeBlock(64, gen_l2=gen_l2)    # Process concatenated features
        self.up2 = ResidualUNetUpBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        
        # Final output: (65, 14, 64) -> (132, 14, 2)
        self.last = tf.keras.layers.Conv2DTranspose(
            output_channels, kernel_size=(4,3), strides=(2,1), padding='valid',
            activation='tanh', kernel_regularizer=kernel_regularizer
        )
            
    def call(self, x, training=False, return_features=False):
        # Enhanced Encoder with multiple abstraction levels
        d1 = self.down1(x, training=training)              # (65, 14, 32)
        d1_deep = self.down1_same(d1, training=training)   # (65, 14, 64) - same-shape processing
        
        d2 = self.down2(d1_deep, training=training)        # (32, 14, 64) 
        d2_deep = self.down2_same(d2, training=training)   # (32, 14, 128) - same-shape processing
        
        d3 = self.down3(d2_deep, training=training)        # (15, 14, 128)
        d3_deep = self.down3_same(d3, training=training)   # (15, 14, 256) - same-shape processing
        
        d4 = self.down4(d3_deep, training=training)        # (7, 14, 256)
        d4_deep1 = self.down4_same1(d4, training=training) # (7, 14, 512) - deep bottleneck
        d4_deep2 = self.down4_same2(d4_deep1, training=training) # (7, 14, 512) - deeper
        d4_deep3 = self.down4_same3(d4_deep2, training=training) # (7, 14, 1024) - deepest
        
        # Enhanced Decoder with skip connections
        # Bottleneck processing
        u0_proc3 = self.up0_same3(d4_deep3, training=training)  # Process deepest features
        u0_proc2 = self.up0_same2(u0_proc3, training=training)  # Continue processing
        u0_proc1 = self.up0_same1(u0_proc2, training=training)  # Prepare for skip connection
        u0 = self.up0(u0_proc1, d3_deep, training=training)     # Skip from d3_deep
        
        # Level 3 reconstruction
        u1_proc = self.up1_same(u0, training=training)          # Process concat features
        u1 = self.up1(u1_proc, d2_deep, training=training)      # Skip from d2_deep
        
        # Level 2 reconstruction  
        u2_proc = self.up2_same(u1, training=training)          # Process concat features
        u2 = self.up2(u2_proc, d1_deep, training=training)      # Skip from d1_deep
        
        # Final output
        u3 = self.last(u2)
        
        if u3.shape[2] > 14:
            u3 = u3[:, :, 1:15, :]
            
        layer_map = {
                'd1': d1, 'd1_deep': d1_deep,
                'd2': d2, 'd2_deep': d2_deep,
                'd3': d3, 'd3_deep': d3_deep,
                'd4': d4, 'd4_deep1': d4_deep1, 'd4_deep2': d4_deep2, 'd4_deep3': d4_deep3,
                'u0_proc3': u0_proc3, 'u0_proc2': u0_proc2, 'u0_proc1': u0_proc1,
                'u1_proc': u1_proc, 'u2_proc': u2_proc
            }
            
        features = []
        for layer_name in self.extract_layers:
            if layer_name in layer_map:
                layer_tensor = layer_map[layer_name]
                features.append(tf.reshape(layer_tensor, [tf.shape(layer_tensor)[0], -1]))
            else:
                print(f"Warning: Layer '{layer_name}' not found in DeeperPix2PixGenerator")
                
        return u3, features


class DeeperGAN_Output:
    """Output dataclass for Deeper GAN"""
    def __init__(self, gen_out, disc_out, extracted_features):
        self.gen_out = gen_out
        self.disc_out = disc_out
        self.extracted_features = extracted_features


class DeeperGAN(tf.keras.Model):
    """Enhanced GAN with deeper generator and multiple feature extraction"""
    
    def __init__(self, n_subc=132, generator=DeeperPix2PixGenerator, discriminator=PatchGANDiscriminator, 
                gen_l2=None, disc_l2=None, extract_layers=['d4_deep1', 'd4_deep2', 'd4_deep3']):
        super().__init__()
            
        self.generator = generator(
            n_subc=n_subc, 
            gen_l2=gen_l2, 
            extract_layers=extract_layers
        )
        self.discriminator = discriminator(n_subc=n_subc, disc_l2=disc_l2)

    def call(self, inputs, training=False):
        x = inputs
        gen_out, features = self.generator(x, training=training, return_features=True)
        disc_out = self.discriminator(gen_out, training=training)
        
        return DeeperGAN_Output(
            gen_out=gen_out,
            disc_out=disc_out,
            extracted_features=features
        )
def post_val(epoc_val_return, epoch, n_epochs, val_metrics, domain_weight=None):
    """
    Updated post_val function that works with validation metrics dictionary
    
    Args:
        epoc_val_return: Dictionary containing validation results
        epoch: Current epoch number
        n_epochs: Total number of epochs
        val_metrics: Dictionary to store validation metrics
        domain_weight: Whether domain adaptation is used
    """
    #
    val_metrics['val_loss'].append(epoc_val_return['avg_total_loss'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) Weighted Total Loss: {epoc_val_return['avg_total_loss']:.6f}")
    
    val_metrics['val_est_loss'].append(epoc_val_return['avg_loss_est'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) Average Estimation Loss (mean): {epoc_val_return['avg_loss_est']:.6f}")
    
    val_metrics['val_est_loss_source'].append(epoc_val_return['avg_loss_est_source'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) Average Estimation Loss (Source): {epoc_val_return['avg_loss_est_source']:.6f}")
    

    val_metrics['val_est_loss_target'].append(epoc_val_return['avg_loss_est_target'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) Average Estimation Loss (Target): {epoc_val_return['avg_loss_est_target']:.6f}")
    
    val_metrics['val_gan_disc_loss'].append(epoc_val_return['avg_gan_disc_loss'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) GAN Discriminator Loss: {epoc_val_return['avg_gan_disc_loss']:.6f}")
    
    if domain_weight!=0:
        val_metrics['val_domain_disc_loss'].append(epoc_val_return['avg_jmmd_loss'])
        print(f"epoch {epoch+1}/{n_epochs} (Val) JMMD Loss: {epoc_val_return['avg_jmmd_loss']:.6f}")
    
    val_metrics['nmse_val_source'].append(epoc_val_return['avg_nmse_source'])
    val_metrics['nmse_val_target'].append(epoc_val_return['avg_nmse_target'])
    val_metrics['nmse_val'].append(epoc_val_return['avg_nmse'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) NMSE (Source): {epoc_val_return['avg_nmse_source']:.6f}, NMSE (Target): {epoc_val_return['avg_nmse_target']:.6f}, NMSE (Mean): {epoc_val_return['avg_nmse']:.6f}")
    
    val_metrics['source_acc'].append(epoc_val_return['avg_domain_acc_source'])
    val_metrics['target_acc'].append(epoc_val_return['avg_domain_acc_target'])
    val_metrics['acc'].append(epoc_val_return['avg_domain_acc'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) Domain Accuracy (Average): {epoc_val_return['avg_domain_acc']:.4f}")
    
    # Add smoothness loss if it exists
    val_metrics['val_smoothness_loss'].append(epoc_val_return['avg_smoothness_loss'])
    print(f"epoch {epoch+1}/{n_epochs} (Val) Smoothness Loss: {epoc_val_return['avg_smoothness_loss']:.6f}")
        
def save_checkpoint_jmmd(model, save_model, model_path, sub_folder, epoch, metrics):
    exclude_keys = {'figLoss', 'savemat', 'optimizer'}  # Add any keys you want to exclude
    
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
    figLoss(line_list=[(metrics['nmse_val_source'], 'Source Domain'), (metrics['nmse_val_target'], 'Target Domain')], xlabel='Epoch', ylabel='NMSE',
                title='NMSE in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='NMSE_val')

    figLoss(line_list=[(metrics['train_est_loss'], 'Train Loss - Source'), (metrics['val_est_loss_source'], 'Val Loss - Source'), (metrics['val_est_loss_target'], 'Val Loss - Target')], 
                xlabel='Epoch', ylabel='Loss',
                title='Estimation Losses', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='GAN_train')
    
    figLoss(line_list=[(metrics['train_est_loss'], 'GAN Generate Loss'), (metrics['train_disc_loss'], 'GAN Discriminator Loss')], xlabel='Epoch', ylabel='Loss',
                title='Training GAN Losses', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='GAN_train')
    
    # to plot from epoch 30 if length > 35:
    train_loss_data = metrics['train_loss'][30:] if len(metrics['train_loss']) > 35 else metrics['train_loss']
    val_loss_data = metrics['val_loss'][30:] if len(metrics['val_loss']) > 35 else metrics['val_loss']
    figLoss(line_list=[(train_loss_data, 'Training'), (val_loss_data, 'Validating')], xlabel='Epoch', ylabel='Total Loss',
            title='Training and Validating Total Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_total')
    ##
    
    figLoss(line_list=[(metrics['train_est_loss'], 'Training-Source'),  (metrics['train_est_loss_target'], 'Training-Target'), 
                            (metrics['val_est_loss_source'], 'Validating-Source'), (metrics['val_est_loss_target'], 'Validating-Target')], xlabel='Epoch', ylabel='Estimation Loss',
                title='Training and Validating Estimation Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_est')
        # estimation loss: MSE loss, before de-scale
    if domain_weight!=0:
        figLoss(line_list=[(metrics['train_domain_loss'], 'Training'), (metrics['val_domain_disc_loss'], 'Validating')], xlabel='Epoch', ylabel='Domain Loss',
                title='Training and Validating Domain Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_domain')


class SimplePix2PixGenerator(tf.keras.Model):
    """
    Simple Pix2Pix generator WITHOUT skip connections (vanilla GAN architecture)
    Same kernel sizes, strides, and layer structure as your U-Net but without skip connections
    """
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, extract_layers=['d2', 'd3', 'd4']):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.extract_layers = extract_layers
        
        # ===== ENCODER (Same as your U-Net) =====
        self.down1 = UNetBlock(32, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down2 = UNetBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.down3 = UNetBlock(128, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down4 = UNetBlock(256, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)  # Bottleneck
        
        # ===== DECODER (NO SKIP CONNECTIONS) =====
        # Simple upsampling blocks without skip connections
        self.up1 = SimpleUpBlock(128, apply_dropout=True, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.up2 = SimpleUpBlock(64, apply_dropout=True, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.up3 = SimpleUpBlock(32, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        
        # Final output layer
        self.last = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=(4,3), strides=(2,1), padding='valid',
                                                    activation='tanh', kernel_regularizer=kernel_regularizer)
            
    def call(self, x, training=False, return_features=False):
        # ===== ENCODER =====
        d1 = self.down1(x, training=training)      # (batch, 65, 14, 32)
        d2 = self.down2(d1, training=training)     # (batch, 32, 14, 64)
        d3 = self.down3(d2, training=training)     # (batch, 15, 14, 128)
        d4 = self.down4(d3, training=training)     # (batch, 7, 14, 256) - Bottleneck
        
        # ===== DECODER (NO SKIP CONNECTIONS) =====
        u1 = self.up1(d4, training=training)       # (batch, 15, 14, 128) - NO skip from d3
        u2 = self.up2(u1, training=training)       # (batch, 32, 14, 64)  - NO skip from d2
        u3 = self.up3(u2, training=training)       # (batch, 65, 14, 32)  - NO skip from d1
        u4 = self.last(u3)                         # (batch, 132, 14, 2)
        
        if u4.shape[2] > 14:
            u4 = u4[:, :, 1:15, :]
            
        # Feature extraction (same as your U-Net)
        features = []
        layer_map = {
            'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4,
            'u1': u1, 'u2': u2, 'u3': u3
        }
        
        for layer_name in self.extract_layers:
            if layer_name in layer_map:
                layer_tensor = layer_map[layer_name]
                features.append(tf.reshape(layer_tensor, [tf.shape(layer_tensor)[0], -1]))
                
        return u4, features


class SimpleUpBlock(tf.keras.layers.Layer):
    """
    Simple upsampling block WITHOUT skip connections
    Same as UNetUpBlock but removes the concatenation with skip connection
    """
    def __init__(self, filters, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        
        self.deconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,
                                                        padding='valid', kernel_regularizer=kernel_regularizer)
        self.norm = InstanceNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None

    def call(self, x, training):
        # NO skip connection parameter - just process input directly
        x = self.deconv(x)
        
        # Handle width adjustment (same as your U-Net)
        if x.shape[2] > 14: 
            x = x[:, :, 1:15, :]
            
        x = self.norm(x, training=training)
        x = tf.nn.relu(x)
        
        if self.dropout:
            x = self.dropout(x, training=training)
            
        return x