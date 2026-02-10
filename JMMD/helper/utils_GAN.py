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
    def __init__(self, filters, apply_dropout=False, kernel_size=(4,3), 
                strides=(2,1), gen_l2=None, dropOut_rate=0.3):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.deconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,
                                                        padding='valid', kernel_regularizer=kernel_regularizer)
        self.norm = InstanceNormalization()
        self.dropout = tf.keras.layers.Dropout(dropOut_rate) if apply_dropout else None

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
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, 
                dropOut_layers=['u1', 'u2'], dropOut_rate=0.3, extract_layers=['d2', 'd3', 'd4']):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.extract_layers = extract_layers
        # Encoder
        self.down1 = UNetBlock(32, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down2 = UNetBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.down3 = UNetBlock(128, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down4 = UNetBlock(256, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        # Decoder
        self.up1 = UNetUpBlock(128, 
                            apply_dropout='u1' in dropOut_layers,  # True if 'u1' in list
                            kernel_size=(3,3), strides=(2,1), 
                            gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        self.up2 = UNetUpBlock(64, 
                            apply_dropout='u2' in dropOut_layers,  # True if 'u2' in list
                            kernel_size=(4,3), strides=(2,1), 
                            gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        self.up3 = UNetUpBlock(32, 
                            apply_dropout='u3' in dropOut_layers,  # True if 'u3' in list
                            kernel_size=(3,3), strides=(2,1), 
                            gen_l2=gen_l2, dropOut_rate=dropOut_rate)
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
        
        # Check if generator is already initialized (instance) or just a class
        if isinstance(generator, tf.keras.Model):
            # Already initialized - use directly
            self.generator = generator
            print("Using pre-initialized generator")
        else:
            self.generator = generator(n_subc=n_subc, gen_l2=gen_l2, extract_layers=extract_layers)
            
        # Check if discriminator is already initialized (instance) or just a class  
        if isinstance(discriminator, tf.keras.Model):
            # Already initialized - use directly
            self.discriminator = discriminator
            print("Using pre-initialized discriminator")
        else:    
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
    def __init__(self, strategy='domain_first_smooth', **kwargs):  
                    # strategy = 'domain_first_smooth' or 'reconstruction_first'
        self.strategy = strategy
        
        # Common parameters for both strategies
        self.temporal_weight = kwargs.get('temporal_weight', 0.02)
        self.frequency_weight = kwargs.get('frequency_weight', 0.1)
        
        if strategy == 'reconstruction_first':
            # Domain scheduling parameters
            self.start_domain_weight = kwargs.get('start_domain_weight', 0.5)
            self.end_domain_weight = kwargs.get('end_domain_weight', 1.5)
            self.warmup_epochs = kwargs.get('warmup_epochs', 150)
            self.schedule_type = kwargs.get('schedule_type', 'linear')  # 'linear', 'cosine', 'exponential'
            
            # Other weight parameters for reconstruction_first
            self.start_est_weight = kwargs.get('start_est_weight', 1.0)
            self.end_est_weight = kwargs.get('end_est_weight', 1.0)  # Can be different if desired
            self.start_adv_weight = kwargs.get('start_adv_weight', 0.005)
            self.end_adv_weight = kwargs.get('end_adv_weight', 0.005)  # Can be different if desired
            
            print(f"WeightScheduler initialized with reconstruction_first strategy:")
            print(f"  - Domain weight: {self.start_domain_weight} → {self.end_domain_weight}")
            print(f"  - Est weight: {self.start_est_weight} → {self.end_est_weight}")
            print(f"  - Adv weight: {self.start_adv_weight} → {self.end_adv_weight}")
            print(f"  - Warmup epochs: {self.warmup_epochs}")
            print(f"  - Schedule type: {self.schedule_type}")
            
        elif strategy == 'domain_first_smooth':
            # Domain scheduling parameters for domain_first_smooth
            self.start_domain_weight = kwargs.get('start_domain_weight', 4.5)  # High start (4.5)
            self.end_domain_weight = kwargs.get('end_domain_weight', 1.5)     # Low end (1.5)
            
            # Est weight parameters for domain_first_smooth
            self.start_est_weight = kwargs.get('start_est_weight', 0.1)       # Low start (0.1)
            self.end_est_weight = kwargs.get('end_est_weight', 0.6)           # High end (0.6)
            
            # Adv weight parameters for domain_first_smooth
            self.start_adv_weight = kwargs.get('start_adv_weight', 0.03)      # Low start (0.03)
            self.end_adv_weight = kwargs.get('end_adv_weight', 0.08)          # High end (0.08)
            
            print(f"WeightScheduler initialized with domain_first_smooth strategy:")
            print(f"  - Domain weight: {self.start_domain_weight} → {self.end_domain_weight}")
            print(f"  - Est weight: {self.start_est_weight} → {self.end_est_weight}")
            print(f"  - Adv weight: {self.start_adv_weight} → {self.end_adv_weight}")
    
    def get_weights(self, epoch, n_epochs):
        """
        Get complete weights dictionary based on selected strategy
        
        Args:
            epoch: Current epoch (0-based)
            n_epochs: Total number of epochs
            
        Returns:
            weights: Dictionary with all weight values
        """
        if self.strategy == 'domain_first_smooth':
            return self.get_weights_domain_first_smooth(epoch, n_epochs)
        elif self.strategy == 'reconstruction_first':
            return self.get_weights_reconstruction_first(epoch, n_epochs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Available: 'domain_first_smooth', 'reconstruction_first'")
    
    def get_weights_domain_first_smooth(self, epoch, n_epochs):
        """
        Domain-first with smooth sigmoid transitions (YOUR ORIGINAL STRATEGY)
        HIGH domain weight early -> LOW domain weight later
        Now supports customizable start/end weights for all parameters
        """
        import math
        progress = epoch / n_epochs
        
        # Domain weight: High early, smooth decay (customizable range)
        domain_factor = 1.0 / (1 + math.exp(15 * (progress - 0.4)))  # Smooth decline
        domain_weight = self.end_domain_weight + (self.start_domain_weight - self.end_domain_weight) * domain_factor
        
        # Estimation weight: Low early, smooth rise (customizable range)
        est_factor = 1.0 / (1 + math.exp(-12 * (progress - 0.5)))  # Smooth rise
        est_weight = self.start_est_weight + (self.end_est_weight - self.start_est_weight) * est_factor
        
        # Adversarial: Moderate throughout (customizable range)
        adv_weight = self.start_adv_weight + (self.end_adv_weight - self.start_adv_weight) * (1 - domain_factor)
        
        return {
            'adv_weight': adv_weight,
            'est_weight': est_weight,
            'domain_weight': domain_weight,
            'temporal_weight': self.temporal_weight,
            'frequency_weight': self.frequency_weight,
        }
    
    def get_weights_reconstruction_first(self, epoch, n_epochs):
        """
        Reconstruction-first approach (ANTI-NEGATIVE TRANSFER)
        LOW domain weight early -> HIGH domain weight later
        Now returns complete weights dictionary like domain_first_smooth
        """
        # Get gradual weights for all parameters
        current_domain_weight = self._get_gradual_weight(
            epoch, n_epochs, self.start_domain_weight, self.end_domain_weight
        )
        current_est_weight = self._get_gradual_weight(
            epoch, n_epochs, self.start_est_weight, self.end_est_weight
        )
        current_adv_weight = self._get_gradual_weight(
            epoch, n_epochs, self.start_adv_weight, self.end_adv_weight
        )
        
        return {
            'adv_weight': current_adv_weight,
            'est_weight': current_est_weight,
            'domain_weight': current_domain_weight,
            'temporal_weight': self.temporal_weight,
            'frequency_weight': self.frequency_weight,
        }
    
    def _get_gradual_weight(self, epoch, n_epochs, start_weight, end_weight):
        """
        Helper function for gradual weight scheduling (generalized for any weight type)
        """
        if epoch < self.warmup_epochs:
            # Gradual change during warmup
            progress = epoch / self.warmup_epochs
            
            if self.schedule_type == 'linear':
                # Linear change: smooth and predictable
                weight = start_weight + (end_weight - start_weight) * progress
            elif self.schedule_type == 'cosine':
                # Cosine schedule: slower start, faster middle, slower end
                import math
                weight = start_weight + (end_weight - start_weight) * (1 - math.cos(progress * math.pi)) / 2
            else:  # exponential
                # Exponential: very slow start, rapid change later
                import math
                if end_weight > 0 and start_weight > 0:
                    weight = start_weight * (end_weight / start_weight) ** progress
                else:
                    # Fallback to linear if weights can be zero
                    weight = start_weight + (end_weight - start_weight) * progress
        else:
            # Maintain final weight after warmup
            weight = end_weight
        
        return weight
    
    
class JMMDLoss(keras.layers.Layer):
    """
    Joint Maximum Mean Discrepancy Loss in TensorFlow
    Computes JMMD across multiple feature layers
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                    enable_pooling=False, **kwargs):
        super(JMMDLoss, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.enable_pooling = enable_pooling
    
        if enable_pooling:
            print(f"JMMDLoss initialized with global pooling enabled")
        else:
            print(f"JMMDLoss initialized without pooling")
    
    def _pool_feature(self, feature):
        """
        Apply global average pooling to a single feature tensor
        """
        if len(feature.shape) == 4:  # [B, H, W, C]
            return tf.reduce_mean(feature, axis=[1, 2])
        elif len(feature.shape) == 3:  # [B, H, C] or [B, W, C]
            return tf.reduce_mean(feature, axis=1)
        elif len(feature.shape) == 2:  # [B, C] - already pooled
            return feature
        else:
            # Flatten and return
            return tf.reshape(feature, [tf.shape(feature)[0], -1])
    
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
        Compute JMMD across multiple layers with optional pooling
        source_list: list of source features from different layers
        target_list: list of target features from different layers
        """
        jmmd_loss = 0.0
        for source_feat, target_feat in zip(source_list, target_list):
            # === APPLY POOLING IF ENABLED ===
            if self.enable_pooling:
                source_pooled = self._pool_feature(source_feat)
                target_pooled = self._pool_feature(target_feat)
            else:
                # Flatten if not pooling
                if len(source_feat.shape) > 2:
                    source_pooled = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
                    target_pooled = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
                else:
                    source_pooled = source_feat
                    target_pooled = target_feat
            
            # Compute MMD for this layer
            jmmd_loss += self.mmd(source_pooled, target_pooled)
        
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
    domain_weight = weights.get('domain_weight')
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_jmmd = 0.0
    N_train = 0
    
    if save_features==True and (domain_weight != 0):
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
                    domain_weight * jmmd_loss + 
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        
        # === 3. Save features (after the bottleneck layer) if required (to calcu PAD) ===
        if save_features and (domain_weight != 0):
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
    if save_features and (domain_weight != 0):    
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
    To PCA fit and save features for PAD calculation
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
    domain_weight = weights.get('domain_weight')
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_jmmd = 0.0
    N_train = 0
    
    if save_features==True and (domain_weight != 0):
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
                    domain_weight * jmmd_loss + 
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        
        # === 3. Save features (after the bottleneck layer) if required (to calcu PAD) ===
        if save_features and (domain_weight != 0):
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
    if save_features and (domain_weight != 0):    
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
    domain_weight = weights.get('domain_weight')  # Higher default for stronger adaptation
    
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
    
    if save_features==True and (domain_weight != 0):
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
                    domain_weight * jmmd_loss + 
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)

        # === 3. Save features (after the bottleneck layer) if required (to calcu PAD) ===
        if save_features and (domain_weight != 0):
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
    if save_features and (domain_weight != 0):    
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
        adv_weight, est_weight, domain_weight: loss weights
        
    Returns:
        H_sample, epoc_eval_return
    """
    from sklearn.metrics import accuracy_score
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight', 0.5)
    
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
        if domain_weight > 0:
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
                     + domain_weight * avg_jmmd_loss + avg_smoothness_loss

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
        adv_weight, est_weight, domain_weight: loss weights
        
    Returns:
        H_sample, epoc_eval_return
    """
    from sklearn.metrics import accuracy_score
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight')
    
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
        if domain_weight > 0:
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
                     + domain_weight * avg_jmmd_loss + avg_smoothness_loss

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
    
    def __init__(self, filters, apply_dropout=False, dropOut_rate=0.3,
                 kernel_size=(4,3), strides=(2,1), gen_l2=None, **kwargs):
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
        
        self.dropout = tf.keras.layers.Dropout(dropOut_rate) if apply_dropout else None
        
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
    
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, 
                dropOut_layers=['up0', 'up1'], dropOut_rate=0.3, 
                extract_layers=['d4_deep1', 'd4_deep2', 'd4_deep3']):
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
        self.up0 = ResidualUNetUpBlock(256, apply_dropout='up0' in dropOut_layers, 
                                       kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        
        # Level 3: (15, 14, 256) -> (32, 14, 128)
        self.up1_same = SameShapeBlock(128, gen_l2=gen_l2)   # Process concatenated features
        self.up1 = ResidualUNetUpBlock(128, apply_dropout='up1' in dropOut_layers, 
                                       kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        
        # Level 2: (32, 14, 128) -> (65, 14, 64)
        self.up2_same = SameShapeBlock(64, gen_l2=gen_l2)    # Process concatenated features
        self.up2 = ResidualUNetUpBlock(64, apply_dropout='up2' in dropOut_layers,
                                       kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        
        # Final output: (65, 14, 64) -> (132, 14, 2)
        self.last = tf.keras.layers.Conv2DTranspose(
            output_channels, kernel_size=(4,3), strides=(2,1), padding='valid',
            activation='tanh', kernel_regularizer=kernel_regularizer
        )
            
    def call(self, x, training=False, return_features=False, depth='deep'): # depth ='deep' or 'deeper'
        # Enhanced Encoder with multiple abstraction levels
        d1 = self.down1(x, training=training)              # (65, 14, 32)
        d1_deep = self.down1_same(d1, training=training) if depth=='deeper' else d1
        # (65, 14, 64 or 32) - same-shape processing
        
        d2 = self.down2(d1_deep, training=training)        # (32, 14, 64) 
        d2_deep = self.down2_same(d2, training=training) if depth=='deeper' else d2
        # (32, 14, 128 or 64) - same-shape processing
        
        d3 = self.down3(d2_deep, training=training)        # (15, 14, 128)
        d3_deep = self.down3_same(d3, training=training) if depth=='deeper' else d3
        # (15, 14, 256 or 128) - same-shape processing  
        
        d4 = self.down4(d3_deep, training=training)        # (7, 14, 256)
        d4_deep1 = self.down4_same1(d4, training=training) if depth=='deeper' else d4  
                # (7, 14, 512) - deep bottleneck
        d4_deep2 = self.down4_same2(d4_deep1, training=training) # (7, 14, 512) - deeper
        d4_deep3 = self.down4_same3(d4_deep2, training=training) # (7, 14, 1024) - deepest
        
        # Enhanced Decoder with skip connections
        # Bottleneck processing
        u0_proc3 = self.up0_same3(d4_deep3, training=training)  # Process deepest features
        u0_proc2 = self.up0_same2(u0_proc3, training=training)  # Continue processing
        u0_proc1 = self.up0_same1(u0_proc2, training=training) if depth=='deeper' else u0_proc2
        u0 = self.up0(u0_proc1, d3_deep, training=training)     # Skip from d3_deep
        
        # Level 3 reconstruction
        u1_proc = self.up1_same(u0, training=training) if depth=='deeper' else u0          # Process concat features
        u1 = self.up1(u1_proc, d2_deep, training=training)      # Skip from d2_deep
        
        # Level 2 reconstruction  
        u2_proc = self.up2_same(u1, training=training) if depth=='deeper' else u1          # Process concat features
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
        
        if isinstance(generator, tf.keras.Model):
            # Already initialized - use directly
            self.generator = generator
            print("Using pre-initialized generator")
        else:    
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
        val_metrics['val_domain_disc_loss'].append(epoc_val_return['avg_domain_loss'])
        print(f"epoch {epoch+1}/{n_epochs} (Val) domain Loss: {epoc_val_return['avg_domain_loss']:.6f}")
    
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

class MultiScalePix2PixGenerator(tf.keras.Model):
    """
    Multi-scale approach: Multiple parallel paths with different receptive fields
    Preserves spatial detail while allowing domain adaptation
    """
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, extract_layers=['d2', 'd3', 'd4']):
        super().__init__()
        self.extract_layers = extract_layers
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        
        # ===== MAIN ENCODING PATH (for domain adaptation) =====
        self.down1 = UNetBlock(32, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down2 = UNetBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)  
        self.down3 = UNetBlock(128, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down4 = UNetBlock(256, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        
        # ===== PARALLEL DETAIL PRESERVATION PATH =====
        # Shallow processing to preserve spatial details
        self.detail_conv1 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)
        self.detail_conv2 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)
        self.detail_conv3 = tf.keras.layers.Conv2D(32, (4,3), strides=(2,1), padding='valid', activation='relu', kernel_regularizer=kernel_regularizer)
        
        # ===== DECODER WITH SKIP CONNECTIONS =====
        # Use regular UNetUpBlock but with modified skip connections
        self.up1 = UNetUpBlock(128, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.up2 = UNetUpBlock(64, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.up3 = UNetUpBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        
        # ===== FUSION LAYER =====
        self.fusion_conv = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)
        self.last = tf.keras.layers.Conv2DTranspose(output_channels, (4,3), strides=(2,1), 
                                            padding='valid', activation='tanh', kernel_regularizer=kernel_regularizer)
    
    def call(self, x, training=False, return_features=False):  # 
        # ===== MAIN ENCODING PATH =====
        d1 = self.down1(x, training=training)
        d2 = self.down2(d1, training=training) 
        d3 = self.down3(d2, training=training)
        d4 = self.down4(d3, training=training)  # ← Domain adaptation happens here
        
        # ===== DETAIL PRESERVATION PATH =====
        detail = self.detail_conv1(x)
        detail = self.detail_conv2(detail)
        detail = reflect_padding_2d(detail, pad_h=0, pad_w=1)  
        detail = self.detail_conv3(detail)  # (132, 14, 32) - preserves spatial resolution
        
        # ===== DECODER WITH REDUCED SKIP INFLUENCE =====
        # Apply skip connection weighting BEFORE passing to UNetUpBlock
        d3_weighted = d3 * 0.3  # Weak skip connection
        d2_weighted = d2 * 0.2  # Weaker skip  
        d1_weighted = d1 * 0.1  # Very weak skip
        
        u1 = self.up1(d4, d3_weighted, training=training) 
        u2 = self.up2(u1, d2_weighted, training=training)  
        u3 = self.up3(u2, d1_weighted, training=training)  
        
        # ===== FUSION =====
        # Combine decoded features with detail path
        fused = tf.concat([u3, detail], axis=-1)  # (132, 14, 64)
        fused = self.fusion_conv(fused)
        output = self.last(fused)
        
        if output.shape[2] > 14:
            output = output[:, :, 1:15, :]
        
        
        features = []
        layer_map = {'d1': d1, 'd2': d2, 'd3': d3, 'd4': d4}
        for layer_name in self.extract_layers:
            if layer_name in layer_map:
                features.append(tf.reshape(layer_map[layer_name], [tf.shape(layer_map[layer_name])[0], -1]))
                
        return output, features  
    
class AttentionPix2PixGenerator(tf.keras.Model):
    """
    Use attention to selectively choose between skip connection and adapted features
    Compatible with existing GAN framework
    """
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, extract_layers=['d2', 'd3', 'd4']):
        super().__init__()
        self.extract_layers = extract_layers
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        
        # Standard encoder (compatible with your existing framework)
        self.down1 = UNetBlock(32, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down2 = UNetBlock(64, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.down3 = UNetBlock(128, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.down4 = UNetBlock(256, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        
        # Use regular UNetUpBlock with attention preprocessing
        self.up1 = UNetUpBlock(128, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        self.up2 = UNetUpBlock(64, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
        self.up3 = UNetUpBlock(32, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
        
        # Attention mechanisms for skip connections
        self.attention1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.attention2 = tf.keras.layers.Dense(1, activation='sigmoid')  
        self.attention3 = tf.keras.layers.Dense(1, activation='sigmoid')
        
        self.last = tf.keras.layers.Conv2DTranspose(output_channels, (4,3), strides=(2,1), 
                                                padding='valid', activation='tanh', kernel_regularizer=kernel_regularizer)
        
        # Store domain weight for use during training
        self.current_domain_weight = 0.0
    
    def call(self, x, training=False, return_features=False, domain_weight=None):  
        current_domain_weight = domain_weight if domain_weight is not None else self.current_domain_weight
        
        # Standard encoding  
        d1 = self.down1(x, training=training)
        d2 = self.down2(d1, training=training)
        d3 = self.down3(d2, training=training) 
        d4 = self.down4(d3, training=training)
        
        # ===== ATTENTION-WEIGHTED SKIP CONNECTIONS =====
        # Attention for d3 skip connection
        d3_pooled = tf.reduce_mean(d3, axis=[1,2], keepdims=True)  # Global average pooling
        attention3 = self.attention1(d3_pooled)
        attention3 = attention3 * (1.0 - current_domain_weight)  # Reduce during adaptation
        d3_weighted = d3 * attention3
        
        # Attention for d2 skip connection  
        d2_pooled = tf.reduce_mean(d2, axis=[1,2], keepdims=True)
        attention2 = self.attention2(d2_pooled)
        attention2 = attention2 * (1.0 - self.current_domain_weight)
        d2_weighted = d2 * attention2
        
        # Attention for d1 skip connection
        d1_pooled = tf.reduce_mean(d1, axis=[1,2], keepdims=True)
        attention1 = self.attention3(d1_pooled)  
        attention1 = attention1 * (1.0 - self.current_domain_weight)
        d1_weighted = d1 * attention1
        
        # ===== STANDARD DECODER WITH ATTENTION-WEIGHTED SKIPS =====
        u1 = self.up1(d4, d3_weighted, training=training)   
        u2 = self.up2(u1, d2_weighted, training=training)  
        u3 = self.up3(u2, d1_weighted, training=training)  
        
        output = self.last(u3)
        if output.shape[2] > 14:
            output = output[:, :, 1:15, :]
            
        # 
        features = []
        layer_map = {'d1': d1, 'd2': d2, 'd3': d3, 'd4': d4}
        for layer_name in self.extract_layers:
            if layer_name in layer_map:
                features.append(tf.reshape(layer_map[layer_name], [tf.shape(layer_map[layer_name])[0], -1]))
                
        return output, features  

class AttentionUNetUpBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(4,3), strides=(2,1), gen_l2=None):
        super().__init__()
        self.deconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='valid')
        self.norm = InstanceNormalization()
        
        # Attention mechanism for skip connection
        self.attention = tf.keras.layers.Dense(1, activation='sigmoid')  # Learn skip weight
        
    def call(self, x, skip, training, domain_weight=0.0):
        x = self.deconv(x)
        if x.shape[2] > 14:
            x = x[:, :, 1:15, :]
        x = self.norm(x, training=training)
        x = tf.nn.relu(x)
        
        # Learn attention weights for skip connection
        # Higher domain_weight → lower skip attention (more adaptation)
        skip_attention = self.attention(tf.reduce_mean(skip, axis=[1,2], keepdims=True))
        skip_attention = skip_attention * (1.0 - domain_weight)  # Reduce skip during adaptation
        
        # Apply attention to skip connection
        weighted_skip = skip * skip_attention
        
        x = tf.concat([x, weighted_skip], axis=-1)
        return x
    
    
### Self training
def assess_channel_quality(H_estimated, quality_thresholds=None):
    """
    Assess quality based on channel estimation physics
    No ground truth needed - just reasonable channel properties
    """
    if quality_thresholds is None:
        quality_thresholds = {
            'freq_smoothness_weight': 1.0,
            'temporal_smoothness_weight': 0.5,
            'energy_weight': 0.3,
            'range_weight': 2.0
        }
    
    # 1. Frequency smoothness (channels should be smooth across subcarriers)
    freq_diff = tf.abs(H_estimated[:, 1:, :, :] - H_estimated[:, :-1, :, :])
    freq_smoothness = -tf.reduce_mean(freq_diff) * quality_thresholds['freq_smoothness_weight']
    
    # 2. Temporal smoothness (adjacent symbols should be correlated) 
    temporal_diff = tf.abs(H_estimated[:, :, 1:, :] - H_estimated[:, :, :-1, :])
    temporal_smoothness = -tf.reduce_mean(temporal_diff) * quality_thresholds['temporal_smoothness_weight']
    
    # 3. Energy consistency (reasonable amplitude range)
    energy = tf.reduce_mean(tf.square(H_estimated))
    energy_penalty = -tf.abs(energy - 0.5) * quality_thresholds['energy_weight']  # Target energy ~0.5
    
    # 4. Amplitude range check (for complex channels)
    amplitude = tf.sqrt(tf.square(H_estimated[:,:,:,0]) + tf.square(H_estimated[:,:,:,1]))
    range_violations = tf.reduce_mean(tf.maximum(0.0, amplitude - 2.0))  # Penalize >2.0 amplitude
    range_score = -range_violations * quality_thresholds['range_weight']
    
    # Combine scores
    total_quality = freq_smoothness + temporal_smoothness + energy_penalty + range_score
    
    return total_quality

def get_prediction_confidence(model, x_target, n_samples=5, dropout_rate=0.1):
    """
    Get prediction confidence using multiple forward passes with dropout
    """
    predictions = []
    
    # Enable dropout during inference for variation
    for i in range(n_samples):
        # Use training=True to enable dropout during inference
        pred, _ = model.generator(x_target, training=True)
        predictions.append(pred)
    
    predictions = tf.stack(predictions, axis=0)  # (n_samples, batch, 132, 14, 2)
    
    # Calculate mean and variance
    mean_pred = tf.reduce_mean(predictions, axis=0)
    variance = tf.reduce_mean(tf.square(predictions - mean_pred), axis=0)
    
    # Confidence score: lower variance = higher confidence
    confidence_score = 1.0 / (1.0 + tf.reduce_mean(variance, axis=[1,2,3]))
    
    return mean_pred, confidence_score

def filter_confident_predictions(predictions, inputs, targets, confidence_scores, quality_scores,
                                confidence_threshold=0.8, quality_threshold=0.5):
    """
    Filter predictions based on confidence and quality criteria
    """
    # Combined filtering criteria
    confident_mask = (confidence_scores > confidence_threshold) & (quality_scores > quality_threshold)
    
    if tf.reduce_sum(tf.cast(confident_mask, tf.int32)) == 0:
        return None, None, None, 0
    
    # Filter data
    filtered_predictions = tf.boolean_mask(predictions, confident_mask)
    filtered_inputs = tf.boolean_mask(inputs, confident_mask)
    filtered_targets = tf.boolean_mask(targets, confident_mask) if targets is not None else None
    
    n_kept = tf.reduce_sum(tf.cast(confident_mask, tf.int32))
    
    return filtered_predictions, filtered_inputs, filtered_targets, n_kept

def train_step_wgan_gp_self_training(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                                save_features=False, nsymb=14, weights=None, linear_interp=False,
                                confidence_threshold=0.8, quality_threshold=0.5,
                                self_training_ratio=0.3):
    """
    WGAN-GP training step with self-training approach
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
    epoc_loss_est_tgt = 0.0
    epoc_confidence_score = 0.0
    N_train = 0
    N_confident_samples = 0
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    # === Phase 1: Generate confident target pseudo-labels ===
    print("=== Generating pseudo-labels for target domain ===")
    target_pseudo_data = []
    
    # Calculate how many target batches to use for pseudo-label generation
    max_target_batches = min(loader_H_input_train_tgt.total_batches, 
                            int(loader_H_true_train_src.total_batches * self_training_ratio))
    
    # Reset target loader for Phase 1
    loader_H_input_train_tgt.reset()
    
    for batch_idx in range(max_target_batches):
        try:
            x_tgt = loader_H_input_train_tgt.next_batch()
        except StopIteration:
            print(f"Target loader exhausted at batch {batch_idx}/{max_target_batches}")
            break
            
        # Preprocess target
        x_tgt_real = complx2real(x_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        
        # Generate predictions with confidence estimation
        pred_tgt, confidence_scores = get_prediction_confidence(model, x_scaled_tgt, n_samples=5)
        
        # Assess quality using channel estimation physics
        quality_scores = assess_channel_quality(pred_tgt)
        
        # Filter confident predictions
        confident_mask = (confidence_scores > confidence_threshold) & (quality_scores > quality_threshold)
        
        if tf.reduce_sum(tf.cast(confident_mask, tf.int32)) > 0:
            # Keep only confident samples
            filtered_preds = tf.boolean_mask(pred_tgt, confident_mask)
            filtered_inputs = tf.boolean_mask(x_scaled_tgt, confident_mask)
            
            target_pseudo_data.append({
                'inputs': filtered_inputs,
                'labels': filtered_preds,
                'confidence': tf.reduce_mean(tf.boolean_mask(confidence_scores, confident_mask))
            })
            
            N_confident_samples += tf.reduce_sum(tf.cast(confident_mask, tf.int32)).numpy()
    
    print(f"Generated {N_confident_samples} confident target pseudo-labels from {len(target_pseudo_data)} batches")
    
    # === Phase 2: Train with source + confident target data ===
    print("=== Training with source + confident target data ===")
    
    # Reset all loaders for Phase 2
    loader_H_input_train_src.reset()
    loader_H_true_train_src.reset()
    loader_H_input_train_tgt.reset()  # Reset again for monitoring
    loader_H_true_train_tgt.reset()   # Reset for monitoring
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get source data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        N_train += x_src.shape[0]

        # Preprocess source
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Combine with confident target data if available
        if target_pseudo_data and batch_idx < len(target_pseudo_data):
            target_batch = target_pseudo_data[batch_idx]
            x_combined = tf.concat([x_scaled_src, target_batch['inputs']], axis=0)
            y_combined = tf.concat([y_scaled_src, target_batch['labels']], axis=0)
            batch_confidence = target_batch['confidence']
        else:
            x_combined = x_scaled_src
            y_combined = y_scaled_src
            batch_confidence = 0.0

        # === 1. Train Discriminator (WGAN-GP) ===
        with tf.GradientTape() as tape_d:
            x_fake_combined, _ = model.generator(x_combined, training=True)
            d_real = model.discriminator(y_combined, training=True)
            d_fake = model.discriminator(x_fake_combined, training=True)
            
            gp = gradient_penalty(model.discriminator, y_combined, x_fake_combined, batch_size=tf.shape(x_combined)[0])
            lambda_gp = 10.0

            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            if model.discriminator.losses:
                d_loss += tf.add_n(model.discriminator.losses)
                
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]

        # === 2. Train Generator ===
        with tf.GradientTape() as tape_g:
            x_fake_combined, features_combined = model.generator(x_combined, training=True)
            d_fake_combined = model.discriminator(x_fake_combined, training=False)
            
            # Generator losses
            g_adv_loss = -tf.reduce_mean(d_fake_combined)
            g_est_loss = loss_fn_est(y_combined, x_fake_combined)
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss = compute_total_smoothness_loss(x_fake_combined, 
                                                              temporal_weight=temporal_weight, 
                                                              frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
        
            # Total generator loss
            g_loss = est_weight * g_est_loss + adv_weight * g_adv_loss + smoothness_loss
            
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_confidence_score += batch_confidence * x_src.shape[0]

        # === 3. Monitor target performance separately (FIXED) ===
        # Only monitor if target batches are still available
        if batch_idx < loader_H_true_train_tgt.total_batches:
            try:
                x_tgt = loader_H_input_train_tgt.next_batch()
                y_tgt = loader_H_true_train_tgt.next_batch()
                
                x_tgt_real = complx2real(x_tgt)
                y_tgt_real = complx2real(y_tgt)
                x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
                y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
                x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
                y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
                
                x_fake_tgt, features_tgt = model.generator(x_scaled_tgt, training=False)
                g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)
                epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]
                
            except StopIteration:
                # Target loader exhausted - skip monitoring for remaining batches
                pass

        
    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train if epoc_loss_est_tgt > 0 else 0.0
    avg_confidence_score = epoc_confidence_score / N_train
    
    # Return compatible output structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_confidence_score,  # Use confidence as domain metric
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=None,
        film_features_source=None,
        avg_epoc_loss_d=avg_loss_d
    )
    
def val_step_wgan_gp_self_training(model, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                                linear_interp=False, return_H_gen=False, 
                                confidence_threshold=0.7, quality_threshold=0.4):
    """
    Validation step for self-training approach - similar to val_step_wgan_gp_jmmd
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (input_src, true_src, input_tgt, true_tgt) DataLoaders
        loss_fn: tuple of (estimation loss, binary cross-entropy loss)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        confidence_threshold: threshold for confident predictions during validation
        quality_threshold: threshold for quality assessment during validation
        
    Returns:
        H_sample, epoc_eval_return (same structure as val_step_wgan_gp_jmmd)
    """
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_confidence_score = 0.0  # Track average confidence instead of JMMD
    epoc_quality_score = 0.0     # Track average quality score
    epoc_smoothness_loss = 0.0
    confident_target_samples = 0  # Count confident target predictions
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

        # === Target domain prediction with confidence assessment ===
        # Generate predictions with confidence estimation for target
        preds_tgt_confident, confidence_scores = get_prediction_confidence(model, x_scaled_tgt, n_samples=3)
        
        # Use deterministic prediction for main evaluation
        preds_tgt, features_tgt = model.generator(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # === Self-training specific metrics ===
        # Assess quality of target predictions
        quality_scores = assess_channel_quality(preds_tgt_confident)
        
        # Track confidence and quality scores
        avg_confidence_batch = tf.reduce_mean(confidence_scores)
        avg_quality_batch = quality_scores  # Already a scalar
        epoc_confidence_score += avg_confidence_batch.numpy() * x_tgt.shape[0]
        epoc_quality_score += avg_quality_batch.numpy() * x_tgt.shape[0]
        
        # Count confident predictions (for monitoring self-training effectiveness)
        confident_mask = (confidence_scores > confidence_threshold) & (quality_scores > quality_threshold)
        confident_target_samples += tf.reduce_sum(tf.cast(confident_mask, tf.int32)).numpy()

        # === WGAN Discriminator Scores (for monitoring only) ===
        # Only considering source domain (same as JMMD version)
        d_real_src = model.discriminator(y_scaled_src, training=False)
        d_fake_src = model.discriminator(preds_src, training=False)
        gp_src = gradient_penalty(model.discriminator, y_scaled_src, preds_src, batch_size=x_scaled_src.shape[0])
        lambda_gp = 10.0
        
        # WGAN critic loss: mean(fake) - mean(real)
        d_loss_src = tf.reduce_mean(d_fake_src) - tf.reduce_mean(d_real_src) + lambda_gp * gp_src
        
        # only observe GAN disc loss on source dataset
        epoc_gan_disc_loss += d_loss_src.numpy() * x_src.shape[0]

        # === ADD SMOOTHNESS LOSS COMPUTATION ===
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
    
    # Self-training specific metrics (replace JMMD)
    avg_confidence_score = epoc_confidence_score / N_val_target
    avg_quality_score = epoc_quality_score / N_val_target
    confident_ratio = confident_target_samples / N_val_target  # Percentage of confident predictions
    
    # Use confidence score as "domain loss" replacement for compatibility
    avg_jmmd_loss = avg_confidence_score  # Higher confidence = better "domain alignment"
    
    # smoothness loss average
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    
    # For compatibility with existing code, we'll set domain accuracy based on confidence
    # High confidence = good domain adaptation
    avg_domain_acc_source = 1.0  # Source is always "confident"
    avg_domain_acc_target = confident_ratio  # Target confidence ratio as "accuracy"
    avg_domain_acc = (avg_domain_acc_source + avg_domain_acc_target) / 2

    # Weighted total loss (for comparison with training) - use confidence as domain metric
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss \
                    + avg_confidence_score + avg_smoothness_loss  # Confidence as "domain" component

    # Compose epoc_eval_return - Same structure as val_step_wgan_gp_jmmd
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': avg_gan_disc_loss,
        'avg_domain_loss': avg_jmmd_loss,  # Use confidence score as JMMD replacement
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss,
        # Additional self-training specific metrics
        'avg_confidence_score': avg_confidence_score,
        'avg_quality_score': avg_quality_score,
        'confident_target_ratio': confident_ratio
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return


# CNN 
class CNNGenerator(tf.keras.Model):
    """
    CNN using your existing SameShapeBlock
    """
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None, 
                n_blocks=6, base_filters=32, extract_layers=['block_2', 'block_3']):
        super().__init__()
        self.n_blocks = n_blocks
        self.extract_layers = extract_layers
        
        # Input adaptation: 2 channels -> base_filters channels
        self.input_conv = tf.keras.layers.Conv2D(
            base_filters, (3, 3), padding='valid', activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(gen_l2) if gen_l2 else None
        )
        
        # Use existing SameShapeBlock with progressive channel expansion
        self.blocks = []
        for i in range(n_blocks):
            # Pyramid channel progression strategy
            if i ==0:
                filters = 64
                print(f"Block {i+1}: Using {filters} filters (increasing)")
            elif i < n_blocks // 2:
                # First half: exponential increase
                filters = min(base_filters * (4 ** i), 1024)   # 2** or 4**
                print(f"Block {i+1}: Using {filters} filters (increasing)")
            elif i == n_blocks -1:
                filters = 64
                print(f"Block {i+1}: Using {filters} filters (decreasing)")
            else:
                # Second half: exponential decrease (mirror of first half)
                mirror_index = n_blocks - i - 1
                filters = min(base_filters * (4 ** mirror_index), 1024)
                print(f"Block {i+1}: Using {filters} filters (decreasing)")
            
            # Use existing SameShapeBlock
            block = SameShapeBlock(filters=filters, gen_l2=gen_l2)
            self.blocks.append(block)
        
        # Output adaptation: final_filters -> output_channels
        self.output_conv = tf.keras.layers.Conv2D(
            output_channels, (3, 3), padding='valid', activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(gen_l2) if gen_l2 else None
        )
    
    def call(self, x, training=False, return_features=False):
        # Input adaptation: (132, 14, 2) -> (132, 14, base_filters)
        x = reflect_padding_2d(x, pad_h=1, pad_w=1)  # Reflect padding
        out = self.input_conv(x)
        
        # Store intermediate features for extraction
        block_outputs = {}
        
        # Process through your SameShapeBlocks
        for i, block in enumerate(self.blocks):
            out = block(out, training=training)
            block_outputs[f'block_{i+1}'] = out
        
        # Output adaptation: (132, 14, final_filters) -> (132, 14, 2)
        out = reflect_padding_2d(out, pad_h=1, pad_w=1)  # Reflect padding
        output = self.output_conv(out)
        
        # Feature extraction for domain adaptation
        features = []
        for layer_name in self.extract_layers:
            if layer_name in block_outputs:
                feature_tensor = block_outputs[layer_name]
                # features.append(tf.reshape(feature_tensor, [tf.shape(feature_tensor)[0], -1]))
                features.append(feature_tensor)  # Keep [B, H, W, C] shape
                
        return output, features
    
    
# Residual Approach
def train_step_wgan_gp_jmmd_residual(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                                save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    WGAN-GP training step with JMMD using residual learning approach
    Model predicts residual correction instead of direct channel estimation
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss)
        optimizers: tuple of (gen_optimizer, disc_optimizer)
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for PAD computation
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]  # Only need first two loss functions
    gen_optimizer, disc_optimizer = optimizers[:2]  # No domain optimizer needed
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight')
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLossNormalized()
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_jmmd = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude
    N_train = 0
    
    if save_features==True and (domain_weight != 0):
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # --- Get data (same as original) ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess (source) - same as original
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target) - same as original
        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === 1. Train Discriminator (WGAN-GP) with RESIDUAL learning ===
        with tf.GradientTape() as tape_d:
            # RESIDUAL LEARNING: Generate residual correction
            residual_src, _ = model.generator(x_scaled_src, training=True)
            x_fake_src = x_scaled_src + residual_src  # ← KEY CHANGE: Add residual to input
            
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)  # ← Discriminate corrected channels
            
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

        # === 2. Train Generator with RESIDUAL learning + JMMD ===
        with tf.GradientTape() as tape_g:
            # RESIDUAL LEARNING: Generate residual corrections with features
            residual_src, features_src = model.generator(x_scaled_src, training=True)
            x_fake_src = x_scaled_src + residual_src  # ← Apply residual correction
            
            residual_tgt, features_tgt = model.generator(x_scaled_tgt, training=True)
            x_fake_tgt = x_scaled_tgt + residual_tgt  # ← Apply residual correction
            
            d_fake_src = model.discriminator(x_fake_src, training=False)
            
            # Generator losses on CORRECTED channels
            g_adv_loss = -tf.reduce_mean(d_fake_src)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)          # ← Loss on corrected source
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)      # ← Loss on corrected target (monitoring)
            
            # JMMD loss between residual features (encourages similar correction patterns)
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            
            # RESIDUAL REGULARIZATION: Encourage small, meaningful corrections
            residual_reg_src = tf.reduce_mean(tf.square(residual_src))
            residual_reg_tgt = tf.reduce_mean(tf.square(residual_tgt))
            residual_reg = (residual_reg_src + residual_reg_tgt) / 2
            residual_penalty = 0.001 * residual_reg  # Small penalty for large residuals
            
            # ADD TEMPORAL SMOOTHNESS LOSS on corrected channels
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss_src = compute_total_smoothness_loss(x_fake_src, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss_tgt = compute_total_smoothness_loss(x_fake_tgt, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            else:
                smoothness_loss = 0.0
        
            # Total generator loss with residual regularization
            g_loss = (est_weight * g_est_loss + 
                    adv_weight * g_adv_loss + 
                    domain_weight * jmmd_loss + 
                    residual_penalty +                # ← Encourage small residuals
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        
        # === 3. Save features if required (same as original) ===
        if save_features and (domain_weight != 0):
            # Save residual features (not corrected channels)
            features_np_source = features_src[-1].numpy()
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
                
            features_np_target = features_tgt[-1].numpy()
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
                
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += residual_reg.numpy() * x_src.shape[0]  # Track residual magnitude
        
    # end batch loop
    if save_features and (domain_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
    
    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train
    avg_residual_norm = epoc_residual_norm / N_train  # Average residual magnitude
    
    # Print residual statistics for monitoring
    print(f"    Residual norm (avg): {avg_residual_norm:.6f}")
    
    # Return compatible output structure (same as original)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd,  # JMMD loss on residual features
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=features_src[-1] if features_src else None,
        film_features_source=features_src[-1] if features_src else None,
        avg_epoc_loss_d=avg_loss_d
    )
    
def val_step_wgan_gp_jmmd_residual(model, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                                linear_interp=False, return_H_gen=False):
    """
    Validation step for WGAN-GP with JMMD using residual learning
    """
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight')
    
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLossNormalized()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_jmmd_loss = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude during validation
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # --- Get data (same as original) ---
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocess (same as original)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model.generator(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # ← Apply residual correction
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model.generator(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # ← Apply residual correction
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # Track residual magnitudes during validation
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === WGAN Discriminator Scores (on corrected channels) ===
        d_real_src = model.discriminator(y_scaled_src, training=False)
        d_fake_src = model.discriminator(preds_src, training=False)  # ← Discriminate corrected channels
        gp_src = gradient_penalty(model.discriminator, y_scaled_src, preds_src, batch_size=x_scaled_src.shape[0])
        lambda_gp = 10.0
        
        d_loss_src = tf.reduce_mean(d_fake_src) - tf.reduce_mean(d_real_src) + lambda_gp * gp_src
        epoc_gan_disc_loss += d_loss_src.numpy() * x_src.shape[0]

        # === JMMD Loss (on residual features) ===
        if domain_weight > 0:
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # === Smoothness loss (on corrected channels) ===
        if temporal_weight != 0 or frequency_weight != 0:
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

        # === Save H samples (same structure as original) ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages (same as original)
    N_val = N_val_source + N_val_target
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_gan_disc_loss = epoc_gan_disc_loss / N_val_source 
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Print residual statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    
    # Same domain accuracy placeholders as original
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss (same structure as original)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss \
                     + domain_weight * avg_jmmd_loss + avg_smoothness_loss

    # Return same structure as original
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': avg_gan_disc_loss,
        'avg_domain_loss': avg_jmmd_loss,
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

def train_step_cnn_residual_jmmd(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                            save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    CNN-only residual training step with JMMD domain adaptation (no discriminator)
    Model predicts residual correction instead of direct channel estimation
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - only first one used
        optimizer: single optimizer for CNN (not tuple like GAN)
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for PAD computation
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss (no BCE for discriminator)
    
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    # Initialize JMMD for domain adaptation
    jmmd_loss_fn = JMMDLossNormalized()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_jmmd = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude
    N_train = 0
    
    if save_features==True and (domain_weight != 0):
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get and preprocess data (same as before)
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocessing (same as original)
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        # === Train CNN with RESIDUAL learning + JMMD ===
        with tf.GradientTape() as tape:
            # RESIDUAL LEARNING: Predict corrections
            residual_src, features_src = model_cnn(x_scaled_src, training=True, return_features=True)
            x_corrected_src = x_scaled_src + residual_src  # Apply residual correction
            
            residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=True, return_features=True)
            x_corrected_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
            
            # Estimation loss: Corrected channels should match perfect
            est_loss = loss_fn_est(y_scaled_src, x_corrected_src)
            
            # Domain adaptation: Residual features should be similar
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            
            # Residual regularization: Encourage small, meaningful corrections
            residual_reg = 0.001 * (tf.reduce_mean(tf.square(residual_src)) + 
                                    tf.reduce_mean(tf.square(residual_tgt)))
            
            # Smoothness loss on corrected channels
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
            
            # Total loss (NO adversarial component)
            total_loss = (est_weight * est_loss + 
                        domain_weight * jmmd_loss + 
                        residual_reg +
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)
        
        # === Save features if required ===
        if save_features and (domain_weight != 0):
            # Save residual features
            features_np_source = features_src[-1].numpy()
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
                
            features_np_target = features_tgt[-1].numpy()
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
        
        # Single optimizer update (no discriminator)
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_src)).numpy() * x_src.shape[0]
    
    # end batch loop
    if save_features and (domain_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    
    # Print residual statistics
    print(f"    Residual norm (avg): {avg_residual_norm:.6f}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd,
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=0.0,  # Can't calculate without target labels
        features_source=features_src[-1] if features_src else None,
        film_features_source=features_src[-1] if features_src else None,
        avg_epoc_loss_d=0.0  # No discriminator loss
    )

def val_step_cnn_residual_jmmd(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                            linear_interp=False, return_H_gen=False):
    """
    Validation step for CNN-only residual learning with JMMD (no discriminator)
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of validation loaders
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    # Initialize JMMD loss
    jmmd_loss_fn = JMMDLossNormalized()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_jmmd_loss = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude during validation
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # Get and preprocess data (same as original)
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocessing (same as original)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
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

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
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

        # Track residual magnitudes during validation
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === JMMD Loss (on residual features) ===
        if domain_weight > 0:
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # === Smoothness loss (on corrected channels) ===
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

        # === Save H samples ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Print residual statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    
    # Domain accuracy placeholders (compatible with existing code)
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss (no discriminator component)
    avg_total_loss = est_weight * avg_loss_est + domain_weight * avg_jmmd_loss + avg_smoothness_loss

    # Return compatible structure
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_domain_loss': avg_jmmd_loss,
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

def train_step_cnn_residual_source_only(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                    save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    CNN-only residual training step using only source domain (no domain adaptation)
    Model predicts residual correction for source domain only
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt) - tgt used only for monitoring
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - only first one used
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for PAD computation
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0  # For monitoring target performance
    epoc_residual_norm = 0.0  # Track residual magnitude
    N_train = 0
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
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

        # === Train CNN with RESIDUAL learning - Source only ===
        with tf.GradientTape() as tape:
            # RESIDUAL LEARNING: Predict correction for source domain
            residual_src, features_src = model_cnn(x_scaled_src, training=True, return_features=True)
            x_corrected_src = x_scaled_src + residual_src  # Apply residual correction
            
            # Estimation loss: Corrected source channels should match perfect source
            est_loss = loss_fn_est(y_scaled_src, x_corrected_src)
            
            # Residual regularization: Encourage small, meaningful corrections
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_src))
            
            # Smoothness loss on corrected source channels (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss = compute_total_smoothness_loss(x_corrected_src, 
                                                            temporal_weight=temporal_weight, 
                                                            frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # Total loss (no domain adaptation)
            total_loss = est_weight * est_loss + residual_reg + smoothness_loss
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)
        
        # Single optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_src)).numpy() * x_src.shape[0]

        # === Optional: Monitor target performance (no training) ===
        if batch_idx < loader_H_true_train_tgt.total_batches:
            try:
                x_tgt = loader_H_input_train_tgt.next_batch()
                y_tgt = loader_H_true_train_tgt.next_batch()
                
                # Preprocess target data
                x_tgt = complx2real(x_tgt)
                y_tgt = complx2real(y_tgt)
                x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
                y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
                x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
                y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
                
                # Test on target (no gradients) - residual learning
                residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
                x_corrected_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
                est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
                epoc_loss_est_tgt += est_loss_tgt.numpy() * x_tgt.shape[0]
                
            except StopIteration:
                # Target loader exhausted - skip monitoring for remaining batches
                pass
        
        # === Save features if required ===
        if save_features:
            # Save source features
            features_np_source = features_src[-1].numpy()
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
            
            # Save target features if available (for monitoring)
            if batch_idx < loader_H_true_train_tgt.total_batches and 'features_tgt' in locals():
                features_np_target = features_tgt[-1].numpy()
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

    # Close feature files
    if save_features:    
        features_h5_source.close()
        if features_dataset_target is not None:
            features_h5_target.close()

    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train if epoc_loss_est_tgt > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_train
    
    # Print residual statistics
    print(f"    Source residual norm (avg): {avg_residual_norm:.6f}")
    
    # Return compatible structure (no domain adaptation)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,  # No domain adaptation
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=features_src[-1] if 'features_src' in locals() else None,
        film_features_source=features_src[-1] if 'features_src' in locals() else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )
    
def val_step_cnn_residual_source_only(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, 
                                    weights=None, linear_interp=False, return_H_gen=False):
    """
    Validation step for CNN-only residual source training.
    Validates on source domain, tests on target domain.
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of validation loaders
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0  # This is now testing on target
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0  # This is now testing on target
    epoc_residual_norm = 0.0    # Track residual magnitude
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    # --- Source domain validation ---
    for idx in range(loader_H_true_val_source.total_batches):
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

        # === RESIDUAL LEARNING: Source validation prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
        if hasattr(preds_src, 'numpy'):
            preds_src_numpy = preds_src.numpy()
        else:
            preds_src_numpy = preds_src
            
        preds_src_descaled = deMinMax(preds_src_numpy, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        
        # Source NMSE
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # Track source residual magnitude
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        epoc_residual_norm += residual_src_norm * x_src.shape[0]

        # Save source samples from first batch
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0])
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

        if return_H_gen:
            H_gen_src_batch = preds_src_descaled.numpy().copy() if hasattr(preds_src_descaled, 'numpy') else preds_src_descaled.copy()
            all_H_gen_src.append(H_gen_src_batch)

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

        # === RESIDUAL LEARNING: Target testing prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
        if hasattr(preds_tgt, 'numpy'):
            preds_tgt_numpy = preds_tgt.numpy()
        else:
            preds_tgt_numpy = preds_tgt
            
        preds_tgt_descaled = deMinMax(preds_tgt_numpy, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        
        # Target NMSE
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # Save target samples from first batch
        if idx == 0:
            n_samples = min(3, x_tgt_real.shape[0])
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
            
            # Combine source and target samples (same format as GAN version)
            H_sample = [H_true_sample, H_input_sample, H_est_sample, nmse_input_source, nmse_est_source,
                    H_true_sample_target, H_input_sample_target, H_est_sample_target, nmse_input_target, nmse_est_target]

        if return_H_gen:
            H_gen_tgt_batch = preds_tgt_descaled.numpy().copy() if hasattr(preds_tgt_descaled, 'numpy') else preds_tgt_descaled.copy()
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
    avg_residual_norm = epoc_residual_norm / N_val_source

    # Print residual statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")

    # Total loss (source validation only)
    avg_total_loss = est_weight * avg_loss_est_source

    # Return compatible structure
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target,  # This is testing loss
        'avg_loss_est': avg_loss_est_source,  # Use source for validation
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_jmmd_loss': 0.0,  # No domain adaptation
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,  # This is testing NMSE
        'avg_nmse': avg_nmse_source,  # Use source for validation
        'avg_domain_acc_source': 0.5,  # Neutral placeholder
        'avg_domain_acc_target': 0.5,  # Neutral placeholder
        'avg_domain_acc': 0.5,         # Neutral placeholder
        'avg_smoothness_loss': 0.0
    }
    
    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return

# CORAL instead of JMMD
class GlobalPoolingCORALLoss(keras.layers.Layer):
    """
    Pure Global Average Pooling CORAL Loss
    Uses 100% global pooling for maximum memory efficiency and channel-wise statistics
    """
    def __init__(self, **kwargs):
        super(GlobalPoolingCORALLoss, self).__init__(**kwargs)
        print(f"GlobalPoolingCORALLoss initialized: 100% Global Average Pooling")
    
    def compute_covariance(self, features):
        """
        Compute covariance matrix of features
        Same as your original implementation
        """
        # Center the features (subtract mean)
        features_centered = features - tf.reduce_mean(features, axis=0, keepdims=True)
        
        # Compute covariance matrix
        n = tf.cast(tf.shape(features_centered)[0], tf.float32)
        cov_matrix = tf.matmul(features_centered, features_centered, transpose_a=True) / (n - 1)
        
        return cov_matrix
    
    def coral_loss(self, source_features, target_features):
        """
        Compute CORAL loss between source and target features
        """
        # Compute covariance matrices
        source_cov = self.compute_covariance(source_features)
        target_cov = self.compute_covariance(target_features)
        
        # CORAL loss: Frobenius norm of covariance difference
        loss = tf.reduce_sum(tf.square(source_cov - target_cov))
        
        # Normalize by feature dimension squared
        d = tf.cast(source_features.shape[1], tf.float32)
        loss = loss / (4.0 * d * d)
        
        return loss
    
    def call(self, source_list, target_list):
        """
        Compute CORAL loss across multiple layers using 100% global pooling
        
        Args:
            source_list: list of source features from different layers
            target_list: list of target features from different layers
        """
        coral_loss_total = 0.0
        
        for i, (source_feat, target_feat) in enumerate(zip(source_list, target_list)):
            # Apply Global Average Pooling for ALL feature types
            if len(source_feat.shape) == 4:  # [B, H, W, C]
                # Global Average Pooling: [B, H, W, C] → [B, C]
                source_pooled = tf.reduce_mean(source_feat, axis=[1, 2])
                target_pooled = tf.reduce_mean(target_feat, axis=[1, 2])
            elif len(source_feat.shape) == 3:  # [B, H, C] or [B, W, C]
                # Global Average Pooling: [B, H, C] → [B, C]
                source_pooled = tf.reduce_mean(source_feat, axis=1)
                target_pooled = tf.reduce_mean(target_feat, axis=1)
            elif len(source_feat.shape) == 2:  # [B, C] - already pooled
                source_pooled = source_feat
                target_pooled = target_feat
            else:
                # Flatten any other shapes and then global pool
                source_flat = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
                target_flat = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
                # For flattened features, just use as-is (treat as [B, Features])
                source_pooled = source_flat
                target_pooled = target_flat
            
            # Compute CORAL loss for this layer
            layer_coral_loss = self.coral_loss(source_pooled, target_pooled)
            coral_loss_total += layer_coral_loss
        
        return coral_loss_total / len(source_list)  # Average across layers


class HybridCORALLoss(keras.layers.Layer):
    """
    Hybrid CORAL Loss with Hybrid Feature Reduction
    Combines global pooling + dimension reduction for optimal performance/memory trade-off
    """
    def __init__(self, max_features=1024, use_global_pooling=True, 
                 global_pooling_weight=0.7, dense_reduction_weight=0.3, **kwargs):
        super(HybridCORALLoss, self).__init__(**kwargs)
        self.max_features = max_features
        self.use_global_pooling = use_global_pooling
        self.gp_weight = global_pooling_weight      # Weight for global pooling branch
        self.dr_weight = dense_reduction_weight     # Weight for dense reduction branch
        
        print(f"HybridCORALLoss initialized:")
        print(f"  - Max features: {max_features}")
        print(f"  - Global pooling: {use_global_pooling}")
        print(f"  - GP weight: {global_pooling_weight}, DR weight: {dense_reduction_weight}")
    
    def reduce_feature_dims(self, features, target_dim=None):
        """Reduce feature dimensions using random projection"""
        if target_dim is None:
            target_dim = self.max_features
            
        if features.shape[-1] > target_dim:
            # Random projection matrix with proper scaling
            proj_matrix = tf.random.normal([features.shape[-1], target_dim], 
                                        stddev=1.0/tf.sqrt(float(target_dim)))
            features = tf.matmul(features, proj_matrix)
        return features
    
    def coral_loss(self, source_features, target_features):
        """
        Compute CORAL loss between source and target features
        Same as your original implementation
        """
        # Compute covariance matrices
        source_cov = self.compute_covariance(source_features)
        target_cov = self.compute_covariance(target_features)
        
        # CORAL loss: Frobenius norm of covariance difference
        loss = tf.reduce_sum(tf.square(source_cov - target_cov))
        
        # Normalize by feature dimension squared
        d = tf.cast(source_features.shape[1], tf.float32)
        loss = loss / (4.0 * d * d)
        
        return loss
    
    def compute_covariance(self, features):
        """
        Compute covariance matrix of features
        Same as your original implementation
        """
        # Center the features (subtract mean)
        features_centered = features - tf.reduce_mean(features, axis=0, keepdims=True)
        
        # Compute covariance matrix
        n = tf.cast(tf.shape(features_centered)[0], tf.float32)
        cov_matrix = tf.matmul(features_centered, features_centered, transpose_a=True) / (n - 1)
        
        return cov_matrix
    
    def hybrid_coral_loss(self, source_feat, target_feat):
        """
        Compute CORAL loss using hybrid approach: global pooling + dense reduction
        """
        total_coral_loss = 0.0
        num_branches = 0
        
        # === Branch 1: Global Pooling (Channel Statistics) ===
        if self.use_global_pooling and len(source_feat.shape) == 4:  # [B, H, W, C]
            src_pooled = tf.reduce_mean(source_feat, axis=[1, 2])  # [B, C]
            tgt_pooled = tf.reduce_mean(target_feat, axis=[1, 2])  # [B, C]
            
            # Further reduce dimensions if needed
            if src_pooled.shape[-1] > self.max_features:
                src_pooled = self.reduce_feature_dims(src_pooled, self.max_features)
                tgt_pooled = self.reduce_feature_dims(tgt_pooled, self.max_features)
            
            coral_pooled = self.coral_loss(src_pooled, tgt_pooled)
            total_coral_loss += self.gp_weight * coral_pooled
            num_branches += 1
            
            # print(f"    Global pooling branch: {source_feat.shape} → {src_pooled.shape}")
        
        # === Branch 2: Dense Reduction (Spatial Relationships) ===
        if self.dr_weight > 0:
            # Calculate potential flattened size
            if len(source_feat.shape) == 4:
                h, w, c = source_feat.shape[1], source_feat.shape[2], source_feat.shape[3]
                potential_flat_size = h * w * c
                
                # SMART POOLING: Apply light pooling if too large
                if potential_flat_size > self.max_features * 2:  # Threshold: 2x max_features
                    # Apply light spatial pooling (2x2 or 3x3 depending on size)
                    if h > 64 or w > 64:
                        # Large spatial dimensions - use 3x3 pooling
                        pool_size = (3, 3)
                        src_pooled_spatial = tf.nn.avg_pool2d(source_feat, pool_size, strides=pool_size, padding='SAME')
                        tgt_pooled_spatial = tf.nn.avg_pool2d(target_feat, pool_size, strides=pool_size, padding='SAME')
                    else:
                        # Moderate spatial dimensions - use 2x2 pooling
                        pool_size = (2, 2)
                        src_pooled_spatial = tf.nn.avg_pool2d(source_feat, pool_size, strides=pool_size, padding='SAME')
                        tgt_pooled_spatial = tf.nn.avg_pool2d(target_feat, pool_size, strides=pool_size, padding='SAME')
                    
                    # print(f"    Applied light pooling {pool_size}: {source_feat.shape} → {src_pooled_spatial.shape}")
                    
                    # Now flatten the pooled features
                    src_flat = tf.reshape(src_pooled_spatial, [tf.shape(src_pooled_spatial)[0], -1])
                    tgt_flat = tf.reshape(tgt_pooled_spatial, [tf.shape(tgt_pooled_spatial)[0], -1])
                else:
                    # Small enough - flatten directly
                    src_flat = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
                    tgt_flat = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
            else:
                # Already flattened or 2D
                src_flat = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
                tgt_flat = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
        
            # Apply dimension reduction
            if src_flat.shape[-1] > self.max_features:
                src_reduced = self.reduce_feature_dims(src_flat, self.max_features)
                tgt_reduced = self.reduce_feature_dims(tgt_flat, self.max_features)
            else:
                src_reduced, tgt_reduced = src_flat, tgt_flat
            
            coral_dense = self.coral_loss(src_reduced, tgt_reduced)
            total_coral_loss += self.dr_weight * coral_dense
            num_branches += 1
            
            # print(f"    Dense reduction branch: {source_feat.shape} → {src_reduced.shape}")
        
        # Return weighted combination
        return total_coral_loss if num_branches > 0 else 0.0
    
    def call(self, source_list, target_list):
        """
        Compute CORAL loss across multiple layers with hybrid feature reduction
        
        Args:
            source_list: list of source features from different layers
            target_list: list of target features from different layers
        """
        coral_loss_total = 0.0
        
        for i, (source_feat, target_feat) in enumerate(zip(source_list, target_list)):
            # print(f"  Processing layer {i+1}: {source_feat.shape}")
            
            # Use hybrid approach for each layer
            layer_coral_loss = self.hybrid_coral_loss(source_feat, target_feat)
            coral_loss_total += layer_coral_loss
        
        return coral_loss_total / len(source_list)  # Average across layers

def train_step_wgan_gp_coral_residual(model, loader_H, loss_fn, optimizers, lower_range=-1, 
                        coral_loss_fn=None, save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    WGAN-GP training step with CORAL domain adaptation using residual learning approach
    Model predicts residual correction instead of direct channel estimation
    
    Args:
        model: GAN model instance with generator and discriminator
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss)
        optimizers: tuple of (gen_optimizer, disc_optimizer)
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for PAD computation
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]  # Only need first two loss functions
    gen_optimizer, disc_optimizer = optimizers[:2]  # No domain optimizer needed
    
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight')
    
    # Initialize CORAL loss in case not provided (should be initialized outside)
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    
    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_coral = 0.0  # Track CORAL loss
    epoc_residual_norm = 0.0  # Track residual magnitude
    N_train = 0
    
    if save_features==True and (domain_weight != 0):
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # --- Get data (same as original) ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess (source) - same as original
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target) - same as original
        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === 1. Train Discriminator (WGAN-GP) with RESIDUAL learning ===
        with tf.GradientTape() as tape_d:
            # RESIDUAL LEARNING: Generate residual correction
            residual_src, _ = model.generator(x_scaled_src, training=True)
            x_fake_src = x_scaled_src + residual_src  # ← KEY CHANGE: Add residual to input
            
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)  # ← Discriminate corrected channels
            
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

        # === 2. Train Generator with RESIDUAL learning + CORAL ===
        with tf.GradientTape() as tape_g:
            # RESIDUAL LEARNING: Generate residual corrections with features
            residual_src, features_src = model.generator(x_scaled_src, training=True)
            x_fake_src = x_scaled_src + residual_src  # ← Apply residual correction
            
            residual_tgt, features_tgt = model.generator(x_scaled_tgt, training=True)
            x_fake_tgt = x_scaled_tgt + residual_tgt  # ← Apply residual correction
            
            d_fake_src = model.discriminator(x_fake_src, training=False)
            
            # Generator losses on CORRECTED channels
            g_adv_loss = -tf.reduce_mean(d_fake_src)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)          # ← Loss on corrected source
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)      # ← Loss on corrected target (monitoring)
            
            # CORAL loss between residual features (encourages similar correction patterns)
            coral_loss = coral_loss_fn(features_src, features_tgt)
            
            # RESIDUAL REGULARIZATION: Encourage small, meaningful corrections
            residual_reg_src = tf.reduce_mean(tf.square(residual_src))
            residual_reg_tgt = tf.reduce_mean(tf.square(residual_tgt))
            residual_reg = (residual_reg_src + residual_reg_tgt) / 2
            residual_penalty = 0.001 * residual_reg  # Small penalty for large residuals
            
            # ADD TEMPORAL SMOOTHNESS LOSS on corrected channels
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_loss_src = compute_total_smoothness_loss(x_fake_src, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss_tgt = compute_total_smoothness_loss(x_fake_tgt, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            else:
                smoothness_loss = 0.0
        
            # Total generator loss with residual regularization
            g_loss = (est_weight * g_est_loss + 
                    adv_weight * g_adv_loss + 
                    domain_weight * coral_loss +   # ← CORAL instead of JMMD
                    residual_penalty +              # ← Encourage small residuals
                    smoothness_loss)
            
            # Add L2 regularization
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        
        # === 3. Save features if required (same as original) ===
        if save_features and (domain_weight != 0):
            # Save residual features (not corrected channels)
            features_np_source = features_src[-1].numpy()
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
                
            features_np_target = features_tgt[-1].numpy()
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
                
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]
        epoc_loss_coral += coral_loss.numpy() * x_src.shape[0]  # Track CORAL instead of JMMD
        epoc_residual_norm += residual_reg.numpy() * x_src.shape[0]  # Track residual magnitude
        
    # end batch loop
    if save_features and (domain_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
    
    # Average losses
    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_coral = epoc_loss_coral / N_train  # CORAL loss average
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train
    avg_residual_norm = epoc_residual_norm / N_train  # Average residual magnitude
    
    # Print residual statistics for monitoring
    print(f"    Residual norm (avg): {avg_residual_norm:.6f}")
    
    # Return compatible output structure (same as JMMD residual version)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_coral,  # CORAL loss on residual features
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=features_src[-1] if features_src else None,
        film_features_source=features_src[-1] if features_src else None,
        avg_epoc_loss_d=avg_loss_d
    )

def val_step_wgan_gp_coral_residual(model, loader_H, loss_fn, lower_range, coral_loss_fn=None, nsymb=14, weights=None, 
                                    linear_interp=False, return_H_gen=False):
    """
    Validation step for WGAN-GP with CORAL using residual learning
    """
    adv_weight = weights.get('adv_weight', 0.01)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight')
    
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce = loss_fn[:2]
    
    # Initialize CORAL loss in case not provided (should be initialized outside)
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_coral_loss = 0.0  # Track CORAL instead of JMMD
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude during validation
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # --- Get data (same as original) ---
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocess (same as original)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model.generator(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # ← Apply residual correction
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model.generator(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # ← Apply residual correction
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # Track residual magnitudes during validation
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === WGAN Discriminator Scores (on corrected channels) ===
        d_real_src = model.discriminator(y_scaled_src, training=False)
        d_fake_src = model.discriminator(preds_src, training=False)  # ← Discriminate corrected channels
        gp_src = gradient_penalty(model.discriminator, y_scaled_src, preds_src, batch_size=x_scaled_src.shape[0])
        lambda_gp = 10.0
        
        d_loss_src = tf.reduce_mean(d_fake_src) - tf.reduce_mean(d_real_src) + lambda_gp * gp_src
        epoc_gan_disc_loss += d_loss_src.numpy() * x_src.shape[0]

        # === CORAL Loss (on residual features) ===
        if domain_weight > 0:
            coral_loss = coral_loss_fn(features_src, features_tgt)
            epoc_coral_loss += coral_loss.numpy() * x_src.shape[0]
        
        # === Smoothness loss (on corrected channels) ===
        if temporal_weight != 0 or frequency_weight != 0:
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

        # === Save H samples (same structure as original) ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages (same as original)
    N_val = N_val_source + N_val_target
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_gan_disc_loss = epoc_gan_disc_loss / N_val_source 
    avg_coral_loss = epoc_coral_loss / N_val_source if epoc_coral_loss > 0 else 0.0  # CORAL average
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Print residual statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    
    # Same domain accuracy placeholders as original
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss (same structure as original)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss \
                     + domain_weight * avg_coral_loss + avg_smoothness_loss

    # Return same structure as original, replace JMMD with CORAL
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': avg_gan_disc_loss,
        'avg_domain_loss': avg_coral_loss,  # ← Use CORAL loss for compatibility
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

def train_step_cnn_residual_coral(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                        coral_loss_fn=None,save_features=False, nsymb=14, weights=None, linear_interp=False):
    """
    CNN-only residual training step with CORAL domain adaptation (no discriminator)
    Model predicts residual correction instead of direct channel estimation
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - only first one used
        optimizer: single optimizer for CNN (not tuple like GAN)
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for PAD computation
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss (no BCE for discriminator)
    
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    # Initialize CORAL in case not provided (should be initialized outside)
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_coral = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude
    N_train = 0
    
    if save_features==True and (domain_weight != 0):
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    epoc_loss_est_tgt = 0.0
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get and preprocess data (same as JMMD version)
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocessing (same as original)
        x_src = complx2real(x_src)
        y_src = complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt = complx2real(x_tgt)
        y_tgt = complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        # === Train CNN with RESIDUAL learning + CORAL ===
        with tf.GradientTape() as tape:
            # RESIDUAL LEARNING: Predict corrections
            residual_src, features_src = model_cnn(x_scaled_src, training=True, return_features=True)
            x_corrected_src = x_scaled_src + residual_src  # Apply residual correction
            
            residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=True, return_features=True)
            x_corrected_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
            
            # Estimation loss: Corrected channels should match perfect
            est_loss = loss_fn_est(y_scaled_src, x_corrected_src)
            
            # Domain adaptation: CORAL loss on residual features
            coral_loss = coral_loss_fn(features_src, features_tgt)
            
            # Residual regularization: Encourage small, meaningful corrections
            residual_reg = 0.001 * (tf.reduce_mean(tf.square(residual_src)) + 
                                    tf.reduce_mean(tf.square(residual_tgt)))
            
            # Smoothness loss on corrected channels
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
            
            # Total loss (NO adversarial component)
            total_loss = (est_weight * est_loss + 
                        domain_weight * coral_loss +  # ← CORAL instead of JMMD
                        residual_reg +
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)
        
        # Calculate target estimation loss AFTER gradient update (no training impact)
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_tgt += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        
        # === Save features if required ===
        if save_features and (domain_weight != 0):
            # Save residual features
            features_np_source = features_src[-1].numpy()
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
                
            features_np_target = features_tgt[-1].numpy()
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
        
        # Single optimizer update (no discriminator)
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_coral += coral_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_src)).numpy() * x_src.shape[0]        
    
    # end batch loop
    if save_features and (domain_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_coral = epoc_loss_coral / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train
    
    # Print residual statistics
    print(f"    Residual norm (avg): {avg_residual_norm:.6f}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_coral,  # CORAL loss 
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_tgt,  
        features_source=features_src[-1] if features_src else None,
        film_features_source=features_src[-1] if features_src else None,
        avg_epoc_loss_d=0.0  # No discriminator loss
    )
    
def val_step_cnn_residual_coral(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                                coral_loss_fn=None, linear_interp=False, return_H_gen=False):
    """
    Validation step for CNN-only residual learning with CORAL (no discriminator)
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of validation loaders
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0)
    domain_weight = weights.get('domain_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    # Initialize CORAL loss in case not provided (should be initialized outside)
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_coral_loss = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude during validation
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # Get and preprocess data (same as JMMD version)
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocessing (same as original)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
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

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
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

        # Track residual magnitudes during validation
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === CORAL Loss (on residual features) ===
        if domain_weight > 0:
            coral_loss = coral_loss_fn(features_src, features_tgt)
            epoc_coral_loss += coral_loss.numpy() * x_src.shape[0]
        
        # === Smoothness loss (on corrected channels) ===
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

        # === Save H samples ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_coral_loss = epoc_coral_loss / N_val_source if epoc_coral_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Print residual statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    
    # Domain accuracy placeholders (compatible with existing code)
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss (no discriminator component)
    avg_total_loss = est_weight * avg_loss_est + domain_weight * avg_coral_loss + avg_smoothness_loss

    # Return compatible structure
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_domain_loss': avg_coral_loss,  # Use CORAL for compatibility with plotting
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


#
##
### Fourier Domain Adaptation

class Fourier_DA(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Minimal placeholder layer to make it a valid model
        self.placeholder = tf.keras.layers.Identity()
    
    def call(self, x, training=False):
        # For now, just return input unchanged
        return self.placeholder(x)

def F_extract_DD(channel_grid):
    """
    Converts Time-Freq channel to Delay-Doppler domain
    
    Args:
        channel_grid: Complex tensor [batch, 132, 14] or [132, 14]
                    132 subcarriers x 14 OFDM symbols
    
    Returns:
        amplitude_spectrum: Magnitude in Delay-Doppler domain (The "Style") 
        phase_spectrum: Phase in Delay-Doppler domain (The "Content")
    """
    
    # Ensure we have complex dtype
    channel_grid = tf.cast(channel_grid, tf.complex64)
    
    # 1. Transform Subcarriers (Frequency) -> Delay
    # IFFT along dimension -2 (subcarriers dimension)
    grid_delay = tf.signal.ifft(channel_grid)
    
    # 2. Transform Symbols (Time) -> Doppler  
    # FFT along dimension -1 (OFDM symbols dimension)
    grid_delay_doppler_raw = tf.signal.fft(grid_delay)
    
    # 3. Shift BOTH dimensions to center
    # Move (Delay=0, Doppler=0) to middle of matrix
    grid_dd_shifted = tf.signal.fftshift(grid_delay_doppler_raw, axes=[-2, -1])
    
    # 4. Extract amplitude and phase
    amplitude_spectrum = tf.abs(grid_dd_shifted)
    phase_spectrum = tf.math.angle(grid_dd_shifted)
    
    return amplitude_spectrum, phase_spectrum


def F_inverse_DD(complex_dd):
    """
    Converts Delay-Doppler grid back to Time-Freq grid
    
    Args:
        complex_dd: Complex tensor in Delay-Doppler domain [batch, 132, 14] or [132, 14]
    
    Returns:
        tf_grid: Complex tensor in Time-Frequency domain
    """
    
    # Ensure complex dtype
    complex_dd = tf.cast(complex_dd, tf.complex64)
    
    # 1. Unshift both dimensions
    # Move zero-delay/zero-doppler from center back to (0,0)
    grid_unshifted = tf.signal.ifftshift(complex_dd, axes=[-2, -1])
    
    # 2. Inverse Doppler -> Time (last dimension)
    # Forward was FFT, so inverse is IFFT
    grid_delay_time = tf.signal.ifft(grid_unshifted)
    
    # 3. Inverse Delay -> Frequency (second-to-last dimension)  
    # Forward was IFFT, so inverse is FFT
    tf_grid = tf.signal.fft(grid_delay_time)
    
    return tf_grid

def fda_mix_pixels(source_img, target_img, win_h_px, win_w_px):
    """
    Simplified version using tf.where for easier understanding
    """
    
    # Handle dimensions
    original_shape = tf.shape(source_img)
    if len(source_img.shape) == 2:
        source_img = tf.expand_dims(source_img, 0)
        target_img = tf.expand_dims(target_img, 0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, h, w = tf.unstack(tf.shape(source_img))
    
    # Calculate center and radius
    cy = h // 2
    cx = w // 2
    r_h = win_h_px // 2
    r_w = win_w_px // 2
    
    # Create coordinate grids
    y_coords = tf.range(h)
    x_coords = tf.range(w)
    y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Create mask condition
    y_condition = tf.logical_and(y_grid >= cy - r_h, y_grid <= cy + r_h)
    x_condition = tf.logical_and(x_grid >= cx - r_w, x_grid <= cx + r_w)
    mask_condition = tf.logical_and(y_condition, x_condition)
    
    # Expand for batch dimension
    mask_condition = tf.expand_dims(mask_condition, 0)
    mask_condition = tf.tile(mask_condition, [batch_size, 1, 1])
    
    # Mix images using tf.where
    mixed_img = tf.where(mask_condition, target_img, source_img)
    
    # Remove batch dimension if needed
    if squeeze_output:
        mixed_img = tf.squeeze(mixed_img, 0)
    
    return mixed_img

def apply_phase_to_amplitude(mixed_amplitude, phase_spectrum):
    """Apply phase to amplitude: amplitude * exp(j * phase)"""
    # Convert inputs to float32
    amplitude = tf.cast(mixed_amplitude, tf.float32)
    phase = tf.cast(phase_spectrum, tf.float32)
    
    # Create complex tensor using TensorFlow's complex function
    # tf.complex(real, imag) where real=amp*cos(phase), imag=amp*sin(phase)
    real_part = amplitude * tf.cos(phase)
    imag_part = amplitude * tf.sin(phase)
    
    return tf.complex(real_part, imag_part)

def train_step_cnn_residual_fda(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                save_features=False, nsymb=14, weights=None, linear_interp=False,
                                fda_win_h=13, fda_win_w=3, fda_weight=1.0):
    """
    CNN-only residual training step with FDA input preprocessing
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_residual_norm = 0.0
    N_train = 0
    
    # Initialize feature variables
    features_src = None
    features_tgt = None
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess source data
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess target data
        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # ============ FDA PREPROCESSING ============
        # Convert back to complex for FDA
        x_src_complex = tf.complex(x_scaled_src[:,:,:,0], x_scaled_src[:,:,:,1])
        x_tgt_complex = tf.complex(x_scaled_tgt[:,:,:,0], x_scaled_tgt[:,:,:,1])
        
        # Extract Delay-Doppler representations
        src_amplitude, src_phase = F_extract_DD(x_src_complex)
        tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)
        
        # Mix amplitudes: Target style in center, Source style in outer regions
        mixed_amplitude = fda_mix_pixels(src_amplitude, tgt_amplitude, fda_win_h, fda_win_w)
        
        # Keep source phase (content preservation)
        mixed_complex_dd = apply_phase_to_amplitude(mixed_amplitude, src_phase)
        
        # Convert back to Time-Frequency domain
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd)
        
        # 
        x_fda_mixed = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        # === Train CNN with RESIDUAL learning ===
        with tf.GradientTape() as tape:
            # 
            batch_size = x_scaled_src.shape[0]  # Use .shape instead of tf.shape
            
            if fda_weight < 1.0:
                # Dual training: Part source, part FDA-mixed
                fda_samples = int(batch_size * fda_weight)
                remaining_samples = batch_size - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    # Split batch
                    x_train_fda = x_fda_mixed[:fda_samples]
                    x_train_src = x_scaled_src[fda_samples:]
                    y_train_combined = y_scaled_src  # All use source labels
                    
                    # Forward pass on FDA-mixed samples
                    residual_fda, features_fda = model_cnn(x_train_fda, training=True, return_features=True)
                    x_corrected_fda = x_train_fda + residual_fda
                    
                    # Forward pass on pure source samples
                    residual_src, features_src = model_cnn(x_train_src, training=True, return_features=True)
                    x_corrected_src = x_train_src + residual_src
                    
                    # Combine corrected outputs
                    x_corrected_combined = tf.concat([x_corrected_fda, x_corrected_src], axis=0)
                    residual_combined = tf.concat([residual_fda, residual_src], axis=0)
                    
                elif fda_samples == 0:
                    # Pure source training
                    residual_src, features_src = model_cnn(x_scaled_src, training=True, return_features=True)
                    x_corrected_combined = x_scaled_src + residual_src
                    residual_combined = residual_src
                else:
                    # Pure FDA training
                    residual_fda, features_src = model_cnn(x_fda_mixed, training=True, return_features=True)
                    x_corrected_combined = x_fda_mixed + residual_fda
                    residual_combined = residual_fda
                
                # Estimation loss
                est_loss = loss_fn_est(y_scaled_src, x_corrected_combined)
                    
            else:
                # Pure FDA training: Use only FDA-mixed input
                residual_fda, features_src = model_cnn(x_fda_mixed, training=True, return_features=True)
                x_corrected_fda = x_fda_mixed + residual_fda
                residual_combined = residual_fda
                
                # Estimation loss: FDA-corrected input vs source labels
                est_loss = loss_fn_est(y_scaled_src, x_corrected_fda)
            
            # Residual regularization
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # Total loss
            total_loss = est_weight * est_loss + residual_reg + smoothness_loss
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)
        
        # Monitor target performance (no gradients)
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        x_corrected_tgt = x_scaled_tgt + residual_tgt
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === Save features if required ===
        if save_features and features_src is not None:
            # Save features
            features_np_source = features_src[-1].numpy()
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
                
            if features_tgt is not None:
                features_np_target = features_tgt[-1].numpy()
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
        
        # Single optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features:    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    
    # Print FDA-specific statistics
    print(f"    FDA residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA mixing window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,  # No domain loss - FDA handles domain gap at input level
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_src[-1] if features_src is not None else None,
        film_features_source=features_src[-1] if features_src is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )
    
def val_step_cnn_residual_fda(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, 
                                weights=None, linear_interp=False, return_H_gen=False,
                                fda_win_h=13, fda_win_w=3, fda_weight=1.0):
    """
    Validate with separate source/target validation sets (Real deployment performance)
    
    Validation step for CNN-only residual learning with FDA input preprocessing
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of validation loaders
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
        fda_win_h: FDA window height for mixing (default 13)
        fda_win_w: FDA window width for mixing (default 3)
        fda_weight: Weight for FDA vs source training (0.5 = 50% FDA, 50% source)
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0)
    temporal_weight = weights.get('temporal_weight', 0.0)
    frequency_weight = weights.get('frequency_weight', 0.0)
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_residual_norm = 0.0
    epoc_smoothness_loss = 0.0
    epoc_fda_loss = 0.0  # Track FDA-mixed performance
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    # --- Source domain validation ---
    for idx in range(loader_H_true_val_source.total_batches):
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

        # === RESIDUAL LEARNING: Source validation prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
        if hasattr(preds_src, 'numpy'):
            preds_src_numpy = preds_src.numpy()
        else:
            preds_src_numpy = preds_src
            
        preds_src_descaled = deMinMax(preds_src_numpy, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        
        # Source NMSE
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # Track source residual magnitude
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        epoc_residual_norm += residual_src_norm * x_src.shape[0]

        # Save source samples from first batch
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0])
            H_true_sample_source = y_src_real[:n_samples].copy()
            H_input_sample_source = x_src_real[:n_samples].copy()
            if hasattr(preds_src_descaled, 'numpy'):
                H_est_sample_source = preds_src_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample_source = preds_src_descaled[:n_samples].copy()
            
            # Calculate source metrics for samples
            mse_sample_source = np.mean((H_est_sample_source - H_true_sample_source) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample_source ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample_source - H_true_sample_source) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)

        if return_H_gen:
            H_gen_src_batch = preds_src_descaled.numpy().copy() if hasattr(preds_src_descaled, 'numpy') else preds_src_descaled.copy()
            all_H_gen_src.append(H_gen_src_batch)

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

        # === RESIDUAL LEARNING: Target testing prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
        if hasattr(preds_tgt, 'numpy'):
            preds_tgt_numpy = preds_tgt.numpy()
        else:
            preds_tgt_numpy = preds_tgt
            
        preds_tgt_descaled = deMinMax(preds_tgt_numpy, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        
        # Target NMSE
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # ============ FDA VALIDATION TEST ============
        # Test FDA mixing during validation for monitoring
        if idx == 0:  # Only do FDA test on first batch for efficiency
            # Get corresponding source batch for FDA mixing
            try:
                x_src_for_fda = loader_H_input_val_source.get_batch(idx)
                y_src_for_fda = loader_H_true_val_source.get_batch(idx)
                
                # Preprocess source for FDA
                x_src_fda_real = complx2real(x_src_for_fda)
                y_src_fda_real = complx2real(y_src_for_fda)
                x_src_fda_real = np.transpose(x_src_fda_real, (0, 2, 3, 1))
                y_src_fda_real = np.transpose(y_src_fda_real, (0, 2, 3, 1))
                x_scaled_src_fda, x_min_src_fda, x_max_src_fda = minmaxScaler(x_src_fda_real, lower_range=lower_range, linear_interp=linear_interp)
                y_scaled_src_fda, _, _ = minmaxScaler(y_src_fda_real, min_pre=x_min_src_fda, max_pre=x_max_src_fda, lower_range=lower_range)
                
                # Apply FDA mixing (same as training)
                x_src_complex_fda = tf.complex(x_scaled_src_fda[:,:,:,0], x_scaled_src_fda[:,:,:,1])
                x_tgt_complex_fda = tf.complex(x_scaled_tgt[:,:,:,0], x_scaled_tgt[:,:,:,1])
                
                # Extract Delay-Doppler representations
                src_amplitude_fda, src_phase_fda = F_extract_DD(x_src_complex_fda)
                tgt_amplitude_fda, tgt_phase_fda = F_extract_DD(x_tgt_complex_fda)
                
                # Mix amplitudes: Target style in center, Source style in outer regions
                mixed_amplitude_fda = fda_mix_pixels(src_amplitude_fda, tgt_amplitude_fda, fda_win_h, fda_win_w)
                
                # Keep source phase (content preservation)
                mixed_complex_dd_fda = apply_phase_to_amplitude(mixed_amplitude_fda, src_phase_fda)
                
                # Convert back to Time-Frequency domain
                x_fda_mixed_complex_fda = F_inverse_DD(mixed_complex_dd_fda)
                
                # Convert back to real format for CNN [batch, 132, 14, 2]
                x_fda_mixed_fda = tf.stack([tf.math.real(x_fda_mixed_complex_fda), tf.math.imag(x_fda_mixed_complex_fda)], axis=-1)
                
                # Test FDA-mixed input performance
                residual_fda, _ = model_cnn(x_fda_mixed_fda, training=False, return_features=False)
                preds_fda = x_fda_mixed_fda + residual_fda
                fda_loss = loss_fn_est(y_scaled_src_fda, preds_fda)
                epoc_fda_loss += fda_loss.numpy() * x_tgt.shape[0]
                
            except (IndexError, AttributeError):
                # Handle case where source loader doesn't have enough batches or get_batch method
                print(f"Warning: Could not perform FDA validation test at batch {idx}")
                pass

        # Save target samples from first batch
        if idx == 0:
            n_samples = min(3, x_tgt_real.shape[0])
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            if hasattr(preds_tgt_descaled, 'numpy'):
                H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample_target = preds_tgt_descaled[:n_samples].copy()
            
            # Calculate target metrics for samples
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            
            # Combine source and target samples
            H_sample = [H_true_sample_source, H_input_sample_source, H_est_sample_source, 
                        nmse_input_source, nmse_est_source,
                        H_true_sample_target, H_input_sample_target, H_est_sample_target,
                        nmse_input_target, nmse_est_target]

        if return_H_gen:
            H_gen_tgt_batch = preds_tgt_descaled.numpy().copy() if hasattr(preds_tgt_descaled, 'numpy') else preds_tgt_descaled.copy()
            all_H_gen_tgt.append(H_gen_tgt_batch)

    # === Smoothness loss calculation ===
    if temporal_weight != 0 or frequency_weight != 0:
        # Calculate on last batch predictions
        preds_src_tensor = tf.convert_to_tensor(preds_src) if not tf.is_tensor(preds_src) else preds_src
        preds_tgt_tensor = tf.convert_to_tensor(preds_tgt) if not tf.is_tensor(preds_tgt) else preds_tgt
        
        smoothness_loss_src = compute_total_smoothness_loss(
            preds_src_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
        )
        smoothness_loss_tgt = compute_total_smoothness_loss(
            preds_tgt_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight
        )
        epoc_smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
    
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
    avg_residual_norm = epoc_residual_norm / N_val_source
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_fda_loss = epoc_fda_loss / N_val_target if epoc_fda_loss > 0 else 0.0

    # Print FDA-specific validation statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA validation loss: {avg_fda_loss:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")

    # Total loss (source validation only, no domain adaptation)
    avg_total_loss = est_weight * avg_loss_est_source + avg_smoothness_loss

    # Return compatible structure
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target,  # This is testing loss
        'avg_loss_est': avg_loss_est_source,  # Use source for validation
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_domain_loss': avg_fda_loss,  # Use FDA loss for compatibility with plotting
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,  # This is testing NMSE
        'avg_nmse': avg_nmse_source,  # Use source for validation
        'avg_domain_acc_source': 0.5,  # Neutral placeholder
        'avg_domain_acc_target': 0.5,  # Neutral placeholder
        'avg_domain_acc': 0.5,         # Neutral placeholder
        'avg_smoothness_loss': avg_smoothness_loss,
        # 'avg_fda_loss': avg_fda_loss   # FDA-specific metric
    }
    
    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return

def train_step_cnn_residual_fda_scaleCombined(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                save_features=False, nsymb=14, weights=None, linear_interp=False,
                                fda_win_h=13, fda_win_w=3, fda_weight=1.0):
    """
    CNN-only residual training step with FDA input preprocessing using combined scaling approach
    
    Workflow:
    1. Combined scaling: Scale source + target together with same min/max
    2. FDA operations: Apply Fourier Domain Adaptation 
    3. Post-FDA scaling: Scale FDA output back to [-1,1] for CNN compatibility
    4. Residual learning: Train CNN to predict corrections
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - only first one used
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for PAD computation
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height for mixing (default 13)
        fda_win_w: FDA window width for mixing (default 3)
        fda_weight: Weight for FDA vs source training (1.0 = 100% FDA, 0.8 = 80% FDA + 20% source)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_residual_norm = 0.0
    epoc_fda_effect = 0.0  # Track FDA mixing magnitude
    N_train = 0
    
    # Initialize feature variables
    features_src = None
    features_tgt = None
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess data to real format
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: COMBINED PRE-SCALING ============
        # Compute scaling parameters for both domains and ground truth
        x_combined_flat = np.concatenate([x_src_real.reshape(-1), x_tgt_real.reshape(-1)])
        y_combined_flat = np.concatenate([y_src_real.reshape(-1), y_tgt_real.reshape(-1)])
        
        # Global parameters for inputs
        global_x_min = np.min(x_combined_flat)
        global_x_max = np.max(x_combined_flat)
        
        # Global parameters for ground truth  
        global_y_min = np.min(y_combined_flat)
        global_y_max = np.max(y_combined_flat)
        
        # Apply SAME scaling to both domains using global parameters
        x_prescaled_src, _, _ = minmaxScaler(x_src_real, min_pre=global_x_min, max_pre=global_x_max, lower_range=lower_range, linear_interp=linear_interp)
        x_prescaled_tgt, _, _ = minmaxScaler(x_tgt_real, min_pre=global_x_min, max_pre=global_x_max, lower_range=lower_range, linear_interp=linear_interp)
        
        # Scale ground truth using global ground truth parameters
        y_prescaled_src, _, _ = minmaxScaler(y_src_real, min_pre=global_y_min, max_pre=global_y_max, lower_range=lower_range, linear_interp=linear_interp)
        y_prescaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=global_y_min, max_pre=global_y_max, lower_range=lower_range, linear_interp=linear_interp)

        # ============ STEP 2: FDA OPERATIONS ============
        # Convert to complex for FDA (inputs now have compatible ranges)
        x_src_complex = tf.complex(x_prescaled_src[:,:,:,0], x_prescaled_src[:,:,:,1])
        x_tgt_complex = tf.complex(x_prescaled_tgt[:,:,:,0], x_prescaled_tgt[:,:,:,1])
        
        # Extract Delay-Doppler representations
        src_amplitude, src_phase = F_extract_DD(x_src_complex)
        tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)
        
        # Mix amplitudes: Target style in center, Source style in outer regions
        # Now both amplitudes have compatible ranges due to combined pre-scaling
        mixed_amplitude = fda_mix_pixels(src_amplitude, tgt_amplitude, fda_win_h, fda_win_w)
        
        # Keep source phase (content preservation)
        mixed_complex_dd = apply_phase_to_amplitude(mixed_amplitude, src_phase)
        
        # Convert back to Time-Frequency domain
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd)
        
        # Convert back to real format
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        # Track FDA mixing effect
        fda_difference = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_prescaled_src))
        epoc_fda_effect += fda_difference.numpy() * x_src.shape[0]
        
        # ============ STEP 3: POST-FDA SCALING (ESSENTIAL!) ============
        # Scale FDA output back to [-1,1] for CNN compatibility
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_scaled_fda, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        # Also ensure source input is properly scaled to [-1,1] for consistency
        x_final_src, src_final_min, src_final_max = minmaxScaler(x_prescaled_src, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: CNN TRAINING with RESIDUAL LEARNING ============
        with tf.GradientTape() as tape:
            batch_size = x_final_src.shape[0]  # Use .shape instead of tf.shape
            
            if fda_weight < 1.0:
                # Hybrid training: Part FDA-mixed, part pure source
                fda_samples = int(batch_size * fda_weight)
                remaining_samples = batch_size - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    # Split batch for hybrid training
                    x_train_fda = x_scaled_fda[:fda_samples]      # FDA-mixed samples [-1,1]
                    x_train_src = x_final_src[fda_samples:]       # Pure source samples [-1,1]
                    x_train_combined = tf.concat([x_train_fda, x_train_src], axis=0)
                    
                    # Forward pass on combined input
                    residual_combined, features_src = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    # Pure source training
                    residual_src, features_src = model_cnn(x_final_src, training=True, return_features=True)
                    x_corrected_combined = x_final_src + residual_src
                    residual_combined = residual_src
                else:
                    # Pure FDA training
                    residual_fda, features_src = model_cnn(x_scaled_fda, training=True, return_features=True)
                    x_corrected_combined = x_scaled_fda + residual_fda
                    residual_combined = residual_fda
                
                # Estimation loss: Use properly scaled ground truth
                est_loss = loss_fn_est(y_prescaled_src, x_corrected_combined)
                    
            else:
                # Pure FDA training: Use only FDA-mixed input
                residual_fda, features_src = model_cnn(x_scaled_fda, training=True, return_features=True)
                x_corrected_fda = x_scaled_fda + residual_fda
                residual_combined = residual_fda
                
                # Estimation loss: FDA-corrected vs source ground truth
                est_loss = loss_fn_est(y_prescaled_src, x_corrected_fda)
            
            # Residual regularization: Encourage small, meaningful corrections
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # Total loss
            total_loss = est_weight * est_loss + residual_reg + smoothness_loss
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # ============ MONITOR TARGET PERFORMANCE ============
        # Apply same scaling workflow for target (for monitoring only)
        x_final_tgt, tgt_final_min, tgt_final_max = minmaxScaler(x_prescaled_tgt, lower_range=lower_range, linear_interp=linear_interp)
        
        residual_tgt, features_tgt = model_cnn(x_final_tgt, training=False, return_features=True)
        x_corrected_tgt = x_final_tgt + residual_tgt
        est_loss_tgt = loss_fn_est(y_prescaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # ============ SAVE FEATURES IF REQUIRED ============
        if save_features and features_src is not None:
            # Save features
            features_np_source = features_src[-1].numpy()
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
                
            if features_tgt is not None:
                features_np_target = features_tgt[-1].numpy()
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
        
        # Single optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features:    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect = epoc_fda_effect / N_train
    
    # Print enhanced FDA statistics
    print(f"    Combined-scaled FDA residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA mixing effect: {avg_fda_effect:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    print(f"    Global input range: [{global_x_min:.6f}, {global_x_max:.6f}]")
    print(f"    Global GT range: [{global_y_min:.6f}, {global_y_max:.6f}]")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,  # No domain loss - FDA handles domain gap at input level
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_src[-1] if features_src is not None else None,
        film_features_source=features_src[-1] if features_src is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )
    

def train_step_cnn_residual_fda_rawThenScale(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                save_features=False, nsymb=14, weights=None, linear_interp=False,
                                fda_win_h=13, fda_win_w=3, fda_weight=1.0):
    """
    CNN-only residual training step with RAW FDA followed by scaling
    
    Workflow:
    1. Raw FDA: Apply FDA directly on unscaled source and target inputs
    2. Convert back to Time-Frequency domain
    3. Scale FDA output to [-1,1] and get its min-max parameters
    4. Scale source ground truth with SAME min-max parameters from FDA output
    5. Train CNN with residual learning
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_residual_norm = 0.0
    epoc_fda_effect = 0.0  # Track FDA mixing magnitude
    N_train = 0
    
    # Initialize feature variables
    features_src = None
    features_tgt = None
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: RAW FDA OPERATIONS (NO PRE-SCALING) ============
        # Convert raw data to complex for FDA
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        # Extract Delay-Doppler representations from RAW data
        src_amplitude, src_phase = F_extract_DD(x_src_complex)
        tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)
        
        # Mix amplitudes: Target style in center, Source style in outer regions
        # WARNING: This may have amplitude range mismatches due to no pre-scaling
        mixed_amplitude = fda_mix_pixels(src_amplitude, tgt_amplitude, fda_win_h, fda_win_w)
        
        # Keep source phase (content preservation)
        mixed_complex_dd = apply_phase_to_amplitude(mixed_amplitude, src_phase)
        
        # Convert back to Time-Frequency domain
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd)
        
        # Convert back to real format [batch, 132, 14, 2]
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        # Track FDA mixing effect
        fda_difference = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect += fda_difference.numpy() * x_src.shape[0]
        
        # ============ STEP 2: SCALE FDA OUTPUT TO [-1,1] ============
        # Scale FDA-mixed output to [-1,1] and get its scaling parameters
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 3: SCALE GROUND TRUTH WITH FDA SCALING PARAMETERS ============
        # Convert FDA scaling parameters to format expected by minmaxScaler
        batch_size = y_src_real.shape[0]
        
        # Handle scalar vs array FDA parameters
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))  # [batch_size, 2]
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))  # [batch_size, 2]
        else:
            # If fda_min/fda_max are already arrays, use them directly
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        # Scale source ground truth using FDA output's scaling parameters
        y_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                    lower_range=lower_range, linear_interp=linear_interp)
        
        # Also scale source input with FDA parameters for consistency (for hybrid training)
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: CNN TRAINING with RESIDUAL LEARNING ============
        with tf.GradientTape() as tape:
            batch_size_int = x_fda_scaled.shape[0]
            
            if fda_weight < 1.0:
                # Hybrid training: Part FDA-mixed, part source (both using FDA scaling parameters)
                fda_samples = int(batch_size_int * fda_weight)
                remaining_samples = batch_size_int - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    # Split batch for hybrid training
                    x_train_fda = x_fda_scaled[:fda_samples]                      # FDA-mixed samples
                    x_train_src = x_src_scaled_with_fda_params[fda_samples:]      # Source scaled with FDA params
                    x_train_combined = tf.concat([x_train_fda, x_train_src], axis=0)
                    
                    # Forward pass on combined input
                    residual_combined, features_src = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    # Pure source training (scaled with FDA parameters)
                    residual_src, features_src = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                    x_corrected_combined = x_src_scaled_with_fda_params + residual_src
                    residual_combined = residual_src
                else:
                    # Pure FDA training
                    residual_fda, features_src = model_cnn(x_fda_scaled, training=True, return_features=True)
                    x_corrected_combined = x_fda_scaled + residual_fda
                    residual_combined = residual_fda
                
                # Estimation loss: Both model output and ground truth in FDA scaling space
                est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_combined)
                    
            else:
                # Pure FDA training: Use only FDA-mixed input
                residual_fda, features_src = model_cnn(x_fda_scaled, training=True, return_features=True)
                x_corrected_fda = x_fda_scaled + residual_fda
                residual_combined = residual_fda
                
                # Estimation loss: FDA-corrected vs ground truth (both in FDA scaling space)
                est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_fda)
            
            # Residual regularization: Encourage small, meaningful corrections
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # Total loss
            total_loss = est_weight * est_loss + residual_reg + smoothness_loss
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # ============ MONITOR TARGET PERFORMANCE (OPTIONAL) ============
        # Scale target with its own parameters for monitoring (traditional approach)
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        x_corrected_tgt = x_scaled_tgt + residual_tgt
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # ============ SAVE FEATURES IF REQUIRED ============
        if save_features and features_src is not None:
            # Save features
            features_np_source = features_src[-1].numpy()
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
                
            if features_tgt is not None:
                features_np_target = features_tgt[-1].numpy()
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
        
        # Single optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features:    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect = epoc_fda_effect / N_train
    
    # Print enhanced FDA statistics
    print(f"    RAW FDA → Scale residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA mixing effect (raw): {avg_fda_effect:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,  # No domain loss - FDA handles domain gap at input level
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_src[-1] if features_src is not None else None,
        film_features_source=features_src[-1] if features_src is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )    


##
def train_step_cnn_residual_fda_jmmd_rawThenScale(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                                save_features=False, nsymb=14, weights=None, linear_interp=False,
                                                fda_win_h=13, fda_win_w=3, fda_weight=1.0, jmmd_loss_fn=None):
    """
    CNN-only residual training step combining RAW FDA input translation with JMMD feature alignment
    
    Workflow:
    1. RAW FDA: Apply FDA directly on unscaled source and target inputs for input translation
    2. Scale FDA output to [-1,1] and get its scaling parameters
    3. Scale ground truth with FDA's scaling parameters
    4. Dual flow: Raw target input + FDA-translated input → CNN → Extract features
    5. JMMD alignment: Minimize feature distance between raw target and FDA-translated features
    6. Residual learning: Train CNN to predict corrections
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of (loader_H_input_train_src, loader_H_true_train_src, 
                        loader_H_input_train_tgt, loader_H_true_train_tgt)
        loss_fn: tuple of loss functions (estimation_loss, bce_loss) - only first one used
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features for PAD computation
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height for mixing (default 13)
        fda_win_w: FDA window width for mixing (default 3) 
        fda_weight: Weight for FDA vs source training (1.0 = 100% FDA)
        jmmd_loss_fn: JMMD loss function instance (if None, will create default)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize JMMD loss if not provided
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_jmmd = 0.0
    epoc_residual_norm = 0.0
    epoc_fda_effect = 0.0
    N_train = 0
    
    # Initialize feature variables
    features_src = None
    features_tgt_raw = None
    features_tgt_fda = None
    
    if save_features and domain_weight != 0:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET - RAW FDA)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: RAW FDA INPUT TRANSLATION ============
        # Convert raw data to complex for FDA
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        # Extract Delay-Doppler representations from RAW data
        src_amplitude, src_phase = F_extract_DD(x_src_complex)
        tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)
        
        # Mix amplitudes: Target style in center, Source style in outer regions
        mixed_amplitude = fda_mix_pixels(src_amplitude, tgt_amplitude, fda_win_h, fda_win_w)
        
        # Keep source phase (content preservation)
        mixed_complex_dd = apply_phase_to_amplitude(mixed_amplitude, src_phase)
        
        # Convert back to Time-Frequency domain
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd)
        
        # Convert back to real format [batch, 132, 14, 2]
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        # Track FDA mixing effect
        fda_difference = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect += fda_difference.numpy() * x_src.shape[0]
        
        # ============ STEP 2: SCALE FDA OUTPUT ============
        # Scale FDA-mixed output to [-1,1] and get its scaling parameters
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 3: SCALE TARGET INPUT SEPARATELY FOR DUAL FLOW ============
        # Scale target input with its own parameters for the dual flow
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE GROUND TRUTH WITH FDA PARAMETERS ============
        # Convert FDA scaling parameters to format expected by minmaxScaler
        batch_size = y_src_real.shape[0]
        
        # Handle scalar vs array FDA parameters
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))  # [batch_size, 2]
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))  # [batch_size, 2]
        else:
            # If fda_min/fda_max are already arrays, use them directly
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        # Scale source ground truth using FDA output's scaling parameters
        y_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                    lower_range=lower_range, linear_interp=linear_interp)
        
        # Also scale source input with FDA parameters for consistency (for hybrid training)
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # === DUAL FLOW FEATURE EXTRACTION + TRAINING ===
        with tf.GradientTape() as tape:
            # === FLOW 1: FDA-TRANSLATED INPUT (Main training flow) ===
            residual_fda, features_tgt_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
            x_corrected_fda = x_fda_scaled + residual_fda
            
            # === FLOW 2: RAW TARGET INPUT (Alignment flow for JMMD) ===
            # Use target's own scaling for consistency with validation
            residual_tgt_raw, features_tgt_raw = model_cnn(x_scaled_tgt, training=False, return_features=True)
            # Note: training=False to prevent this flow from affecting main training gradients
            
            # === FLOW 3: SOURCE INPUT (Hybrid training if needed) ===
            if fda_weight < 1.0:
                residual_src, features_src = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                x_corrected_src = x_src_scaled_with_fda_params + residual_src
            else:
                features_src = features_tgt_fda  # Use FDA features as source features
            
            # === ESTIMATION LOSS ===
            if fda_weight < 1.0:
                # Hybrid training: FDA + Source
                batch_size_int = x_fda_scaled.shape[0]
                fda_samples = int(batch_size_int * fda_weight)
                
                if fda_samples > 0 and (batch_size_int - fda_samples) > 0:
                    # Mixed batch: Part FDA, part source
                    est_loss_fda = loss_fn_est(y_scaled_with_fda_params[:fda_samples], x_corrected_fda[:fda_samples])
                    est_loss_src = loss_fn_est(y_scaled_with_fda_params[fda_samples:], x_corrected_src[fda_samples:])
                    est_loss = (fda_weight * est_loss_fda + (1 - fda_weight) * est_loss_src)
                    
                    # Combined residual for regularization
                    residual_combined = tf.concat([residual_fda[:fda_samples], residual_src[fda_samples:]], axis=0)
                elif fda_samples == 0:
                    # Pure source
                    est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_src)
                    residual_combined = residual_src
                else:
                    # Pure FDA
                    est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_fda)
                    residual_combined = residual_fda
            else:
                # Pure FDA training
                est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_fda)
                residual_combined = residual_fda
            
            # === JMMD LOSS: Align Raw Target and FDA-Translated Features ===
            # This encourages the model to extract similar features from both raw target and FDA-translated inputs
            jmmd_loss = jmmd_loss_fn(features_tgt_raw, features_tgt_fda)
            
            # === REGULARIZATION LOSSES ===
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            
            # Smoothness loss
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0 and fda_samples > 0 and (batch_size_int - fda_samples) > 0:
                    smoothness_fda = compute_total_smoothness_loss(x_corrected_fda[:fda_samples], 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                    smoothness_src = compute_total_smoothness_loss(x_corrected_src[fda_samples:], 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                    smoothness_loss = (smoothness_fda + smoothness_src) / 2
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # === TOTAL LOSS ===
            total_loss = (est_weight * est_loss + 
                        domain_weight * jmmd_loss +      # FDA-JMMD alignment loss
                        residual_reg + 
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # === MONITOR TARGET PERFORMANCE (separate target evaluation) ===
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        x_corrected_tgt_raw = x_scaled_tgt + residual_tgt_raw
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt_raw)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === SAVE FEATURES IF REQUIRED ===
        if save_features and domain_weight != 0:
            # Save FDA-translated features
            features_np_fda = features_tgt_fda[-1].numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features', data=features_np_fda,
                    maxshape=(None,) + features_np_fda.shape[1:], chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_fda.shape[0], axis=0)
                features_dataset_source[-features_np_fda.shape[0]:] = features_np_fda
                
            # Save raw target features
            features_np_raw = features_tgt_raw[-1].numpy()
            if features_dataset_target is None:
                features_dataset_target = features_h5_target.create_dataset(
                    'features', data=features_np_raw,
                    maxshape=(None,) + features_np_raw.shape[1:], chunks=True
                )
            else:
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_raw.shape[0], axis=0)
                features_dataset_target[-features_np_raw.shape[0]:] = features_np_raw
        
        # Update model
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features and domain_weight != 0:
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect = epoc_fda_effect / N_train
    
    # Print enhanced statistics
    print(f"    RAW FDA + JMMD residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    RAW FDA + JMMD alignment loss: {avg_loss_jmmd:.6f}")
    print(f"    RAW FDA mixing effect: {avg_fda_effect:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd,               # RAW FDA-JMMD alignment loss
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_tgt_fda[-1] if features_tgt_fda is not None else None,
        film_features_source=features_tgt_fda[-1] if features_tgt_fda is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )

#
    

#
def train_step_cnn_residual_fda_jmmd_rawThenScale_domainAware(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                                            save_features=False, nsymb=14, weights=None, linear_interp=False,
                                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0, jmmd_loss_fn=None,
                                                            consistency_weight=0.3, residual_consistency_weight=0.2,
                                                            improvement_consistency_weight=0.1):
    """
    CNN-only residual training step combining RAW FDA input translation, JMMD feature alignment, 
    and domain-aware consistency for robust residual learning
    
    Workflow:
    1. RAW FDA: Apply FDA directly on unscaled source and target inputs for input translation
    2. Scale FDA output to [-1,1] and get its scaling parameters
    3. Scale ground truth with FDA's scaling parameters
    4. Dual flow: Raw target input + FDA-translated input → CNN → Extract features
    5. JMMD alignment: Minimize feature distance between raw target and FDA-translated features
    6. Domain-aware consistency: Similar improvement ratios across domains
    7. Residual pattern consistency: Similar residual correction patterns (relative to input magnitude)
    8. Residual learning: Train CNN to predict corrections
    
    Args:
        consistency_weight: Weight for improvement ratio consistency (default 0.3)
        residual_consistency_weight: Weight for residual pattern consistency (default 0.2)
        improvement_consistency_weight: Weight for improvement percentage consistency (default 0.1)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize JMMD loss if not provided
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_jmmd = 0.0
    epoc_loss_domain_aware_consistency = 0.0  # Track domain-aware consistency
    epoc_loss_residual_consistency = 0.0      # Track residual pattern consistency
    epoc_loss_improvement_consistency = 0.0   # Track improvement percentage consistency
    epoc_residual_norm = 0.0
    epoc_fda_effect = 0.0
    N_train = 0
    
    # Initialize feature variables
    features_src = None
    features_tgt_raw = None
    features_tgt_fda = None
    
    if save_features and domain_weight != 0:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET - RAW FDA)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: RAW FDA INPUT TRANSLATION ============
        # Convert raw data to complex for FDA
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        # Extract Delay-Doppler representations from RAW data
        src_amplitude, src_phase = F_extract_DD(x_src_complex)
        tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)
        
        # Mix amplitudes: Target style in center, Source style in outer regions
        mixed_amplitude = fda_mix_pixels(src_amplitude, tgt_amplitude, fda_win_h, fda_win_w)
        
        # Keep source phase (content preservation)
        mixed_complex_dd = apply_phase_to_amplitude(mixed_amplitude, src_phase)
        
        # Convert back to Time-Frequency domain
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd)
        
        # Convert back to real format [batch, 132, 14, 2]
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        # Track FDA mixing effect
        fda_difference = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect += fda_difference.numpy() * x_src.shape[0]
        
        # ============ STEP 2: SCALE FDA OUTPUT ============
        # Scale FDA-mixed output to [-1,1] and get its scaling parameters
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 3: SCALE TARGET INPUT SEPARATELY FOR DUAL FLOW ============
        # Scale target input with its own parameters for the dual flow
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE GROUND TRUTH WITH FDA PARAMETERS ============
        # Convert FDA scaling parameters to format expected by minmaxScaler
        batch_size = y_src_real.shape[0]
        
        # Handle scalar vs array FDA parameters
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))
        else:
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        # Scale source ground truth using FDA output's scaling parameters
        y_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                    lower_range=lower_range, linear_interp=linear_interp)
        
        # Also scale source input with FDA parameters for consistency (for hybrid training)
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # === DUAL FLOW FEATURE EXTRACTION + TRAINING + DOMAIN-AWARE CONSISTENCY ===
        with tf.GradientTape() as tape:
            # === FLOW 1: FDA-TRANSLATED INPUT (Main training flow) ===
            residual_fda, features_tgt_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
            x_corrected_fda = x_fda_scaled + residual_fda
            
            # === FLOW 2: RAW TARGET INPUT (Alignment flow for JMMD) ===
            residual_tgt_raw, features_tgt_raw = model_cnn(x_scaled_tgt, training=False, return_features=True)
            x_corrected_tgt_raw = x_scaled_tgt + residual_tgt_raw
            # Note: training=False to prevent this flow from affecting main training gradients
            
            # === FLOW 3: SOURCE INPUT (Hybrid training if needed) ===
            if fda_weight < 1.0:
                residual_src, features_src = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                x_corrected_src = x_src_scaled_with_fda_params + residual_src
            else:
                features_src = features_tgt_fda  # Use FDA features as source features
            
            # === ESTIMATION LOSS ===
            if fda_weight < 1.0:
                # Hybrid training: FDA + Source
                batch_size_int = x_fda_scaled.shape[0]
                fda_samples = int(batch_size_int * fda_weight)
                
                if fda_samples > 0 and (batch_size_int - fda_samples) > 0:
                    est_loss_fda = loss_fn_est(y_scaled_with_fda_params[:fda_samples], x_corrected_fda[:fda_samples])
                    est_loss_src = loss_fn_est(y_scaled_with_fda_params[fda_samples:], x_corrected_src[fda_samples:])
                    est_loss = (fda_weight * est_loss_fda + (1 - fda_weight) * est_loss_src)
                    
                    residual_combined = tf.concat([residual_fda[:fda_samples], residual_src[fda_samples:]], axis=0)
                elif fda_samples == 0:
                    est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_src)
                    residual_combined = residual_src
                else:
                    est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_fda)
                    residual_combined = residual_fda
            else:
                # Pure FDA training
                est_loss = loss_fn_est(y_scaled_with_fda_params, x_corrected_fda)
                residual_combined = residual_fda
            
            # === JMMD LOSS: Align Raw Target and FDA-Translated Features ===
            jmmd_loss = jmmd_loss_fn(features_tgt_raw, features_tgt_fda)
            
            # === DOMAIN-AWARE CONSISTENCY LOSSES (NEW!) ===
            # Note: For consistency calculation, we need target ground truth scaled with target's own parameters
            y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
            
            # 1. IMPROVEMENT RATIO CONSISTENCY (Main domain-aware consistency)
            if improvement_consistency_weight > 0:
                # Calculate improvement ratios for both domains
                # FDA domain: input error vs corrected error
                fda_input_error = tf.reduce_mean(tf.abs(x_fda_scaled - y_scaled_with_fda_params), axis=[1,2,3])
                fda_corrected_error = tf.reduce_mean(tf.abs(x_corrected_fda - y_scaled_with_fda_params), axis=[1,2,3])
                fda_improvement = (fda_input_error - fda_corrected_error) / (fda_input_error + 1e-8)
                
                # Target domain: input error vs corrected error
                tgt_input_error = tf.reduce_mean(tf.abs(x_scaled_tgt - y_scaled_tgt), axis=[1,2,3])
                tgt_corrected_error = tf.reduce_mean(tf.abs(x_corrected_tgt_raw - y_scaled_tgt), axis=[1,2,3])
                tgt_improvement = (tgt_input_error - tgt_corrected_error) / (tgt_input_error + 1e-8)
                
                # Consistency: similar improvement percentages
                improvement_consistency_loss = tf.reduce_mean(tf.square(fda_improvement - tgt_improvement))
            else:
                improvement_consistency_loss = 0.0
            
            # 2. RESIDUAL PATTERN CONSISTENCY (Perfect for your residual learning!)
            if residual_consistency_weight > 0:
                # Normalize residuals by input magnitude to account for domain differences
                fda_input_magnitude = tf.abs(x_fda_scaled) + 1e-8
                tgt_input_magnitude = tf.abs(x_scaled_tgt) + 1e-8
                
                # Normalized residual patterns
                fda_residual_normalized = residual_fda / fda_input_magnitude
                tgt_residual_normalized = residual_tgt_raw / tgt_input_magnitude
                
                # Consistency: similar relative correction patterns
                residual_pattern_consistency = tf.reduce_mean(tf.square(fda_residual_normalized - tgt_residual_normalized))
            else:
                residual_pattern_consistency = 0.0
            
            # 3. DIRECT CONSISTENCY (Traditional approach, optional)
            if consistency_weight > 0:
                # Direct output consistency (less suitable for your case but included for completeness)
                batch_size_tensor = tf.shape(x_fda_scaled)[0]
                min_batch_size = tf.minimum(batch_size_tensor, tf.shape(x_scaled_tgt)[0])
                
                # Take subset for consistency calculation
                fda_subset = x_corrected_fda[:min_batch_size]
                tgt_subset = x_corrected_tgt_raw[:min_batch_size]
                
                # Normalize by respective input magnitudes for fair comparison
                fda_input_ref = tf.abs(x_fda_scaled[:min_batch_size]) + 1e-8
                tgt_input_ref = tf.abs(x_scaled_tgt[:min_batch_size]) + 1e-8
                
                fda_normalized = fda_subset / fda_input_ref
                tgt_normalized = tgt_subset / tgt_input_ref
                
                direct_consistency_loss = tf.reduce_mean(tf.square(fda_normalized - tgt_normalized))
            else:
                direct_consistency_loss = 0.0
            
            # === REGULARIZATION LOSSES ===
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            
            # Smoothness loss
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0 and fda_samples > 0 and (batch_size_int - fda_samples) > 0:
                    smoothness_fda = compute_total_smoothness_loss(x_corrected_fda[:fda_samples], 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                    smoothness_src = compute_total_smoothness_loss(x_corrected_src[fda_samples:], 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                    smoothness_loss = (smoothness_fda + smoothness_src) / 2
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # === TOTAL LOSS (WITH DOMAIN-AWARE CONSISTENCY) ===
            total_loss = (est_weight * est_loss + 
                        domain_weight * jmmd_loss +                              # Feature alignment
                        improvement_consistency_weight * improvement_consistency_loss +  # NEW: Improvement ratio consistency
                        residual_consistency_weight * residual_pattern_consistency +    # NEW: Residual pattern consistency  
                        consistency_weight * direct_consistency_loss +                  # NEW: Direct consistency (optional)
                        residual_reg + 
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # === MONITOR TARGET PERFORMANCE (separate target evaluation) ===
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt_raw)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === SAVE FEATURES IF REQUIRED ===
        if save_features and domain_weight != 0:
            # Save FDA-translated features
            features_np_fda = features_tgt_fda[-1].numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features', data=features_np_fda,
                    maxshape=(None,) + features_np_fda.shape[1:], chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_fda.shape[0], axis=0)
                features_dataset_source[-features_np_fda.shape[0]:] = features_np_fda
                
            # Save raw target features
            features_np_raw = features_tgt_raw[-1].numpy()
            if features_dataset_target is None:
                features_dataset_target = features_h5_target.create_dataset(
                    'features', data=features_np_raw,
                    maxshape=(None,) + features_np_raw.shape[1:], chunks=True
                )
            else:
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_raw.shape[0], axis=0)
                features_dataset_target[-features_np_raw.shape[0]:] = features_np_raw
        
        # Update model
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0]
        epoc_loss_domain_aware_consistency += improvement_consistency_loss.numpy() * x_src.shape[0]  # Track new consistency
        epoc_loss_residual_consistency += residual_pattern_consistency.numpy() * x_src.shape[0]      # Track residual consistency
        epoc_loss_improvement_consistency += direct_consistency_loss.numpy() * x_src.shape[0]        # Track direct consistency
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features and domain_weight != 0:
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_loss_domain_aware_consistency = epoc_loss_domain_aware_consistency / N_train  # New metric
    avg_loss_residual_consistency = epoc_loss_residual_consistency / N_train          # New metric
    avg_loss_improvement_consistency = epoc_loss_improvement_consistency / N_train     # New metric
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect = epoc_fda_effect / N_train
    
    # Print enhanced statistics with new consistency metrics
    print(f"    RAW FDA + JMMD + Domain-Aware residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    RAW FDA + JMMD alignment loss: {avg_loss_jmmd:.6f}")
    print(f"    Improvement ratio consistency: {avg_loss_domain_aware_consistency:.6f}")  # NEW
    print(f"    Residual pattern consistency: {avg_loss_residual_consistency:.6f}")      # NEW  
    print(f"    Direct consistency: {avg_loss_improvement_consistency:.6f}")             # NEW
    print(f"    RAW FDA mixing effect: {avg_fda_effect:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd + avg_loss_domain_aware_consistency + avg_loss_residual_consistency,  # Combined domain losses
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_tgt_fda[-1] if features_tgt_fda is not None else None,
        film_features_source=features_tgt_fda[-1] if features_tgt_fda is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )

# 
def val_step_cnn_residual_fda_jmmd_rawThenScale_domainAware(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, 
                                                        weights=None, linear_interp=False, return_H_gen=False,
                                                        fda_win_h=13, fda_win_w=3, fda_weight=1.0, jmmd_loss_fn=None,
                                                        consistency_weight=0.3, residual_consistency_weight=0.2,
                                                        improvement_consistency_weight=0.1):
    """
    Validation step for CNN-only residual learning with RAW FDA + JMMD + Domain-Aware Consistency
    
    Validates model performance using direct application to source and target inputs (no FDA translation during validation)
    Calculates domain-aware consistency metrics for monitoring training effectiveness
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of validation loaders (input_src, true_src, input_tgt, true_tgt)
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
        fda_win_h: FDA window height (for optional FDA validation test)
        fda_win_w: FDA window width (for optional FDA validation test)
        fda_weight: FDA weight (for compatibility)
        jmmd_loss_fn: JMMD loss function instance
        consistency_weight: Weight for direct consistency (for monitoring)
        residual_consistency_weight: Weight for residual pattern consistency (for monitoring)
        improvement_consistency_weight: Weight for improvement ratio consistency (for monitoring)
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize JMMD loss if not provided
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_jmmd_loss = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0
    
    # Domain-aware consistency tracking
    epoc_improvement_consistency = 0.0
    epoc_residual_consistency = 0.0
    epoc_direct_consistency = 0.0
    
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # Get and preprocess data (same as training)
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocessing (same as training)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
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

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
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

        # Track residual magnitudes during validation
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === JMMD Loss (on features for monitoring) ===
        if domain_weight > 0:
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # === DOMAIN-AWARE CONSISTENCY METRICS (FOR MONITORING) ===
        
        # 1. IMPROVEMENT RATIO CONSISTENCY
        if improvement_consistency_weight > 0:
            # Source domain improvement ratio
            src_input_error = tf.reduce_mean(tf.abs(x_scaled_src - y_scaled_src), axis=[1,2,3])
            src_corrected_error = tf.reduce_mean(tf.abs(preds_src - y_scaled_src), axis=[1,2,3])
            src_improvement = (src_input_error - src_corrected_error) / (src_input_error + 1e-8)
            
            # Target domain improvement ratio
            tgt_input_error = tf.reduce_mean(tf.abs(x_scaled_tgt - y_scaled_tgt), axis=[1,2,3])
            tgt_corrected_error = tf.reduce_mean(tf.abs(preds_tgt - y_scaled_tgt), axis=[1,2,3])
            tgt_improvement = (tgt_input_error - tgt_corrected_error) / (tgt_input_error + 1e-8)
            
            # Consistency metric: How similar are improvement ratios?
            improvement_consistency = tf.reduce_mean(tf.square(src_improvement - tgt_improvement))
            epoc_improvement_consistency += improvement_consistency.numpy() * x_src.shape[0]
        
        # 2. RESIDUAL PATTERN CONSISTENCY
        if residual_consistency_weight > 0:
            # Normalize residuals by input magnitude
            src_input_magnitude = tf.abs(x_scaled_src) + 1e-8
            tgt_input_magnitude = tf.abs(x_scaled_tgt) + 1e-8
            
            src_residual_normalized = residual_src / src_input_magnitude
            tgt_residual_normalized = residual_tgt / tgt_input_magnitude
            
            # Consistency metric: How similar are relative correction patterns?
            residual_pattern_consistency = tf.reduce_mean(tf.square(src_residual_normalized - tgt_residual_normalized))
            epoc_residual_consistency += residual_pattern_consistency.numpy() * x_src.shape[0]
        
        # 3. DIRECT OUTPUT CONSISTENCY (OPTIONAL)
        if consistency_weight > 0:
            # Direct output consistency (less suitable but included for completeness)
            batch_size_tensor = tf.shape(x_scaled_src)[0]
            min_batch_size = tf.minimum(batch_size_tensor, tf.shape(x_scaled_tgt)[0])
            
            # Take subset for consistency calculation
            src_subset = preds_src[:min_batch_size]
            tgt_subset = preds_tgt[:min_batch_size]
            
            # Normalize by respective input magnitudes for fair comparison
            src_input_ref = tf.abs(x_scaled_src[:min_batch_size]) + 1e-8
            tgt_input_ref = tf.abs(x_scaled_tgt[:min_batch_size]) + 1e-8
            
            src_normalized = src_subset / src_input_ref
            tgt_normalized = tgt_subset / tgt_input_ref
            
            direct_consistency = tf.reduce_mean(tf.square(src_normalized - tgt_normalized))
            epoc_direct_consistency += direct_consistency.numpy() * x_src.shape[0]
        
        # === SMOOTHNESS LOSS COMPUTATION ===
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

        # ============ OPTIONAL FDA VALIDATION TEST ============
        # Test FDA mixing during validation for monitoring (only on first batch)
        if idx == 0 and fda_weight > 0:
            try:
                # Apply FDA mixing like in training but for validation monitoring
                x_src_complex_val = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
                x_tgt_complex_val = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
                
                # Extract Delay-Doppler representations
                src_amplitude_val, src_phase_val = F_extract_DD(x_src_complex_val)
                tgt_amplitude_val, tgt_phase_val = F_extract_DD(x_tgt_complex_val)
                
                # Mix amplitudes
                mixed_amplitude_val = fda_mix_pixels(src_amplitude_val, tgt_amplitude_val, fda_win_h, fda_win_w)
                
                # Keep source phase
                mixed_complex_dd_val = apply_phase_to_amplitude(mixed_amplitude_val, src_phase_val)
                
                # Convert back to Time-Frequency domain
                x_fda_mixed_complex_val = F_inverse_DD(mixed_complex_dd_val)
                x_fda_mixed_real_val = tf.stack([tf.math.real(x_fda_mixed_complex_val), tf.math.imag(x_fda_mixed_complex_val)], axis=-1)
                
                # Scale FDA output
                x_fda_mixed_numpy_val = x_fda_mixed_real_val.numpy()
                x_fda_scaled_val, _, _ = minmaxScaler(x_fda_mixed_numpy_val, lower_range=lower_range, linear_interp=linear_interp)
                
                # Test FDA-mixed input performance
                residual_fda_val, _ = model_cnn(x_fda_scaled_val, training=False, return_features=False)
                preds_fda_val = x_fda_scaled_val + residual_fda_val
                
                print(f"    FDA validation test completed on batch {idx}")
                
            except Exception as e:
                print(f"    Warning: FDA validation test failed: {e}")

        # === Save H samples (same structure as training validation) ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Domain-aware consistency averages
    avg_improvement_consistency = epoc_improvement_consistency / N_val_source if epoc_improvement_consistency > 0 else 0.0
    avg_residual_consistency = epoc_residual_consistency / N_val_source if epoc_residual_consistency > 0 else 0.0
    avg_direct_consistency = epoc_direct_consistency / N_val_source if epoc_direct_consistency > 0 else 0.0
    
    # Print enhanced validation statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    Validation JMMD loss: {avg_jmmd_loss:.6f}")
    print(f"    Validation improvement consistency: {avg_improvement_consistency:.6f}")
    print(f"    Validation residual pattern consistency: {avg_residual_consistency:.6f}")
    print(f"    Validation direct consistency: {avg_direct_consistency:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Domain accuracy placeholders (compatible with existing code)
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss (no discriminator component)
    avg_total_loss = (est_weight * avg_loss_est + 
                    domain_weight * avg_jmmd_loss + 
                    improvement_consistency_weight * avg_improvement_consistency +
                    residual_consistency_weight * avg_residual_consistency +
                    consistency_weight * avg_direct_consistency +
                    avg_smoothness_loss)

    # Return compatible structure with enhanced domain-aware metrics
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_domain_loss': avg_jmmd_loss + avg_improvement_consistency + avg_residual_consistency,  # Combined domain-aware loss
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss,
        # Additional domain-aware consistency metrics for detailed monitoring
        'avg_improvement_consistency': avg_improvement_consistency,
        'avg_residual_consistency': avg_residual_consistency,
        'avg_direct_consistency': avg_direct_consistency
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return

#
def train_step_cnn_residual_fda_fullTranslation1(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                            save_features=False, nsymb=14, weights=None, linear_interp=False,
                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0, flag_residual_reg=True):
    """
    CNN-only residual training step with FULL FDA translation (both input AND labels) using RAW FDA first
    
    Workflow:
    1. RAW FDA Input Translation: Source input → Target style input (on unscaled data)
    2. RAW FDA Label Translation: Source labels → Target style input (on unscaled data)
    3. Scale FDA outputs and use FDA scaling parameters for consistency
    4. Train CNN with residual learning using FULLY translated pairs
    
    Training pairs:
    - Input: Source content + Target style (FDA-mixed input, then scaled)
    - Label: Source labels + Target style (FDA-mixed labels using TARGET INPUT as style, then scaled)
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of loaders
        loss_fn: tuple of loss functions (only first one used)
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features
        nsymb: number of symbols  
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height (default 13)
        fda_win_w: FDA window width (default 3)
        fda_weight: Weight for FDA vs source training (1.0 = 100% FDA)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_residual_norm = 0.0
    epoc_fda_effect_input = 0.0   # Track FDA effect on inputs
    epoc_fda_effect_label = 0.0   # Track FDA effect on labels
    N_train = 0
    
    # Initialize feature variables
    features_src = None
    features_tgt = None
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()  # Not used in label translation (we don't have target labels)
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET - RAW FDA FIRST!)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        # Note: We don't process y_tgt since we don't have target labels for style reference

        # ============ STEP 1: RAW FDA INPUT TRANSLATION (NO PRE-SCALING) ============
        # Convert raw data to complex for FDA
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        # Extract Delay-Doppler representations for RAW INPUTS
        src_amplitude_input, src_phase_input = F_extract_DD(x_src_complex)
        tgt_amplitude_input, tgt_phase_input = F_extract_DD(x_tgt_complex)
        
        # Mix amplitudes: Target style in center, Source style in outer regions
        mixed_amplitude_input = fda_mix_pixels(src_amplitude_input, tgt_amplitude_input, fda_win_h, fda_win_w)
        
        # Keep source phase (content preservation)
        mixed_complex_dd_input = apply_phase_to_amplitude(mixed_amplitude_input, src_phase_input)
        
        # Convert back to Time-Frequency domain
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_input)
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        # Track FDA effect on inputs
        fda_difference_input = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect_input += fda_difference_input.numpy() * x_src.shape[0]
        
        # ============ STEP 2: RAW FDA LABEL TRANSLATION - KEY FIX! ============
        # Convert raw labels to complex for FDA
        y_src_complex = tf.complex(y_src_real[:,:,:,0], y_src_real[:,:,:,1])
        # ← KEY FIX: Use TARGET INPUT (not target labels) as style reference for label translation
        # x_tgt_complex already computed above
        
        # Extract Delay-Doppler representations: SOURCE LABELS + TARGET INPUT STYLE
        src_amplitude_label, src_phase_label = F_extract_DD(y_src_complex)              # ← Source labels content
        tgt_amplitude_style, tgt_phase_style = F_extract_DD(x_tgt_complex)              # ← Target INPUT style (not labels!)
        
        # Mix label amplitudes: Use TARGET INPUT STYLE for labels (since we don't have target labels)
        mixed_amplitude_label = fda_mix_pixels(src_amplitude_label, tgt_amplitude_style, fda_win_h, fda_win_w)
        
        # Keep source label phase (content preservation for labels)
        mixed_complex_dd_label = apply_phase_to_amplitude(mixed_amplitude_label, src_phase_label)
        
        # Convert back to Time-Frequency domain
        y_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_label)
        y_fda_mixed_real = tf.stack([tf.math.real(y_fda_mixed_complex), tf.math.imag(y_fda_mixed_complex)], axis=-1)
        
        # Track FDA effect on labels
        fda_difference_label = tf.reduce_mean(tf.abs(y_fda_mixed_real - y_src_real))
        epoc_fda_effect_label += fda_difference_label.numpy() * x_src.shape[0]
        
        # ============ STEP 3: SCALE FDA OUTPUTS WITH SHARED PARAMETERS ============
        # Scale FDA-mixed input to [-1,1] and get scaling parameters
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        # Scale FDA-mixed labels using THE SAME scaling parameters for consistency
        batch_size = y_fda_mixed_real.shape[0]
        
        # Handle scalar vs array FDA parameters for labels
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))
        else:
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        # Scale FDA-mixed labels with FDA input's scaling parameters
        y_fda_mixed_numpy = y_fda_mixed_real.numpy()
        y_fda_scaled, _, _ = minmaxScaler(y_fda_mixed_numpy, min_pre=fda_min_array, max_pre=fda_max_array, 
                                        lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE SOURCE DATA WITH FDA PARAMETERS FOR HYBRID TRAINING ============
        # Scale source input and labels with FDA parameters for consistency (for hybrid training)
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)
        y_src_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # ============ STEP 5: CNN TRAINING with RESIDUAL LEARNING ============
        with tf.GradientTape() as tape:
            batch_size_int = x_fda_scaled.shape[0]
            
            if fda_weight < 1.0:
                # Hybrid training: Part FDA-translated, part source (both using FDA scaling parameters)
                fda_samples = int(batch_size_int * fda_weight)
                remaining_samples = batch_size_int - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    # Split batch for hybrid training
                    x_train_fda = x_fda_scaled[:fda_samples]                           # FDA-translated inputs
                    y_train_fda = y_fda_scaled[:fda_samples]                           # FDA-translated labels (using TARGET INPUT style)
                    
                    x_train_src = x_src_scaled_with_fda_params[fda_samples:]          # Source scaled with FDA params
                    y_train_src = y_src_scaled_with_fda_params[fda_samples:]          # Source labels scaled with FDA params
                    
                    # Combine for training (both in FDA scaling space)
                    x_train_combined = tf.concat([x_train_fda, x_train_src], axis=0)
                    y_train_combined = tf.concat([y_train_fda, y_train_src], axis=0)   # ← FDA labels + source labels
                    
                    # Forward pass on combined input
                    residual_combined, features_src = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    # Pure source training (scaled with FDA parameters)
                    residual_src, features_src = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                    x_corrected_combined = x_src_scaled_with_fda_params + residual_src
                    y_train_combined = y_src_scaled_with_fda_params  # Source labels in FDA scaling space
                    residual_combined = residual_src
                else:
                    # Pure FDA training (fully translated pairs using TARGET INPUT style for labels)
                    residual_fda, features_src = model_cnn(x_fda_scaled, training=True, return_features=True)
                    x_corrected_combined = x_fda_scaled + residual_fda
                    y_train_combined = y_fda_scaled  # ← FDA-translated labels using TARGET INPUT style
                    residual_combined = residual_fda
                
                # Estimation loss: Both inputs and labels in FDA scaling space
                est_loss = loss_fn_est(y_train_combined, x_corrected_combined)
                    
            else:
                # Pure FDA training: Use FULLY translated pairs (TARGET INPUT style for labels)
                residual_fda, features_src = model_cnn(x_fda_scaled, training=True, return_features=True)
                x_corrected_fda = x_fda_scaled + residual_fda
                residual_combined = residual_fda
                
                # Estimation loss: FDA-corrected input vs FDA-translated labels (labels use TARGET INPUT style)
                est_loss = loss_fn_est(y_fda_scaled, x_corrected_fda)
            
            # Residual regularization: Encourage small, meaningful corrections
            if flag_residual_reg:
                residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            else:
                residual_reg = 0.0
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # Total loss
            total_loss = est_weight * est_loss + residual_reg + smoothness_loss
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # ============ MONITOR TARGET PERFORMANCE (OPTIONAL) ============
        # Scale target with its own parameters for monitoring (traditional approach)
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        # Note: We can't monitor target labels since we don't have them - this is just input monitoring
        
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        x_corrected_tgt = x_scaled_tgt + residual_tgt
        # For monitoring, we can only compare corrected target input vs original target input (no ground truth)
        est_loss_tgt = loss_fn_est(x_scaled_tgt, x_corrected_tgt)  # Self-consistency check
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === Save features if required ===
        if save_features and features_src is not None:
            # Save features from FDA training
            features_np_source = features_src[-1].numpy()
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
                
            if features_tgt is not None:
                features_np_target = features_tgt[-1].numpy()
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
        
        # Single optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features:    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect_input = epoc_fda_effect_input / N_train
    avg_fda_effect_label = epoc_fda_effect_label / N_train
    
    # Print enhanced FDA statistics
    print(f"    RAW Full FDA Translation (Target Input Style) residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    RAW FDA input translation effect: {avg_fda_effect_input:.6f}")
    print(f"    RAW FDA label translation effect (using target input style): {avg_fda_effect_label:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    print(f"    Labels translated using TARGET INPUT style (no target labels needed)")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=0.0,  # No domain loss - Full FDA handles domain gap at data level
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_src[-1] if features_src is not None else None,
        film_features_source=features_src[-1] if features_src is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )
    
#
def train_step_cnn_residual_fda_fullTranslation2(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                            save_features=False, nsymb=14, weights=None, linear_interp=False,
                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0):
    """
    CNN-only residual training step with Target→Source FULL FDA translation for BOTH input and pseudo-labels
    
    Key Innovation: Uses TARGET data as primary training source (no target labels needed!)
    
    Workflow:
    1. FDA Input Translation: Target input → Source input
    2. FDA Pseudo-Label Generation: Target input → Source label
    3. Train CNN with residual learning using: (Translated target input, Translated target pseudo-labels)
    
    Training pairs:
    - Input: Target input translated to source input style
    - Label: Target input translated to source label style 

    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of loaders (target true is unused - we don't have target labels)
        loss_fn: tuple of loss functions (only first one used)
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features
        nsymb: number of symbols  
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height (default 13)
        fda_win_w: FDA window width (default 3)
        fda_weight: Weight for FDA vs pure target training (1.0 = 100% FDA)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_source = 0.0  # Monitor source performance (reverse testing)
    epoc_residual_norm = 0.0
    epoc_fda_effect_input = 0.0   # Track FDA effect on input translation
    epoc_fda_effect_label = 0.0   # Track FDA effect on label translation
    N_train = 0
    
    # Initialize feature variables
    features_src = None
    features_tgt = None
    
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_tgt.total_batches):  # ← KEY: Use TARGET batches as primary
        # Get data - TARGET is primary, SOURCE is for style reference only
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        # NOTE: We DON'T use y_tgt (target labels) - we don't have them!
        N_train += x_tgt.shape[0]  # ← Count target samples as primary

        # Preprocess data to real format (NO SCALING YET - RAW FDA FIRST!)
        # SOURCE data (for FDA style reference)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        # TARGET data (primary content source)
        x_tgt_real = complx2real(x_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: RAW FDA INPUT TRANSLATION (Target→Source INPUT) ============
        # Convert raw data to complex for FDA
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])  # ← TARGET content
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])  # ← SOURCE INPUT style
        
        # Extract Delay-Doppler representations for INPUT translation
        tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)      # ← Target content
        src_input_amplitude, src_input_phase = F_extract_DD(x_src_complex)  # ← Source INPUT style
        
        # Mix amplitudes: Target content + Source INPUT style
        mixed_amplitude_input = fda_mix_pixels(tgt_amplitude, src_input_amplitude, fda_win_h, fda_win_w)
        
        # Keep target phase (content preservation)
        mixed_complex_dd_input = apply_phase_to_amplitude(mixed_amplitude_input, tgt_phase)
        
        # Convert back to Time-Frequency domain
        x_fda_input_complex = F_inverse_DD(mixed_complex_dd_input)
        x_fda_input_real = tf.stack([tf.math.real(x_fda_input_complex), tf.math.imag(x_fda_input_complex)], axis=-1)
        
        # Track FDA effect on input translation
        fda_difference_input = tf.reduce_mean(tf.abs(x_fda_input_real - x_tgt_real))
        epoc_fda_effect_input += fda_difference_input.numpy() * x_tgt.shape[0]
        
        # ============ STEP 2: RAW FDA LABEL TRANSLATION (Target→Source LABEL) ============
        # Convert SOURCE LABELS to complex for FDA style reference
        y_src_complex = tf.complex(y_src_real[:,:,:,0], y_src_real[:,:,:,1])  # ← SOURCE LABEL style
        # Target content already available as x_tgt_complex
        
        # Extract Delay-Doppler representations for LABEL translation
        # Target content + Source LABEL style
        src_label_amplitude, src_label_phase = F_extract_DD(y_src_complex)  # ← Source LABEL style
        # tgt_amplitude, tgt_phase already computed above
        
        # Mix amplitudes: Target content + Source LABEL style  
        mixed_amplitude_label = fda_mix_pixels(tgt_amplitude, src_label_amplitude, fda_win_h, fda_win_w)
        
        # Keep target phase (content preservation for labels)
        mixed_complex_dd_label = apply_phase_to_amplitude(mixed_amplitude_label, tgt_phase)
        
        # Convert back to Time-Frequency domain
        y_fda_label_complex = F_inverse_DD(mixed_complex_dd_label)
        y_fda_label_real = tf.stack([tf.math.real(y_fda_label_complex), tf.math.imag(y_fda_label_complex)], axis=-1)
        
        # Track FDA effect on label translation
        fda_difference_label = tf.reduce_mean(tf.abs(y_fda_label_real - x_tgt_real))
        epoc_fda_effect_label += fda_difference_label.numpy() * x_tgt.shape[0]
        
        # ============ STEP 3: SCALE FDA OUTPUTS WITH SHARED PARAMETERS ============
        # Scale FDA-mixed input to [-1,1] and get scaling parameters
        x_fda_input_numpy = x_fda_input_real.numpy()
        x_fda_input_scaled, fda_min, fda_max = minmaxScaler(x_fda_input_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        # Scale FDA-mixed labels using THE SAME scaling parameters for consistency
        batch_size = y_fda_label_real.shape[0]
        
        # Handle scalar vs array FDA parameters for labels
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))
        else:
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        # Scale FDA-mixed labels with FDA input's scaling parameters
        y_fda_label_numpy = y_fda_label_real.numpy()
        y_fda_label_scaled, _, _ = minmaxScaler(y_fda_label_numpy, min_pre=fda_min_array, max_pre=fda_max_array, 
                                            lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE TARGET DATA WITH FDA PARAMETERS FOR HYBRID TRAINING ============
        # Scale original target data with FDA parameters for consistency (for hybrid training)
        x_tgt_scaled_with_fda_params, _, _ = minmaxScaler(x_tgt_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # ============ STEP 5: CNN TRAINING with RESIDUAL LEARNING ============
        with tf.GradientTape() as tape:
            batch_size_int = x_fda_input_scaled.shape[0]
            
            if fda_weight < 1.0:
                # Hybrid training: Part FDA-translated, part original target (both using FDA scaling parameters)
                fda_samples = int(batch_size_int * fda_weight)
                remaining_samples = batch_size_int - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    # Split batch for hybrid training
                    x_train_fda = x_fda_input_scaled[:fda_samples]                        # FDA-translated target→source input
                    y_train_fda = y_fda_label_scaled[:fda_samples]                        # FDA-translated target→source labels
                    
                    x_train_tgt = x_tgt_scaled_with_fda_params[fda_samples:]              # Target scaled with FDA params
                    y_train_tgt = x_tgt_scaled_with_fda_params[fda_samples:]              # Target as its own pseudo-label
                    
                    # Combine for training (both in FDA scaling space)
                    x_train_combined = tf.concat([x_train_fda, x_train_tgt], axis=0)
                    y_train_combined = tf.concat([y_train_fda, y_train_tgt], axis=0)      # ← FDA labels + target pseudo-labels
                    
                    # Forward pass on combined input
                    residual_combined, features_src = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    # Pure target training (scaled with FDA parameters)
                    residual_tgt, features_src = model_cnn(x_tgt_scaled_with_fda_params, training=True, return_features=True)
                    x_corrected_combined = x_tgt_scaled_with_fda_params + residual_tgt
                    y_train_combined = x_tgt_scaled_with_fda_params  # Target as its own pseudo-label
                    residual_combined = residual_tgt
                else:
                    # Pure FDA training (fully translated pairs: target→source input & target→source labels)
                    residual_fda, features_src = model_cnn(x_fda_input_scaled, training=True, return_features=True)
                    x_corrected_combined = x_fda_input_scaled + residual_fda
                    y_train_combined = y_fda_label_scaled  # ← FDA-translated target→source labels
                    residual_combined = residual_fda
                
                # Estimation loss: Both inputs and labels in FDA scaling space
                est_loss = loss_fn_est(y_train_combined, x_corrected_combined)
                    
            else:
                # Pure FDA training: Use FULLY translated pairs (target→source input & target→source labels)
                residual_fda, features_src = model_cnn(x_fda_input_scaled, training=True, return_features=True)
                x_corrected_fda = x_fda_input_scaled + residual_fda
                residual_combined = residual_fda
                
                # Estimation loss: FDA-corrected input vs FDA-translated labels
                est_loss = loss_fn_est(y_fda_label_scaled, x_corrected_fda)
            
            # Residual regularization: Encourage small, meaningful corrections
            residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # Total loss
            total_loss = est_weight * est_loss + residual_reg + smoothness_loss
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # ============ MONITOR SOURCE PERFORMANCE (reverse testing) ============
        # Test how well the model (trained on translated target) performs on source domain
        if batch_idx < loader_H_true_train_src.total_batches:
            # Scale source with its own parameters for monitoring (traditional approach)
            x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
            y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)
            
            residual_src, features_tgt = model_cnn(x_scaled_src, training=False, return_features=True)
            x_corrected_src = x_scaled_src + residual_src
            est_loss_src = loss_fn_est(y_scaled_src, x_corrected_src)  # ← We DO have source labels for testing
            epoc_loss_est_source += est_loss_src.numpy() * x_src.shape[0]
        
        # === Save features if required ===
        if save_features and features_src is not None:
            # Save features from target-primary training
            features_np_source = features_src[-1].numpy()
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
                
            if features_tgt is not None:
                features_np_target = features_tgt[-1].numpy()
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
        
        # Single optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_tgt.shape[0]  # ← Use target batch size
        epoc_loss_est += est_loss.numpy() * x_tgt.shape[0]      # ← Primary training loss
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_tgt.shape[0]
    
    # Close feature files
    if save_features:    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_source = epoc_loss_est_source / N_train if epoc_loss_est_source > 0 else 0.0  # ← Source monitoring
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect_input = epoc_fda_effect_input / N_train
    avg_fda_effect_label = epoc_fda_effect_label / N_train
    
    # Print enhanced FDA statistics for target→source with separate input/label translations
    print(f"    Target→Source FULL FDA Translation residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA input translation effect (Target→Source INPUT): {avg_fda_effect_input:.6f}")
    print(f"    FDA label translation effect (Target→Source LABEL): {avg_fda_effect_label:.6f}")
    print(f"    Source domain testing performance: {avg_loss_est_source:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    print(f"    Training: Target→Source input style + Target→Source label style")
    
    # Return compatible structure (note the role reversal)
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,                 # Target-based training loss
        avg_epoc_loss_domain=0.0,                       # No domain loss - FDA handles domain gap at data level
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_source,   # ← SOURCE becomes "target" for monitoring (role reversal)
        features_source=features_src[-1] if features_src is not None else None,
        film_features_source=features_src[-1] if features_src is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )
    
##    
def train_step_cnn_residual_FDAfullTranslation1_coral(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                            save_features=False, nsymb=14, weights=None, linear_interp=False,
                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0, coral_loss_fn=None, flag_residual_reg=True):
    """
    CNN-only residual training step combining FDA Full Translation 1 + CORAL domain adaptation
    
    Workflow:
    1. RAW FDA FULL Translation: Source input → Target style input AND 
                                Source labels → Target style labels
    2. Scale FDA outputs with shared parameters for consistency
    3. CORAL Loss: Align features between FDA-translated and raw target domains
    4. Residual learning: Train CNN to predict corrections
    
    Training pairs:
    - Input: Source content + Target style (FDA-mixed input, then scaled)
    - Label: Source labels + Target INPUT style (FDA-mixed labels using TARGET INPUT as style)
    - Feature alignment: CORAL loss between FDA features and raw target features
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of loaders
        loss_fn: tuple of loss functions (only first one used)
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features
        nsymb: number of symbols  
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height (default 13)
        fda_win_w: FDA window width (default 3)
        fda_weight: Weight for FDA vs source training (1.0 = 100% FDA)
        coral_loss_fn: CORAL loss function instance (if None, will create default)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize CORAL loss if not provided
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_coral = 0.0
    epoc_residual_norm = 0.0
    epoc_fda_effect_input = 0.0   # Track FDA effect on inputs
    epoc_fda_effect_label = 0.0   # Track FDA effect on labels
    N_train = 0
    
    # Initialize feature variables
    features_fda = None
    features_tgt_raw = None
    
    if save_features and domain_weight != 0:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()  # Not used in label translation (we don't have target labels)
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET - RAW FDA FIRST!)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        # Note: We don't process y_tgt since we don't have target labels for style reference

        # ============ STEP 1: RAW FDA INPUT TRANSLATION (Source→Target INPUT) ============
        # Convert raw data to complex for FDA
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        # Extract Delay-Doppler representations for RAW INPUTS
        src_amplitude_input, src_phase_input = F_extract_DD(x_src_complex)
        tgt_amplitude_input, tgt_phase_input = F_extract_DD(x_tgt_complex)
        
        # Mix amplitudes: Target style in center, Source style in outer regions
        mixed_amplitude_input = fda_mix_pixels(src_amplitude_input, tgt_amplitude_input, fda_win_h, fda_win_w)
        
        # Keep source phase (content preservation)
        mixed_complex_dd_input = apply_phase_to_amplitude(mixed_amplitude_input, src_phase_input)
        
        # Convert back to Time-Frequency domain
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_input)
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        # Track FDA effect on inputs
        fda_difference_input = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect_input += fda_difference_input.numpy() * x_src.shape[0]
        
        # ============ STEP 2: RAW FDA LABEL TRANSLATION (Source Labels→Target INPUT Style) ============
        # Convert raw labels to complex for FDA
        y_src_complex = tf.complex(y_src_real[:,:,:,0], y_src_real[:,:,:,1])
        # KEY: Use TARGET INPUT (not target labels) as style reference for label translation
        # x_tgt_complex already computed above
        
        # Extract Delay-Doppler representations: SOURCE LABELS + TARGET INPUT STYLE
        src_amplitude_label, src_phase_label = F_extract_DD(y_src_complex)              # Source labels content
        tgt_amplitude_style, tgt_phase_style = F_extract_DD(x_tgt_complex)              # Target INPUT style (not labels!)
        
        # Mix label amplitudes: Use TARGET INPUT STYLE for labels (since we don't have target labels)
        mixed_amplitude_label = fda_mix_pixels(src_amplitude_label, tgt_amplitude_style, fda_win_h, fda_win_w)
        
        # Keep source label phase (content preservation for labels)
        mixed_complex_dd_label = apply_phase_to_amplitude(mixed_amplitude_label, src_phase_label)
        
        # Convert back to Time-Frequency domain
        y_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_label)
        y_fda_mixed_real = tf.stack([tf.math.real(y_fda_mixed_complex), tf.math.imag(y_fda_mixed_complex)], axis=-1)
        
        # Track FDA effect on labels
        fda_difference_label = tf.reduce_mean(tf.abs(y_fda_mixed_real - y_src_real))
        epoc_fda_effect_label += fda_difference_label.numpy() * x_src.shape[0]
        
        # ============ STEP 3: SCALE FDA OUTPUTS WITH SHARED PARAMETERS ============
        # Scale FDA-mixed input to [-1,1] and get scaling parameters
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        # Scale FDA-mixed labels using THE SAME scaling parameters for consistency
        batch_size = y_fda_mixed_real.shape[0]
        
        # Handle scalar vs array FDA parameters for labels
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))
        else:
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        # Scale FDA-mixed labels with FDA input's scaling parameters
        y_fda_mixed_numpy = y_fda_mixed_real.numpy()
        y_fda_scaled, _, _ = minmaxScaler(y_fda_mixed_numpy, min_pre=fda_min_array, max_pre=fda_max_array, 
                                        lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE TARGET FOR CORAL COMPARISON ============
        # Scale target input with its own parameters for CORAL feature comparison
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 5: SCALE SOURCE DATA WITH FDA PARAMETERS FOR HYBRID TRAINING ============
        # Scale source input and labels with FDA parameters for consistency (for hybrid training)
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)
        y_src_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # ============ STEP 6: DUAL FLOW CNN TRAINING + CORAL ALIGNMENT ============
        with tf.GradientTape() as tape:
            # === FLOW 1: FDA-TRANSLATED INPUT (Main training flow) ===
            batch_size_int = x_fda_scaled.shape[0]
            
            if fda_weight < 1.0:
                # Hybrid training: Part FDA-translated, part source (both using FDA scaling parameters)
                fda_samples = int(batch_size_int * fda_weight)
                remaining_samples = batch_size_int - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    # Split batch for hybrid training
                    x_train_fda = x_fda_scaled[:fda_samples]                           # FDA-translated inputs
                    y_train_fda = y_fda_scaled[:fda_samples]                           # FDA-translated labels (using TARGET INPUT style)
                    
                    x_train_src = x_src_scaled_with_fda_params[fda_samples:]          # Source scaled with FDA params
                    y_train_src = y_src_scaled_with_fda_params[fda_samples:]          # Source labels scaled with FDA params
                    
                    # Combine for training (both in FDA scaling space)
                    x_train_combined = tf.concat([x_train_fda, x_train_src], axis=0)
                    y_train_combined = tf.concat([y_train_fda, y_train_src], axis=0)   # FDA labels + source labels
                    
                    # Forward pass on combined input
                    residual_combined, features_fda = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    # Pure source training (scaled with FDA parameters)
                    residual_src, features_fda = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                    x_corrected_combined = x_src_scaled_with_fda_params + residual_src
                    y_train_combined = y_src_scaled_with_fda_params  # Source labels in FDA scaling space
                    residual_combined = residual_src
                else:
                    # Pure FDA training (fully translated pairs using TARGET INPUT style for labels)
                    residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                    x_corrected_combined = x_fda_scaled + residual_fda
                    y_train_combined = y_fda_scaled  # FDA-translated labels using TARGET INPUT style
                    residual_combined = residual_fda
                
                # Estimation loss: Both inputs and labels in FDA scaling space
                est_loss = loss_fn_est(y_train_combined, x_corrected_combined)
                    
            else:
                # Pure FDA training: Use FULLY translated pairs (TARGET INPUT style for labels)
                residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                x_corrected_fda = x_fda_scaled + residual_fda
                residual_combined = residual_fda
                
                # Estimation loss: FDA-corrected input vs FDA-translated labels (labels use TARGET INPUT style)
                est_loss = loss_fn_est(y_fda_scaled, x_corrected_fda)
            
            # === FLOW 2: RAW TARGET INPUT (For CORAL alignment) ===
            # Get features from raw target domain for CORAL comparison
            residual_tgt_raw, features_tgt_raw = model_cnn(x_scaled_tgt, training=False, return_features=True)
            # Note: training=False to prevent this flow from affecting main training gradients
            
            # === CORAL LOSS: Align FDA Features with Raw Target Features ===
            if domain_weight > 0:
                coral_loss = coral_loss_fn(features_fda, features_tgt_raw)
            else:
                coral_loss = 0.0
            
            # === REGULARIZATION LOSSES ===
            # Residual regularization: Encourage small, meaningful corrections
            if flag_residual_reg:
                residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            else:
                residual_reg = 0.0
            
            # Smoothness loss (optional)
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # === TOTAL LOSS ===
            total_loss = (est_weight * est_loss + 
                        domain_weight * coral_loss +   # CORAL feature alignment
                        residual_reg + 
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # ============ MONITOR TARGET PERFORMANCE (OPTIONAL) ============
        # Monitor target domain performance using traditional scaling
        y_tgt_real = complx2real(y_tgt)
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        x_corrected_tgt = x_scaled_tgt + residual_tgt_raw
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === Save features if required ===
        if save_features and domain_weight != 0:
            # Save FDA features
            features_np_fda = features_fda[-1].numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features',
                    data=features_np_fda,
                    maxshape=(None,) + features_np_fda.shape[1:],
                    chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_fda.shape[0], axis=0)
                features_dataset_source[-features_np_fda.shape[0]:] = features_np_fda
                
            # Save raw target features
            features_np_target = features_tgt_raw[-1].numpy()
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
        
        # Single optimizer update
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_coral += coral_loss.numpy() * x_src.shape[0] if domain_weight > 0 else 0.0
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features and domain_weight != 0:    
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_coral = epoc_loss_coral / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect_input = epoc_fda_effect_input / N_train
    avg_fda_effect_label = epoc_fda_effect_label / N_train
    
    # Print enhanced FDA + CORAL statistics
    print(f"    FDA Full Translation + CORAL residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA input translation effect (Source→Target INPUT): {avg_fda_effect_input:.6f}")
    print(f"    FDA label translation effect (Source Labels→Target INPUT style): {avg_fda_effect_label:.6f}")
    print(f"    CORAL feature alignment loss: {avg_loss_coral:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    print(f"    Training: Source→Target input style + Source labels→Target INPUT style + CORAL alignment")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_coral,  # CORAL alignment loss
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_fda[-1] if features_fda is not None else None,
        film_features_source=features_fda[-1] if features_fda is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )
    


def train_step_cnn_residual_FDAfullTranslation1_jmmd(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                            save_features=False, nsymb=14, weights=None, linear_interp=False,
                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0, jmmd_loss_fn=None, flag_residual_reg=True):
    """
    CNN-only residual training step combining FDA Full Translation 1 + JMMD domain adaptation
    
    Workflow:
    1. RAW FDA FULL Translation: Source input → Target style input AND 
                                Source labels → Target style labels
    2. Scale FDA outputs with shared parameters for consistency
    3. JMMD Loss: Align features between FDA-translated and raw target domains
    4. Residual learning: Train CNN to predict corrections
    
    Training pairs:
    - Input: Source content + Target style (FDA-mixed input, then scaled)
    - Label: Source labels + Target INPUT style (FDA-mixed labels using TARGET INPUT as style)
    - Feature alignment: JMMD loss between FDA features and raw target features
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of loaders
        loss_fn: tuple of loss functions (only first one used)
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features
        nsymb: number of symbols  
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height (default 13)
        fda_win_w: FDA window width (default 3)
        fda_weight: Weight for FDA vs source training (1.0 = 100% FDA)
        jmmd_loss_fn: JMMD loss function instance (if None, will create default)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize JMMD loss if not provided
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_jmmd = 0.0
    epoc_residual_norm = 0.0
    epoc_fda_effect_input = 0.0
    epoc_fda_effect_label = 0.0
    N_train = 0
    
    # Initialize feature variables
    features_fda = None
    features_tgt_raw = None
    
    if save_features and domain_weight != 0:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None

    def _pool_feature_list(feat_list):
        pooled = []
        for f in feat_list:
            if len(f.shape) == 4:   # [B, H, W, C]
                pooled.append(tf.reduce_mean(f, axis=[1, 2]))
            elif len(f.shape) == 3: # [B, H, C] or [B, W, C]
                pooled.append(tf.reduce_mean(f, axis=1))
            elif len(f.shape) == 2: # [B, C]
                pooled.append(f)
            else:
                pooled.append(tf.reshape(f, [tf.shape(f)[0], -1]))
        return pooled
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()  # Not used in label translation
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: RAW FDA INPUT TRANSLATION (Source→Target INPUT) ============
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        src_amplitude_input, src_phase_input = F_extract_DD(x_src_complex)
        tgt_amplitude_input, tgt_phase_input = F_extract_DD(x_tgt_complex)
        mixed_amplitude_input = fda_mix_pixels(src_amplitude_input, tgt_amplitude_input, fda_win_h, fda_win_w)
        mixed_complex_dd_input = apply_phase_to_amplitude(mixed_amplitude_input, src_phase_input)
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_input)
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        fda_difference_input = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect_input += fda_difference_input.numpy() * x_src.shape[0]
        
        # ============ STEP 2: RAW FDA LABEL TRANSLATION (Source Labels→Target INPUT Style) ============
        y_src_complex = tf.complex(y_src_real[:,:,:,0], y_src_real[:,:,:,1])
        src_amplitude_label, src_phase_label = F_extract_DD(y_src_complex)
        tgt_amplitude_style, tgt_phase_style = F_extract_DD(x_tgt_complex)
        mixed_amplitude_label = fda_mix_pixels(src_amplitude_label, tgt_amplitude_style, fda_win_h, fda_win_w)
        mixed_complex_dd_label = apply_phase_to_amplitude(mixed_amplitude_label, src_phase_label)
        y_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_label)
        y_fda_mixed_real = tf.stack([tf.math.real(y_fda_mixed_complex), tf.math.imag(y_fda_mixed_complex)], axis=-1)
        
        fda_difference_label = tf.reduce_mean(tf.abs(y_fda_mixed_real - y_src_real))
        epoc_fda_effect_label += fda_difference_label.numpy() * x_src.shape[0]
        
        # ============ STEP 3: SCALE FDA OUTPUTS WITH SHARED PARAMETERS ============
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        batch_size = y_fda_mixed_real.shape[0]
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))
        else:
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        y_fda_mixed_numpy = y_fda_mixed_real.numpy()
        y_fda_scaled, _, _ = minmaxScaler(y_fda_mixed_numpy, min_pre=fda_min_array, max_pre=fda_max_array, 
                                        lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE TARGET FOR JMMD COMPARISON ============
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 5: SCALE SOURCE DATA WITH FDA PARAMETERS FOR HYBRID TRAINING ============
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)
        y_src_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # ============ STEP 6: DUAL FLOW CNN TRAINING + JMMD ALIGNMENT ============
        with tf.GradientTape() as tape:
            batch_size_int = x_fda_scaled.shape[0]
            
            if fda_weight < 1.0:
                fda_samples = int(batch_size_int * fda_weight)
                remaining_samples = batch_size_int - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    x_train_fda = x_fda_scaled[:fda_samples]
                    y_train_fda = y_fda_scaled[:fda_samples]
                    x_train_src = x_src_scaled_with_fda_params[fda_samples:]
                    y_train_src = y_src_scaled_with_fda_params[fda_samples:]
                    
                    x_train_combined = tf.concat([x_train_fda, x_train_src], axis=0)
                    y_train_combined = tf.concat([y_train_fda, y_train_src], axis=0)
                    
                    residual_combined, features_fda = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    residual_src, features_fda = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                    x_corrected_combined = x_src_scaled_with_fda_params + residual_src
                    y_train_combined = y_src_scaled_with_fda_params
                    residual_combined = residual_src
                else:
                    residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                    x_corrected_combined = x_fda_scaled + residual_fda
                    y_train_combined = y_fda_scaled
                    residual_combined = residual_fda
                
                est_loss = loss_fn_est(y_train_combined, x_corrected_combined)
                    
            else:
                residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                x_corrected_fda = x_fda_scaled + residual_fda
                residual_combined = residual_fda
                est_loss = loss_fn_est(y_fda_scaled, x_corrected_fda)
            
            residual_tgt_raw, features_tgt_raw = model_cnn(x_scaled_tgt, training=False, return_features=True)
            
            if domain_weight > 0:
                jmmd_loss = jmmd_loss_fn(features_fda, features_tgt_raw)
            else:
                jmmd_loss = 0.0
            
            if flag_residual_reg:
                residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            else:
                residual_reg = 0.0
            
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            total_loss = (est_weight * est_loss + 
                        domain_weight * jmmd_loss +   
                        residual_reg + 
                        smoothness_loss)
            
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        y_tgt_real = complx2real(y_tgt)
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        x_corrected_tgt = x_scaled_tgt + residual_tgt_raw
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        if save_features and domain_weight != 0:
            features_np_fda = features_fda[-1].numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features',
                    data=features_np_fda,
                    maxshape=(None,) + features_np_fda.shape[1:],
                    chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_fda.shape[0], axis=0)
                features_dataset_source[-features_np_fda.shape[0]:] = features_np_fda
                
            features_np_target = features_tgt_raw[-1].numpy()
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
        
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0] if domain_weight > 0 else 0.0
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    if save_features and domain_weight != 0:    
        features_h5_source.close()
        features_h5_target.close()
    
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect_input = epoc_fda_effect_input / N_train
    avg_fda_effect_label = epoc_fda_effect_label / N_train
    
    print(f"    FDA Full Translation + JMMD residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA input translation effect (Source→Target INPUT): {avg_fda_effect_input:.6f}")
    print(f"    FDA label translation effect (Source Labels→Target INPUT style): {avg_fda_effect_label:.6f}")
    print(f"    JMMD feature alignment loss: {avg_loss_jmmd:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_jmmd,
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_fda[-1] if features_fda is not None else None,
        film_features_source=features_fda[-1] if features_fda is not None else None,
        avg_epoc_loss_d=0.0
    )

def val_step_cnn_residual_jmmd(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, weights=None, 
                                jmmd_loss_fn=None, linear_interp=False, return_H_gen=False):
    """
    Validation step for CNN-only residual learning with JMMD domain adaptation (no discriminator)
    
    Args:
        model_cnn: CNN model (CNNGenerator instance, not GAN wrapper)
        loader_H: tuple of validation loaders
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        jmmd_loss_fn: JMMD loss function instance (if None, will create default)
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize JMMD loss if not provided
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_jmmd_loss = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0  # Track residual magnitude during validation
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # Get and preprocess data (same as CORAL version)
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocessing (same as original)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
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

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
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

        # Track residual magnitudes during validation
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === JMMD Loss (on residual features) ===
        if domain_weight > 0:
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # === Smoothness loss (on corrected channels) ===
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

        # === Save H samples ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Print residual statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    Validation JMMD loss: {avg_jmmd_loss:.6f}")
    
    # Domain accuracy placeholders (compatible with existing code)
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss (no discriminator component)
    avg_total_loss = est_weight * avg_loss_est + domain_weight * avg_jmmd_loss + avg_smoothness_loss

    # Return compatible structure
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_domain_loss': avg_jmmd_loss,  # Use JMMD loss for compatibility with plotting
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

def train_step_cnn_residual_FDAfullTranslation1_coral_domainAware(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                            save_features=False, nsymb=14, weights=None, linear_interp=False,
                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0, coral_loss_fn=None,
                                            residual_consistency_weight=0.2, flag_residual_reg=True):
    """
    CNN-only residual training step combining FDA Full Translation 1 + CORAL + Domain-Aware Consistency
    
    Workflow:
    1. RAW FDA FULL Translation: Source input → Target style input AND 
                                Source labels → Target INPUT style labels
    2. Scale FDA outputs with shared parameters for consistency
    3. CORAL Loss: Align features between FDA-translated and raw target domains
    4. Domain-Aware Consistency: Similar improvement ratios and residual patterns across domains
    5. Residual learning: Train CNN to predict corrections
    
    Training pairs:
    - Input: Source content + Target style (FDA-mixed input, then scaled)
    - Label: Source labels + Target INPUT style (FDA-mixed labels using TARGET INPUT as style)
    - Feature alignment: CORAL loss between FDA features and raw target features
    - Domain consistency: Align improvement patterns and residual behaviors
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of loaders
        loss_fn: tuple of loss functions (only first one used)
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features
        nsymb: number of symbols  
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height (default 13)
        fda_win_w: FDA window width (default 3)
        fda_weight: Weight for FDA vs source training (1.0 = 100% FDA)
        coral_loss_fn: CORAL loss function instance (if None, will create default)
        consistency_weight: Weight for direct consistency (default 0.3)
        residual_consistency_weight: Weight for residual pattern consistency (default 0.2)
        improvement_consistency_weight: Weight for improvement ratio consistency (default 0.1)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize CORAL loss if not provided
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_coral = 0.0
    epoc_loss_improvement_consistency = 0.0   
    epoc_loss_residual_consistency = 0.0
    epoc_loss_direct_consistency = 0.0
    epoc_residual_norm = 0.0
    epoc_fda_effect_input = 0.0
    epoc_fda_effect_label = 0.0
    N_train = 0
    
    # Initialize feature variables
    features_fda = None
    features_tgt_raw = None
    
    if save_features and domain_weight != 0:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()  # Not used in label translation
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET - RAW FDA FIRST!)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: RAW FDA INPUT TRANSLATION (Source→Target INPUT) ============
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        src_amplitude_input, src_phase_input = F_extract_DD(x_src_complex)
        tgt_amplitude_input, tgt_phase_input = F_extract_DD(x_tgt_complex)
        mixed_amplitude_input = fda_mix_pixels(src_amplitude_input, tgt_amplitude_input, fda_win_h, fda_win_w)
        mixed_complex_dd_input = apply_phase_to_amplitude(mixed_amplitude_input, src_phase_input)
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_input)
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        fda_difference_input = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect_input += fda_difference_input.numpy() * x_src.shape[0]
        
        # ============ STEP 2: RAW FDA LABEL TRANSLATION (Source Labels→Target INPUT Style) ============
        y_src_complex = tf.complex(y_src_real[:,:,:,0], y_src_real[:,:,:,1])
        src_amplitude_label, src_phase_label = F_extract_DD(y_src_complex)
        tgt_amplitude_style, tgt_phase_style = F_extract_DD(x_tgt_complex)
        mixed_amplitude_label = fda_mix_pixels(src_amplitude_label, tgt_amplitude_style, fda_win_h, fda_win_w)
        mixed_complex_dd_label = apply_phase_to_amplitude(mixed_amplitude_label, src_phase_label)
        y_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_label)
        y_fda_mixed_real = tf.stack([tf.math.real(y_fda_mixed_complex), tf.math.imag(y_fda_mixed_complex)], axis=-1)
        
        fda_difference_label = tf.reduce_mean(tf.abs(y_fda_mixed_real - y_src_real))
        epoc_fda_effect_label += fda_difference_label.numpy() * x_src.shape[0]
        
        # ============ STEP 3: SCALE FDA OUTPUTS WITH SHARED PARAMETERS ============
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        batch_size = y_fda_mixed_real.shape[0]
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))
        else:
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        y_fda_mixed_numpy = y_fda_mixed_real.numpy()
        y_fda_scaled, _, _ = minmaxScaler(y_fda_mixed_numpy, min_pre=fda_min_array, max_pre=fda_max_array, 
                                        lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE TARGET FOR CORAL & CONSISTENCY COMPARISON ============
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 5: SCALE SOURCE DATA WITH FDA PARAMETERS FOR HYBRID TRAINING ============
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)
        y_src_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # ============ STEP 6: DUAL FLOW CNN TRAINING + CORAL + DOMAIN-AWARE CONSISTENCY ============
        with tf.GradientTape() as tape:
            # === FLOW 1: FDA-TRANSLATED INPUT (Main training flow) ===
            batch_size_int = x_fda_scaled.shape[0]
            
            if fda_weight < 1.0:
                fda_samples = int(batch_size_int * fda_weight)
                remaining_samples = batch_size_int - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    x_train_fda = x_fda_scaled[:fda_samples]
                    y_train_fda = y_fda_scaled[:fda_samples]
                    x_train_src = x_src_scaled_with_fda_params[fda_samples:]
                    y_train_src = y_src_scaled_with_fda_params[fda_samples:]
                    
                    x_train_combined = tf.concat([x_train_fda, x_train_src], axis=0)
                    y_train_combined = tf.concat([y_train_fda, y_train_src], axis=0)
                    
                    residual_combined, features_fda = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    residual_src, features_fda = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                    x_corrected_combined = x_src_scaled_with_fda_params + residual_src
                    y_train_combined = y_src_scaled_with_fda_params
                    residual_combined = residual_src
                else:
                    residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                    x_corrected_combined = x_fda_scaled + residual_fda
                    y_train_combined = y_fda_scaled
                    residual_combined = residual_fda
                
                est_loss = loss_fn_est(y_train_combined, x_corrected_combined)
                    
            else:
                # Pure FDA training
                residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                x_corrected_fda = x_fda_scaled + residual_fda
                residual_combined = residual_fda
                est_loss = loss_fn_est(y_fda_scaled, x_corrected_fda)
            
            # === FLOW 2: RAW TARGET INPUT (For CORAL & consistency alignment) ===
            residual_tgt_raw, features_tgt_raw = model_cnn(x_scaled_tgt, training=False, return_features=True)
            x_corrected_tgt = x_scaled_tgt + residual_tgt_raw
            
            # === CORAL LOSS: Align FDA Features with Raw Target Features ===
            if domain_weight > 0:
                coral_loss = coral_loss_fn(features_fda, features_tgt_raw)
            else:
                coral_loss = 0.0
            
            # === DOMAIN-AWARE CONSISTENCY ===            
            # RESIDUAL PATTERN CONSISTENCY 
            if residual_consistency_weight > 0:
                # Normalize residuals by input magnitude to account for domain differences
                fda_input_magnitude = tf.abs(x_fda_scaled) + 1e-8
                tgt_input_magnitude = tf.abs(x_scaled_tgt) + 1e-8
                
                # Normalized residual patterns
                fda_residual_normalized = residual_combined / fda_input_magnitude
                tgt_residual_normalized = residual_tgt_raw / tgt_input_magnitude
                
                # Consistency: similar relative correction patterns
                residual_pattern_consistency = tf.reduce_mean(tf.square(fda_residual_normalized - tgt_residual_normalized))
            else:
                residual_pattern_consistency = 0.0
            
            # Residual regularization: Encourage small, meaningful corrections
            if flag_residual_reg:
                residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            else:
                residual_reg = 0.0
                
            # === SMOOTHNESS LOSS ===
            if temporal_weight != 0 or frequency_weight != 0:
                # Check if we have hybrid training variables
                if fda_weight < 1.0:
                    # Try to access fda_samples safely
                    try:
                        if 'fda_samples' in locals() and fda_samples > 0 and (batch_size_int - fda_samples) > 0:
                            smoothness_fda = compute_total_smoothness_loss(x_corrected_combined[:fda_samples], 
                                                                        temporal_weight=temporal_weight, 
                                                                        frequency_weight=frequency_weight)
                            smoothness_src = compute_total_smoothness_loss(x_corrected_combined[fda_samples:], 
                                                                        temporal_weight=temporal_weight, 
                                                                        frequency_weight=frequency_weight)
                            smoothness_loss = (smoothness_fda + smoothness_src) / 2
                        else:
                            # Fallback to full batch
                            smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                        temporal_weight=temporal_weight, 
                                                                        frequency_weight=frequency_weight)
                    except:
                        # Safe fallback
                        smoothness_loss = compute_total_smoothness_loss(x_corrected_combined if fda_weight < 1.0 else x_corrected_fda, 
                                                                    temporal_weight=temporal_weight, 
                                                                    frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # === TOTAL LOSS (WITH DOMAIN-AWARE CONSISTENCY) ===
            total_loss = (est_weight * est_loss + 
                        domain_weight * coral_loss +
                        residual_consistency_weight * residual_pattern_consistency +    
                        residual_reg +                   
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # === MONITOR TARGET PERFORMANCE (separate target evaluation) ===
        y_tgt_real = complx2real(y_tgt)
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === Save features if required ===
        if save_features and domain_weight != 0:
            features_np_fda = features_fda[-1].numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features', data=features_np_fda,
                    maxshape=(None,) + features_np_fda.shape[1:], chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_fda.shape[0], axis=0)
                features_dataset_source[-features_np_fda.shape[0]:] = features_np_fda
                
            features_np_target = features_tgt_raw[-1].numpy()
            if features_dataset_target is None:
                features_dataset_target = features_h5_target.create_dataset(
                    'features', data=features_np_target,
                    maxshape=(None,) + features_np_target.shape[1:], chunks=True
                )
            else:
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
                features_dataset_target[-features_np_target.shape[0]:] = features_np_target
        
        # Update model
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_coral += coral_loss.numpy() * x_src.shape[0] if domain_weight > 0 else 0.0
        epoc_loss_residual_consistency += residual_pattern_consistency.numpy() * x_src.shape[0]    
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features and domain_weight != 0:
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_coral = epoc_loss_coral / N_train
    avg_loss_improvement_consistency = epoc_loss_improvement_consistency / N_train  
    avg_loss_residual_consistency = epoc_loss_residual_consistency / N_train
    avg_loss_direct_consistency = epoc_loss_direct_consistency / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect_input = epoc_fda_effect_input / N_train
    avg_fda_effect_label = epoc_fda_effect_label / N_train
    
    # Print enhanced statistics with correct names
    print(f"    FDA Full Translation + CORAL + Domain-Aware residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA input translation effect (Source→Target INPUT): {avg_fda_effect_input:.6f}")
    print(f"    FDA label translation effect (Source Labels→Target INPUT style): {avg_fda_effect_label:.6f}")
    print(f"    CORAL feature alignment loss: {avg_loss_coral:.6f}")
    print(f"    Improvement ratio consistency: {avg_loss_improvement_consistency:.6f}")  
    print(f"    Residual pattern consistency: {avg_loss_residual_consistency:.6f}")
    print(f"    Direct consistency: {avg_loss_direct_consistency:.6f}")  
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_coral  + avg_loss_residual_consistency,  # consider adding residual_reg if needed
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_fda[-1] if features_fda is not None else None,
        film_features_source=features_fda[-1] if features_fda is not None else None,
        avg_epoc_loss_d=0.0  
    )
def val_step_cnn_residual_FDAfullTranslation1_coral_domainAware(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, 
                                                            weights=None, linear_interp=False, return_H_gen=False,
                                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0, coral_loss_fn=None,
                                                            residual_consistency_weight=0.2):
    """
    Validation step for CNN-only residual learning with FDA Full Translation 1 + CORAL + Domain-Aware Consistency
    
    Validates model performance using DIRECT validation (no FDA translation during validation - test on raw domains)
    Calculates CORAL loss and domain-aware consistency metrics for monitoring training effectiveness
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of validation loaders (input_src, true_src, input_tgt, true_tgt)
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
        fda_win_h: FDA window height (for reference, not used in validation)
        fda_win_w: FDA window width (for reference, not used in validation)
        fda_weight: FDA weight (for reference, not used in validation)
        coral_loss_fn: CORAL loss function instance (if None, will create default)
        residual_consistency_weight: Weight for residual pattern consistency (for monitoring)
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize CORAL loss if not provided
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_coral_loss = 0.0
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0
    
    # Domain-aware consistency tracking
    epoc_residual_consistency = 0.0 
    
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # Get and preprocess data (DIRECT validation - no FDA)
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocessing (DIRECT validation - no FDA translation)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
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

        # Track source residual magnitude
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        epoc_residual_norm += residual_src_norm * x_src.shape[0]

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
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

        # Track target residual magnitude
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += residual_tgt_norm * x_tgt.shape[0]

        # === CORAL LOSS (on residual features for monitoring) ===
        if domain_weight > 0:
            coral_loss = coral_loss_fn(features_src, features_tgt)
            epoc_coral_loss += coral_loss.numpy() * x_src.shape[0]
        
        # === DOMAIN-AWARE CONSISTENCY: RESIDUAL PATTERN CONSISTENCY ===
        if residual_consistency_weight > 0:
            # Normalize residuals by input magnitude to account for domain differences
            fda_input_magnitude = tf.abs(x_scaled_src) + 1e-8  # Use source as "FDA" reference
            tgt_input_magnitude = tf.abs(x_scaled_tgt) + 1e-8
            
            # Normalized residual patterns
            fda_residual_normalized = residual_src / fda_input_magnitude  # Source residual patterns
            tgt_residual_normalized = residual_tgt / tgt_input_magnitude  # Target residual patterns
            
            # Consistency: similar relative correction patterns
            residual_pattern_consistency = tf.reduce_mean(tf.square(fda_residual_normalized - tgt_residual_normalized))
            epoc_residual_consistency += residual_pattern_consistency.numpy() * x_src.shape[0]
        
        # === SMOOTHNESS LOSS COMPUTATION ===
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

        # === Save H samples (same structure as training validation) ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_coral_loss = epoc_coral_loss / N_val_source if epoc_coral_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / (N_val_source + N_val_target)  # Average of both domains
    
    # Domain-aware consistency averages (MATCHING TRAINING FUNCTION)
    avg_residual_consistency = epoc_residual_consistency / N_val_source if epoc_residual_consistency > 0 else 0.0
    
    # Print enhanced validation statistics (MATCHING TRAINING OUTPUT)
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    Validation CORAL loss: {avg_coral_loss:.6f}")
    print(f"    Validation residual pattern consistency: {avg_residual_consistency:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Domain accuracy placeholders (compatible with existing code)
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss (MATCHING TRAINING FUNCTION STRUCTURE)
    avg_total_loss = (est_weight * avg_loss_est + 
                    domain_weight * avg_coral_loss + 
                    residual_consistency_weight * avg_residual_consistency +   
                    avg_smoothness_loss)

    # Return compatible structure (MATCHING TRAINING FUNCTION OUTPUT)
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_domain_loss': avg_coral_loss + avg_residual_consistency, 
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss,
        # Additional metrics for detailed monitoring
        'avg_residual_consistency': avg_residual_consistency,
        'avg_coral_loss': avg_coral_loss  # Explicit CORAL loss for tracking
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return

### Combined JMMD and CORAL
def train_step_cnn_residual_coral_jmmd(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1,
                                    save_features=False, nsymb=14, weights=None, linear_interp=False,
                                    coral_loss_fn=None, jmmd_loss_fn=None, 
                                    coral_weight=0.5, jmmd_weight=0.5):
    """
    CNN-only residual training with BOTH CORAL and JMMD domain adaptation
    
    Args:
        coral_loss_fn: CORAL loss function instance
        jmmd_loss_fn: JMMD loss function instance  
        coral_weight: Weight for CORAL loss (default 0.5)
        jmmd_weight: Weight for JMMD loss (default 0.5)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize both loss functions if not provided
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_coral = 0.0      # Track CORAL loss separately
    epoc_loss_jmmd = 0.0       # Track JMMD loss separately
    epoc_residual_norm = 0.0
    N_train = 0
    
    # Feature storage setup
    features_src = None
    features_tgt = None
    
    if save_features and domain_weight != 0:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get and preprocess data (same as your existing code)
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocessing (same as your existing code)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        # === CNN Training with DUAL Domain Adaptation ===
        with tf.GradientTape() as tape:
            # Residual learning on source
            residual_src, features_src = model_cnn(x_scaled_src, training=True, return_features=True)
            x_corrected_src = x_scaled_src + residual_src
            
            # Forward pass on target (for domain adaptation)
            residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=True, return_features=True)
            x_corrected_tgt = x_scaled_tgt + residual_tgt
            
            # Estimation loss (main task)
            est_loss = loss_fn_est(y_scaled_src, x_corrected_src)
            
            # === DUAL DOMAIN ADAPTATION: CORAL + JMMD ===
            if domain_weight > 0:
                # CORAL loss: Align covariance structures
                coral_loss = coral_loss_fn(features_src, features_tgt)
                
                # JMMD loss: Align overall distributions  
                jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
                
                # Combined domain loss with separate weights
                combined_domain_loss = (coral_weight * coral_loss + 
                                       jmmd_weight * jmmd_loss)
            else:
                coral_loss = 0.0
                jmmd_loss = 0.0
                combined_domain_loss = 0.0
            
            # Residual regularization
            residual_reg = 0.001 * (tf.reduce_mean(tf.square(residual_src)) + 
                                tf.reduce_mean(tf.square(residual_tgt)))
            
            # Smoothness loss
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
            
            # === TOTAL LOSS with DUAL Domain Adaptation ===
            total_loss = (est_weight * est_loss + 
                        domain_weight * combined_domain_loss +  # Combined CORAL + JMMD
                        residual_reg + 
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)
        
        # Monitor target performance (no gradients)
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === Save features if required ===
        if save_features and domain_weight != 0:
            # Save features for analysis
            features_np_source = features_src[-1].numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features', data=features_np_source,
                    maxshape=(None,) + features_np_source.shape[1:], chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_source.shape[0], axis=0)
                features_dataset_source[-features_np_source.shape[0]:] = features_np_source
                
            features_np_target = features_tgt[-1].numpy()
            if features_dataset_target is None:
                features_dataset_target = features_h5_target.create_dataset(
                    'features', data=features_np_target,
                    maxshape=(None,) + features_np_target.shape[1:], chunks=True
                )
            else:
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
                features_dataset_target[-features_np_target.shape[0]:] = features_np_target
        
        # Update model
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_coral += coral_loss.numpy() * x_src.shape[0] if domain_weight > 0 else 0.0
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0] if domain_weight > 0 else 0.0
        epoc_residual_norm += (tf.reduce_mean(tf.abs(residual_src)) + tf.reduce_mean(tf.abs(residual_tgt))).numpy() / 2 * x_src.shape[0]
    
    # Close feature files
    if save_features and domain_weight != 0:
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_coral = epoc_loss_coral / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    
    # Print enhanced statistics showing both losses
    print(f"    CORAL + JMMD residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    CORAL loss: {avg_loss_coral:.6f} (weight: {coral_weight})")
    print(f"    JMMD loss: {avg_loss_jmmd:.6f} (weight: {jmmd_weight})")
    print(f"    Combined domain loss: {avg_loss_coral + avg_loss_jmmd:.6f}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_coral + avg_loss_jmmd,  # Combined domain loss
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_src[-1] if features_src is not None else None,
        film_features_source=features_src[-1] if features_src is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )

def val_step_cnn_residual_coral_jmmd(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, 
                                    weights=None, linear_interp=False, return_H_gen=False,
                                    coral_loss_fn=None, jmmd_loss_fn=None,
                                    coral_weight=0.5, jmmd_weight=0.5):
    """
    Validation step for CNN-only residual learning with BOTH CORAL and JMMD
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize loss functions
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_coral_loss = 0.0      # Track CORAL separately
    epoc_jmmd_loss = 0.0       # Track JMMD separately
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # Get and preprocess data (same as your existing validation code)
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocessing (same as training)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === Residual learning predictions ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src
        
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt
        
        # Safe tensor conversion and evaluation (same as your existing code)
        preds_src_numpy = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_tgt_numpy = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        
        preds_src_descaled = deMinMax(preds_src_numpy, x_min_src, x_max_src, lower_range=lower_range)
        preds_tgt_descaled = deMinMax(preds_tgt_numpy, x_min_tgt, x_max_tgt, lower_range=lower_range)
        
        # Calculate losses and metrics (same as your existing validation code)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        
        # NMSE calculations (same as your existing code)
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]
        
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # Track residual magnitudes
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === DUAL Domain Adaptation Loss (CORAL + JMMD) ===
        if domain_weight > 0:
            coral_loss = coral_loss_fn(features_src, features_tgt)
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            
            epoc_coral_loss += coral_loss.numpy() * x_src.shape[0]
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # Smoothness loss computation (same as your existing code)
        if temporal_weight != 0 or frequency_weight != 0:
            preds_src_tensor = tf.convert_to_tensor(preds_src_numpy) if not tf.is_tensor(preds_src) else preds_src
            preds_tgt_tensor = tf.convert_to_tensor(preds_tgt_numpy) if not tf.is_tensor(preds_tgt) else preds_tgt
            
            smoothness_loss_src = compute_total_smoothness_loss(preds_src_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
            smoothness_loss_tgt = compute_total_smoothness_loss(preds_tgt_tensor, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
            batch_smoothness_loss = (smoothness_loss_src + smoothness_loss_tgt) / 2
            epoc_smoothness_loss += batch_smoothness_loss.numpy() * x_src.shape[0]

        # Save H samples (same as your existing code)
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # [Your existing H_sample creation code here]
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            if hasattr(preds_src_descaled, 'numpy'):
                H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            else:
                H_est_sample = preds_src_descaled[:n_samples].copy()
            
            # Calculate metrics for samples
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            
            # Target samples
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
        H_gen = {'H_gen_src': H_gen_src_all, 'H_gen_tgt': H_gen_tgt_all}

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_coral_loss = epoc_coral_loss / N_val_source if epoc_coral_loss > 0 else 0.0
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Print enhanced validation statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    Validation CORAL loss: {avg_coral_loss:.6f} (weight: {coral_weight})")
    print(f"    Validation JMMD loss: {avg_jmmd_loss:.6f} (weight: {jmmd_weight})")
    print(f"    Validation combined domain loss: {avg_coral_loss + avg_jmmd_loss:.6f}")
    
    # Domain accuracy placeholders
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss with both domain adaptation methods
    avg_total_loss = (est_weight * avg_loss_est + 
                    domain_weight * (coral_weight * avg_coral_loss + jmmd_weight * avg_jmmd_loss) + 
                    avg_smoothness_loss)

    # Return structure with combined domain loss
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,
        'avg_domain_loss': avg_coral_loss + avg_jmmd_loss,  # Combined domain loss for plotting
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss,
        # Additional metrics for detailed analysis
        'avg_coral_loss': avg_coral_loss,
        'avg_jmmd_loss': avg_jmmd_loss
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return

# combined FDA full translate1 and jmmd and coral
def train_step_cnn_residual_FDAfullTranslation1_coral_jmmd(model_cnn, loader_H, loss_fn, optimizer, lower_range=-1, 
                                            save_features=False, nsymb=14, weights=None, linear_interp=False,
                                            fda_win_h=13, fda_win_w=3, fda_weight=1.0, 
                                            coral_loss_fn=None, jmmd_loss_fn=None,
                                            coral_weight=0.5, jmmd_weight=0.5, flag_residual_reg=True):
    """
    CNN-only residual training step combining FDA Full Translation 1 + CORAL + JMMD domain adaptation
    
    Workflow:
    1. RAW FDA FULL Translation: Source input → Target style input AND 
                                Source labels → Target INPUT style labels
    2. Scale FDA outputs with shared parameters for consistency
    3. DUAL Domain Adaptation: CORAL loss (covariance alignment) + JMMD loss (distribution alignment)
    4. Residual learning: Train CNN to predict corrections
    
    Training pairs:
    - Input: Source content + Target style (FDA-mixed input, then scaled)
    - Label: Source labels + Target INPUT style (FDA-mixed labels using TARGET INPUT as style)
    - Feature alignment: CORAL + JMMD losses between FDA features and raw target features
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of loaders
        loss_fn: tuple of loss functions (only first one used)
        optimizer: single optimizer for CNN
        lower_range: lower range for min-max scaling
        save_features: bool, whether to save features
        nsymb: number of symbols  
        weights: weight dictionary
        linear_interp: linear interpolation flag
        fda_win_h: FDA window height (default 13)
        fda_win_w: FDA window width (default 3)
        fda_weight: Weight for FDA vs source training (1.0 = 100% FDA)
        coral_loss_fn: CORAL loss function instance (if None, will create default)
        jmmd_loss_fn: JMMD loss function instance (if None, will create default)
        coral_weight: Weight for CORAL loss (default 0.5)
        jmmd_weight: Weight for JMMD loss (default 0.5)
        flag_residual_reg: Enable residual regularization (default True)
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize CORAL and JMMD losses if not provided
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_coral = 0.0      # Track CORAL separately
    epoc_loss_jmmd = 0.0       # Track JMMD separately
    epoc_residual_norm = 0.0
    epoc_fda_effect_input = 0.0   # Track FDA effect on inputs
    epoc_fda_effect_label = 0.0   # Track FDA effect on labels
    N_train = 0
    
    # Initialize feature variables
    features_fda = None
    features_tgt_raw = None
    
    if save_features and domain_weight != 0:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_path_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get data
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()  # Not used in label translation
        N_train += x_src.shape[0]

        # Preprocess data to real format (NO SCALING YET - RAW FDA FIRST!)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))

        x_tgt_real = complx2real(x_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))

        # ============ STEP 1: RAW FDA INPUT TRANSLATION (Source→Target INPUT) ============
        x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
        x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
        
        src_amplitude_input, src_phase_input = F_extract_DD(x_src_complex)
        tgt_amplitude_input, tgt_phase_input = F_extract_DD(x_tgt_complex)
        mixed_amplitude_input = fda_mix_pixels(src_amplitude_input, tgt_amplitude_input, fda_win_h, fda_win_w)
        mixed_complex_dd_input = apply_phase_to_amplitude(mixed_amplitude_input, src_phase_input)
        x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_input)
        x_fda_mixed_real = tf.stack([tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], axis=-1)
        
        fda_difference_input = tf.reduce_mean(tf.abs(x_fda_mixed_real - x_src_real))
        epoc_fda_effect_input += fda_difference_input.numpy() * x_src.shape[0]
        
        # ============ STEP 2: RAW FDA LABEL TRANSLATION (Source Labels→Target INPUT Style) ============
        y_src_complex = tf.complex(y_src_real[:,:,:,0], y_src_real[:,:,:,1])
        src_amplitude_label, src_phase_label = F_extract_DD(y_src_complex)
        tgt_amplitude_style, tgt_phase_style = F_extract_DD(x_tgt_complex)
        mixed_amplitude_label = fda_mix_pixels(src_amplitude_label, tgt_amplitude_style, fda_win_h, fda_win_w)
        mixed_complex_dd_label = apply_phase_to_amplitude(mixed_amplitude_label, src_phase_label)
        y_fda_mixed_complex = F_inverse_DD(mixed_complex_dd_label)
        y_fda_mixed_real = tf.stack([tf.math.real(y_fda_mixed_complex), tf.math.imag(y_fda_mixed_complex)], axis=-1)
        
        fda_difference_label = tf.reduce_mean(tf.abs(y_fda_mixed_real - y_src_real))
        epoc_fda_effect_label += fda_difference_label.numpy() * x_src.shape[0]
        
        # ============ STEP 3: SCALE FDA OUTPUTS WITH SHARED PARAMETERS ============
        x_fda_mixed_numpy = x_fda_mixed_real.numpy()
        x_fda_scaled, fda_min, fda_max = minmaxScaler(x_fda_mixed_numpy, lower_range=lower_range, linear_interp=linear_interp)
        
        batch_size = y_fda_mixed_real.shape[0]
        if np.isscalar(fda_min):
            fda_min_array = np.tile([[fda_min, fda_min]], (batch_size, 1))
            fda_max_array = np.tile([[fda_max, fda_max]], (batch_size, 1))
        else:
            fda_min_array = fda_min if fda_min.shape == (batch_size, 2) else np.tile(fda_min, (batch_size, 1))
            fda_max_array = fda_max if fda_max.shape == (batch_size, 2) else np.tile(fda_max, (batch_size, 1))
        
        y_fda_mixed_numpy = y_fda_mixed_real.numpy()
        y_fda_scaled, _, _ = minmaxScaler(y_fda_mixed_numpy, min_pre=fda_min_array, max_pre=fda_max_array, 
                                        lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 4: SCALE TARGET FOR DUAL DOMAIN ADAPTATION ============
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        
        # ============ STEP 5: SCALE SOURCE DATA WITH FDA PARAMETERS FOR HYBRID TRAINING ============
        x_src_scaled_with_fda_params, _, _ = minmaxScaler(x_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)
        y_src_scaled_with_fda_params, _, _ = minmaxScaler(y_src_real, min_pre=fda_min_array, max_pre=fda_max_array, 
                                                        lower_range=lower_range, linear_interp=linear_interp)

        # ============ STEP 6: DUAL FLOW CNN TRAINING + DUAL DOMAIN ADAPTATION ============
        with tf.GradientTape() as tape:
            # === FLOW 1: FDA-TRANSLATED INPUT (Main training flow) ===
            batch_size_int = x_fda_scaled.shape[0]
            
            if fda_weight < 1.0:
                fda_samples = int(batch_size_int * fda_weight)
                remaining_samples = batch_size_int - fda_samples
                
                if fda_samples > 0 and remaining_samples > 0:
                    x_train_fda = x_fda_scaled[:fda_samples]
                    y_train_fda = y_fda_scaled[:fda_samples]
                    x_train_src = x_src_scaled_with_fda_params[fda_samples:]
                    y_train_src = y_src_scaled_with_fda_params[fda_samples:]
                    
                    x_train_combined = tf.concat([x_train_fda, x_train_src], axis=0)
                    y_train_combined = tf.concat([y_train_fda, y_train_src], axis=0)
                    
                    residual_combined, features_fda = model_cnn(x_train_combined, training=True, return_features=True)
                    x_corrected_combined = x_train_combined + residual_combined
                    
                elif fda_samples == 0:
                    residual_src, features_fda = model_cnn(x_src_scaled_with_fda_params, training=True, return_features=True)
                    x_corrected_combined = x_src_scaled_with_fda_params + residual_src
                    y_train_combined = y_src_scaled_with_fda_params
                    residual_combined = residual_src
                else:
                    residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                    x_corrected_combined = x_fda_scaled + residual_fda
                    y_train_combined = y_fda_scaled
                    residual_combined = residual_fda
                
                est_loss = loss_fn_est(y_train_combined, x_corrected_combined)
                    
            else:
                # Pure FDA training
                residual_fda, features_fda = model_cnn(x_fda_scaled, training=True, return_features=True)
                x_corrected_fda = x_fda_scaled + residual_fda
                residual_combined = residual_fda
                est_loss = loss_fn_est(y_fda_scaled, x_corrected_fda)
            
            # === FLOW 2: RAW TARGET INPUT (For dual domain adaptation) ===
            residual_tgt_raw, features_tgt_raw = model_cnn(x_scaled_tgt, training=False, return_features=True)
            x_corrected_tgt = x_scaled_tgt + residual_tgt_raw
            
            # === DUAL DOMAIN ADAPTATION: CORAL + JMMD ===
            if domain_weight > 0:
                # CORAL loss: Align covariance structures
                coral_loss = coral_loss_fn(features_fda, features_tgt_raw)
                
                # JMMD loss: Align overall distributions
                jmmd_loss = jmmd_loss_fn(features_fda, features_tgt_raw)
                
                # Combined domain loss with separate weights
                combined_domain_loss = (coral_weight * coral_loss + 
                                       jmmd_weight * jmmd_loss)
            else:
                coral_loss = 0.0
                jmmd_loss = 0.0
                combined_domain_loss = 0.0
            
            # === REGULARIZATION LOSSES ===
            # Residual regularization: Encourage small, meaningful corrections
            if flag_residual_reg:
                residual_reg = 0.001 * tf.reduce_mean(tf.square(residual_combined))
            else:
                residual_reg = 0.0
            
            # Smoothness loss
            if temporal_weight != 0 or frequency_weight != 0:
                if fda_weight < 1.0:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_combined, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
                else:
                    smoothness_loss = compute_total_smoothness_loss(x_corrected_fda, 
                                                                temporal_weight=temporal_weight, 
                                                                frequency_weight=frequency_weight)
            else:
                smoothness_loss = 0.0
            
            # === TOTAL LOSS WITH DUAL DOMAIN ADAPTATION ===
            total_loss = (est_weight * est_loss + 
                        domain_weight * combined_domain_loss +  # Combined CORAL + JMMD
                        residual_reg + 
                        smoothness_loss)
            
            # Add L2 regularization from model
            if model_cnn.losses:
                total_loss += tf.add_n(model_cnn.losses)

        # === MONITOR TARGET PERFORMANCE ===
        y_tgt_real = complx2real(y_tgt)
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)
        
        est_loss_tgt = loss_fn_est(y_scaled_tgt, x_corrected_tgt)
        epoc_loss_est_target += est_loss_tgt.numpy() * x_tgt.shape[0]
        
        # === Save features if required ===
        if save_features and domain_weight != 0:
            features_np_fda = features_fda[-1].numpy()
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features', data=features_np_fda,
                    maxshape=(None,) + features_np_fda.shape[1:], chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + features_np_fda.shape[0], axis=0)
                features_dataset_source[-features_np_fda.shape[0]:] = features_np_fda
                
            features_np_target = features_tgt_raw[-1].numpy()
            if features_dataset_target is None:
                features_dataset_target = features_h5_target.create_dataset(
                    'features', data=features_np_target,
                    maxshape=(None,) + features_np_target.shape[1:], chunks=True
                )
            else:
                features_dataset_target.resize(features_dataset_target.shape[0] + features_np_target.shape[0], axis=0)
                features_dataset_target[-features_np_target.shape[0]:] = features_np_target
        
        # Update model
        grads = tape.gradient(total_loss, model_cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_cnn.trainable_variables))
        
        # Track metrics
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss.numpy() * x_src.shape[0]
        epoc_loss_coral += coral_loss.numpy() * x_src.shape[0] if domain_weight > 0 else 0.0
        epoc_loss_jmmd += jmmd_loss.numpy() * x_src.shape[0] if domain_weight > 0 else 0.0
        epoc_residual_norm += tf.reduce_mean(tf.abs(residual_combined)).numpy() * x_src.shape[0]
    
    # Close feature files
    if save_features and domain_weight != 0:
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_coral = epoc_loss_coral / N_train
    avg_loss_jmmd = epoc_loss_jmmd / N_train
    avg_residual_norm = epoc_residual_norm / N_train
    avg_fda_effect_input = epoc_fda_effect_input / N_train
    avg_fda_effect_label = epoc_fda_effect_label / N_train
    
    # Print enhanced statistics
    print(f"    FDA Full Translation + CORAL + JMMD residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    FDA input translation effect (Source→Target INPUT): {avg_fda_effect_input:.6f}")
    print(f"    FDA label translation effect (Source Labels→Target INPUT style): {avg_fda_effect_label:.6f}")
    print(f"    CORAL loss: {avg_loss_coral:.6f} (weight: {coral_weight})")
    print(f"    JMMD loss: {avg_loss_jmmd:.6f} (weight: {jmmd_weight})")
    print(f"    Combined domain loss: {avg_loss_coral + avg_loss_jmmd:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_coral + avg_loss_jmmd,  # Combined domain loss
        avg_epoc_loss=avg_loss_total,
        avg_epoc_loss_est_target=avg_loss_est_target,
        features_source=features_fda[-1] if features_fda is not None else None,
        film_features_source=features_fda[-1] if features_fda is not None else None,
        avg_epoc_loss_d=0.0  # No discriminator
    )


def val_step_cnn_residual_FDAfullTranslation1_coral_jmmd(model_cnn, loader_H, loss_fn, lower_range, nsymb=14, 
                                                        weights=None, linear_interp=False, return_H_gen=False,
                                                        fda_win_h=13, fda_win_w=3, fda_weight=1.0, 
                                                        coral_loss_fn=None, jmmd_loss_fn=None,
                                                        coral_weight=0.5, jmmd_weight=0.5):
    """
    Validation step for CNN-only residual learning with FDA Full Translation 1 + CORAL + JMMD
    
    Validates model performance using DIRECT validation (no FDA translation during validation - test on raw domains)
    Calculates CORAL and JMMD losses for monitoring training effectiveness
    
    Args:
        model_cnn: CNN model (CNNGenerator instance)
        loader_H: tuple of validation loaders (input_src, true_src, input_tgt, true_tgt)
        loss_fn: tuple of loss functions (only first one used)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        weights: weight dictionary
        linear_interp: linear interpolation flag
        return_H_gen: whether to return generated H matrices
        fda_win_h: FDA window height (for reference, not used in validation)
        fda_win_w: FDA window width (for reference, not used in validation)
        fda_weight: FDA weight (for reference, not used in validation)
        coral_loss_fn: CORAL loss function instance (if None, will create default)
        jmmd_loss_fn: JMMD loss function instance (if None, will create default)
        coral_weight: Weight for CORAL loss (for monitoring)
        jmmd_weight: Weight for JMMD loss (for monitoring)
    """
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est = loss_fn[0]  # Only need estimation loss
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    
    # Initialize CORAL and JMMD losses if not provided
    if coral_loss_fn is None:
        coral_loss_fn = GlobalPoolingCORALLoss()
    if jmmd_loss_fn is None:
        jmmd_loss_fn = JMMDLoss()
    
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_coral_loss = 0.0      # Track CORAL separately
    epoc_jmmd_loss = 0.0       # Track JMMD separately
    epoc_smoothness_loss = 0.0
    epoc_residual_norm = 0.0
    H_sample = []
    
    if return_H_gen:
        all_H_gen_src = []
        all_H_gen_tgt = []

    for idx in range(loader_H_true_val_source.total_batches):
        # Get and preprocess data (DIRECT validation - no FDA translation)
        x_src = loader_H_input_val_source.next_batch()
        y_src = loader_H_true_val_source.next_batch()
        x_tgt = loader_H_input_val_target.next_batch()
        y_tgt = loader_H_true_val_target.next_batch()
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocessing (DIRECT validation - no FDA translation)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === RESIDUAL LEARNING: Source domain prediction ===
        residual_src, features_src = model_cnn(x_scaled_src, training=False, return_features=True)
        preds_src = x_scaled_src + residual_src  # Apply residual correction
        
        # Safe tensor conversion
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

        # === RESIDUAL LEARNING: Target domain prediction ===
        residual_tgt, features_tgt = model_cnn(x_scaled_tgt, training=False, return_features=True)
        preds_tgt = x_scaled_tgt + residual_tgt  # Apply residual correction
        
        # Safe tensor conversion
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

        # Track residual magnitudes during validation
        residual_src_norm = tf.reduce_mean(tf.square(residual_src)).numpy()
        residual_tgt_norm = tf.reduce_mean(tf.square(residual_tgt)).numpy()
        epoc_residual_norm += (residual_src_norm + residual_tgt_norm) / 2 * x_src.shape[0]

        # === DUAL DOMAIN ADAPTATION: CORAL + JMMD (for monitoring) ===
        if domain_weight > 0:
            # CORAL loss: Align covariance structures
            coral_loss = coral_loss_fn(features_src, features_tgt)
            epoc_coral_loss += coral_loss.numpy() * x_src.shape[0]
            
            # JMMD loss: Align overall distributions
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            epoc_jmmd_loss += jmmd_loss.numpy() * x_src.shape[0]
        
        # === SMOOTHNESS LOSS COMPUTATION ===
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

        # === Save H samples (same structure as training validation) ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source samples
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
            
            # Target samples
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
            # Convert to numpy if needed and append to lists
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

    # Calculate averages
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_coral_loss = epoc_coral_loss / N_val_source if epoc_coral_loss > 0 else 0.0
    avg_jmmd_loss = epoc_jmmd_loss / N_val_source if epoc_jmmd_loss > 0 else 0.0
    avg_smoothness_loss = epoc_smoothness_loss / N_val_source if epoc_smoothness_loss > 0 else 0.0
    avg_residual_norm = epoc_residual_norm / N_val_source
    
    # Print enhanced validation statistics
    print(f"    Validation residual norm (avg): {avg_residual_norm:.6f}")
    print(f"    Validation CORAL loss: {avg_coral_loss:.6f} (weight: {coral_weight})")
    print(f"    Validation JMMD loss: {avg_jmmd_loss:.6f} (weight: {jmmd_weight})")
    print(f"    Validation combined domain loss: {avg_coral_loss + avg_jmmd_loss:.6f}")
    print(f"    FDA window: {fda_win_h}x{fda_win_w}, weight: {fda_weight}")
    
    # Domain accuracy placeholders (compatible with existing code)
    avg_domain_acc_source = 0.5
    avg_domain_acc_target = 0.5
    avg_domain_acc = 0.5

    # Total loss with dual domain adaptation
    avg_total_loss = (est_weight * avg_loss_est + 
                    domain_weight * (coral_weight * avg_coral_loss + jmmd_weight * avg_jmmd_loss) + 
                    avg_smoothness_loss)

    # Return compatible structure with combined domain loss
    epoc_eval_return = {
        'avg_total_loss': avg_total_loss,
        'avg_loss_est_source': avg_loss_est_source,
        'avg_loss_est_target': avg_loss_est_target, 
        'avg_loss_est': avg_loss_est,
        'avg_gan_disc_loss': 0.0,  # No discriminator
        'avg_domain_loss': avg_coral_loss + avg_jmmd_loss,  # Combined domain loss for plotting
        'avg_nmse_source': avg_nmse_source,
        'avg_nmse_target': avg_nmse_target,
        'avg_nmse': avg_nmse,
        'avg_domain_acc_source': avg_domain_acc_source,
        'avg_domain_acc_target': avg_domain_acc_target,
        'avg_domain_acc': avg_domain_acc,
        'avg_smoothness_loss': avg_smoothness_loss,
        # Additional metrics for detailed monitoring
        'avg_coral_loss': avg_coral_loss,
        'avg_jmmd_loss': avg_jmmd_loss
    }

    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen
    return H_sample, epoc_eval_return




### Input Domain translation 
class CycleGANGenerator(tf.keras.Model):
    """Generator for CycleGAN B→A translation"""
    def __init__(self, input_channels=2, output_channels=2):
        super().__init__()        
        self.down1 = tf.keras.layers.Conv2D(64, kernel=(4,3), strides=(2,1), padding='valid', activation='relu')
        self.down2 = tf.keras.layers.Conv2D(128, kernel=(3,3), strides=(2,1), padding='valid', activation='relu')
        self.down3 = tf.keras.layers.Conv2D(256, kernel=(4,3), strides=(2,1), padding='valid', activation='relu')
        
        self.up1 = tf.keras.layers.Conv2DTranspose(128, kernel=(4,3), strides=(2,1), padding='valid', activation='relu')
        self.up2 = tf.keras.layers.Conv2DTranspose(64, kernel=(3,3), strides=(2,1), padding='valid', activation='relu')
        self.up3 = tf.keras.layers.Conv2DTranspose(2, kernel=(4,3), strides=(2,1), padding='valid', activation='tanh')

    
    def call(self, x):
        out = reflect_padding_2d(x, pad_h=0, pad_w=1)
        out = self.down1(out)
        out = reflect_padding_2d(out, pad_h=0, pad_w=1)
        out = self.down2(out)
        out = reflect_padding_2d(out, pad_h=0, pad_w=1)
        out = self.down3(out)   
        #
        out = reflect_padding_2d(out, pad_h=0, pad_w=1)
        out = self.up1(out)
        out = reflect_padding_2d(out, pad_h=0, pad_w=1)
        out = self.up2(out) 
        out = reflect_padding_2d(out, pad_h=0, pad_w=1)
        out = self.up3(out)
        return out

class CycleGANDiscriminator(tf.keras.Model):
    """Discriminator for CycleGAN"""
    def __init__(self, input_channels=2):
        super().__init__()
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel=(4,3), strides=(2,1), padding='valid'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(128, kernel=(3,3), strides=(2,1), padding='valid'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(256, kernel=(4,3), strides=(2,1), padding='valid'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(1, kernel=(3,3), padding='valid', activation='sigmoid'),
        ])
    
    def call(self, x):
        return self.discriminator(x)

def train_step_cyclegan_input_translation(generators, discriminators, loader_H, loss_fn, optimizers, 
                                        lower_range=-1, save_features=False, nsymb=14, weights=None, 
                                        linear_interp=False, cycle_consistency_weight=10.0, 
                                        identity_weight=0.5, translation_weight=1.0):
    """
    STANDARD CycleGAN for input domain translation (TDL-B → TDL-D style)
    Generators do DIRECT translation, not residual learning
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    
    # Unpack models - these are DIRECT translation generators
    G_A2B, G_B2A = generators  # A=TDL-D, B=TDL-B - DIRECT translation
    D_A, D_B = discriminators
    
    # Unpack optimizers
    if len(optimizers) == 4:
        G_A2B_opt, G_B2A_opt, D_A_opt, D_B_opt = optimizers
    else:
        gen_opt, disc_opt = optimizers
        G_A2B_opt = G_B2A_opt = gen_opt
        D_A_opt = D_B_opt = disc_opt
    
    # Loss functions and weights
    loss_fn_est = loss_fn[0]
    loss_fn_bce = loss_fn[1] if len(loss_fn) > 1 else tf.keras.losses.BinaryCrossentropy()
    
    est_weight = weights.get('est_weight', 1.0) if weights else 1.0
    domain_weight = weights.get('domain_weight', 1.0) if weights else 1.0
    temporal_weight = weights.get('temporal_weight', 0.0) if weights else 0.0
    frequency_weight = weights.get('frequency_weight', 0.0) if weights else 0.0
    adv_weight = weights.get('adv_weight', 0.01) if weights else 0.01
    
    # Training metrics
    epoc_loss_total = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_target = 0.0
    epoc_loss_cycle = 0.0
    epoc_loss_identity = 0.0
    epoc_loss_discriminator = 0.0
    N_train = 0
    
    # Initialize for return statement
    last_translated_A = None
    
    # Feature storage
    if save_features:
        features_h5_path_source = 'features_source.h5'
        if os.path.exists(features_h5_path_source):
            os.remove(features_h5_path_source)
        features_h5_source = h5py.File(features_h5_path_source, 'w')
        features_dataset_source = None

        features_h5_path_target = 'features_target.h5'
        if os.path.exists(features_h5_path_target):
            os.remove(features_h5_path_target)   
        features_h5_target = h5py.File(features_h5_target, 'w')
        features_dataset_target = None
    
    for batch_idx in range(loader_H_true_train_src.total_batches):
        # Get and preprocess data (same as before)
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocessing (same as before)
        x_src_real = complx2real(x_src)
        y_src_real = complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_A, x_min_A, x_max_A = minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_A, _, _ = minmaxScaler(y_src_real, min_pre=x_min_A, max_pre=x_max_A, lower_range=lower_range)

        x_tgt_real = complx2real(x_tgt)
        y_tgt_real = complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_B, x_min_B, x_max_B = minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_B, _, _ = minmaxScaler(y_tgt_real, min_pre=x_min_B, max_pre=x_max_B, lower_range=lower_range)

        # === 1. Train Discriminators (STANDARD CycleGAN) ===
        with tf.GradientTape() as tape_D_A:
            # DIRECT TRANSLATION (not residual)
            translated_A = G_B2A(x_scaled_B, training=False)  # B → A translation ✅
            
            # Discriminator A: Real A vs Translated A (B→A)
            d_real_A = D_A(y_scaled_A, training=True)  # Real clean A channels
            d_fake_A = D_A(translated_A, training=True)  # Translated B→A channels
            
            real_A_loss = loss_fn_bce(tf.ones_like(d_real_A), d_real_A)
            fake_A_loss = loss_fn_bce(tf.zeros_like(d_fake_A), d_fake_A)
            D_A_loss = (real_A_loss + fake_A_loss) / 2
            
        grads_D_A = tape_D_A.gradient(D_A_loss, D_A.trainable_variables)
        D_A_opt.apply_gradients(zip(grads_D_A, D_A.trainable_variables))

        with tf.GradientTape() as tape_D_B:
            # DIRECT TRANSLATION (not residual)
            translated_B = G_A2B(x_scaled_A, training=False)  # A → B translation ✅
            
            # Discriminator B: Real B vs Translated B (A→B)
            d_real_B = D_B(y_scaled_B, training=True)  # Real clean B channels
            d_fake_B = D_B(translated_B, training=True)  # Translated A→B channels
            
            real_B_loss = loss_fn_bce(tf.ones_like(d_real_B), d_real_B)
            fake_B_loss = loss_fn_bce(tf.zeros_like(d_fake_B), d_fake_B)
            D_B_loss = (real_B_loss + fake_B_loss) / 2
            
        grads_D_B = tape_D_B.gradient(D_B_loss, D_B.trainable_variables)
        D_B_opt.apply_gradients(zip(grads_D_B, D_B.trainable_variables))

        # === 2. Train Generators (STANDARD CycleGAN) ===
        with tf.GradientTape() as tape_G_A2B, tf.GradientTape() as tape_G_B2A:
            # DIRECT TRANSLATIONS (not residual learning) ✅
            translated_B = G_A2B(x_scaled_A, training=True)  # A → B translation
            translated_A = G_B2A(x_scaled_B, training=True)  # B → A translation
            
            # Store for return statement
            last_translated_A = translated_A
            
            # === CYCLE CONSISTENCY (STANDARD CycleGAN) ===
            cycle_A = G_B2A(translated_B, training=True)  # A → B → A
            cycle_B = G_A2B(translated_A, training=True)  # B → A → B
            
            # === IDENTITY MAPPING (STANDARD CycleGAN) ===
            identity_A = G_B2A(x_scaled_A, training=True)  # Should return A when given A
            identity_B = G_A2B(x_scaled_B, training=True)  # Should return B when given B
            
            # === ADVERSARIAL LOSSES (fool discriminators) ===
            d_fake_A_for_gen = D_A(translated_A, training=False)
            d_fake_B_for_gen = D_B(translated_B, training=False)
            
            G_B2A_adv_loss = loss_fn_bce(tf.ones_like(d_fake_A_for_gen), d_fake_A_for_gen)
            G_A2B_adv_loss = loss_fn_bce(tf.ones_like(d_fake_B_for_gen), d_fake_B_for_gen)
            
            # === CYCLE CONSISTENCY LOSS ===
            cycle_A_loss = tf.reduce_mean(tf.abs(cycle_A - x_scaled_A))  # L1 loss
            cycle_B_loss = tf.reduce_mean(tf.abs(cycle_B - x_scaled_B))  # L1 loss
            cycle_consistency_loss = cycle_A_loss + cycle_B_loss
            
            # === IDENTITY LOSS ===
            identity_A_loss = tf.reduce_mean(tf.abs(identity_A - x_scaled_A))  # L1 loss
            identity_B_loss = tf.reduce_mean(tf.abs(identity_B - x_scaled_B))  # L1 loss
            identity_loss = identity_A_loss + identity_B_loss
            
            # === ESTIMATION LOSSES with HYBRID TRAINING ===
            batch_size = tf.shape(x_scaled_A)[0]
            
            if translation_weight < 1.0:
                # HYBRID TRAINING: Mix translated and original samples
                translated_samples = tf.cast(tf.cast(batch_size, tf.float32) * translation_weight, tf.int32)
                original_samples = batch_size - translated_samples
                
                if translated_samples > 0 and original_samples > 0:
                    # Split batch for hybrid training
                    # Part 1: Translated estimation
                    est_loss_A_translated = loss_fn_est(y_scaled_A[:translated_samples], 
                                                        translated_A[:translated_samples])
                    est_loss_B_translated = loss_fn_est(y_scaled_B[:translated_samples], 
                                                        translated_B[:translated_samples])
                    
                    # Part 2: Original estimation (source preservation)
                    est_loss_A_original = loss_fn_est(y_scaled_A[translated_samples:], 
                                                    x_scaled_A[translated_samples:])  # ← Original source
                    est_loss_B_original = loss_fn_est(y_scaled_B[translated_samples:], 
                                                    x_scaled_B[translated_samples:])  # ← Original target
                    
                    # Weighted combination
                    est_loss_A = (translation_weight * est_loss_A_translated + 
                                (1.0 - translation_weight) * est_loss_A_original)
                    est_loss_B = (translation_weight * est_loss_B_translated + 
                                (1.0 - translation_weight) * est_loss_B_original)
                    
                elif translated_samples >= original_samples:
                    # Mostly translated
                    est_loss_A = loss_fn_est(y_scaled_A, translated_A)
                    est_loss_B = loss_fn_est(y_scaled_B, translated_B)
                else:
                    # Mostly original
                    est_loss_A = loss_fn_est(y_scaled_A, x_scaled_A)  # Original source
                    est_loss_B = loss_fn_est(y_scaled_B, x_scaled_B)  # Original target
            else:
                # Pure translated training (original behavior)
                est_loss_A = loss_fn_est(y_scaled_A, translated_A)
                est_loss_B = loss_fn_est(y_scaled_B, translated_B)
            
            # === SMOOTHNESS REGULARIZATION ===
            if temporal_weight != 0 or frequency_weight != 0:
                smoothness_A = compute_total_smoothness_loss(translated_A, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_B = compute_total_smoothness_loss(translated_B, temporal_weight=temporal_weight, frequency_weight=frequency_weight)
                smoothness_loss = (smoothness_A + smoothness_B) / 2
            else:
                smoothness_loss = 0.0
            
            # === TOTAL GENERATOR LOSSES ===
            G_A2B_total_loss = (adv_weight * G_A2B_adv_loss + 
                                cycle_consistency_weight * cycle_consistency_loss + 
                                identity_weight * identity_loss +
                                est_weight * est_loss_B +  # Estimation quality on translated/hybrid B
                                smoothness_loss)
            
            G_B2A_total_loss = (adv_weight * G_B2A_adv_loss + 
                                cycle_consistency_weight * cycle_consistency_loss + 
                                identity_weight * identity_loss +
                                est_weight * est_loss_A +  # Estimation quality on translated/hybrid A
                                smoothness_loss)
        
        # Update generators
        grads_G_A2B = tape_G_A2B.gradient(G_A2B_total_loss, G_A2B.trainable_variables)
        G_A2B_opt.apply_gradients(zip(grads_G_A2B, G_A2B.trainable_variables))
        
        grads_G_B2A = tape_G_B2A.gradient(G_B2A_total_loss, G_B2A.trainable_variables)
        G_B2A_opt.apply_gradients(zip(grads_G_B2A, G_B2A.trainable_variables))
        
        # === Save features if required ===
        if save_features:
            # Save translated features for analysis
            translated_A_numpy = translated_A.numpy()
            translated_B_numpy = translated_B.numpy()
            
            if features_dataset_source is None:
                features_dataset_source = features_h5_source.create_dataset(
                    'features', data=translated_A_numpy,
                    maxshape=(None,) + translated_A_numpy.shape[1:], chunks=True
                )
            else:
                features_dataset_source.resize(features_dataset_source.shape[0] + translated_A_numpy.shape[0], axis=0)
                features_dataset_source[-translated_A_numpy.shape[0]:] = translated_A_numpy
                
            if features_dataset_target is None:
                features_dataset_target = features_h5_target.create_dataset(
                    'features', data=translated_B_numpy,
                    maxshape=(None,) + translated_B_numpy.shape[1:], chunks=True
                )
            else:
                features_dataset_target.resize(features_dataset_target.shape[0] + translated_B_numpy.shape[0], axis=0)
                features_dataset_target[-translated_B_numpy.shape[0]:] = translated_B_numpy

        # === Track metrics ===
        total_loss = (G_A2B_total_loss + G_B2A_total_loss) / 2  # Average generator loss
        epoc_loss_total += total_loss.numpy() * x_src.shape[0]
        epoc_loss_est += est_loss_A.numpy() * x_src.shape[0]           # Source estimation (main)
        epoc_loss_est_target += est_loss_B.numpy() * x_tgt.shape[0]    # Target estimation (monitoring)
        epoc_loss_cycle += cycle_consistency_loss.numpy() * x_src.shape[0]
        epoc_loss_identity += identity_loss.numpy() * x_src.shape[0]
        epoc_loss_discriminator += (D_A_loss + D_B_loss).numpy() / 2 * x_src.shape[0]

    # Close feature files
    if save_features:
        features_h5_source.close()
        features_h5_target.close()
    
    # Calculate averages
    avg_loss_total = epoc_loss_total / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_est_target = epoc_loss_est_target / N_train
    avg_loss_cycle = epoc_loss_cycle / N_train
    avg_discriminator_loss = epoc_loss_discriminator / N_train
    
    # Print CycleGAN-specific statistics
    print(f"    CycleGAN G_A2B loss: {avg_loss_total:.6f}")
    print(f"    CycleGAN Cycle consistency: {avg_loss_cycle:.6f}")
    print(f"    CycleGAN Discriminator loss: {avg_discriminator_loss:.6f}")
    
    # Return compatible structure
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,                    # Average estimation performance
        avg_epoc_loss_domain=avg_loss_cycle,               # Use cycle loss as "domain" loss
        avg_epoc_loss=avg_loss_total,                      # Average generator loss
        avg_epoc_loss_est_target=avg_loss_est_target,      # Target estimation quality
        features_source=last_translated_A.numpy() if last_translated_A is not None else None,  # Translated source
        film_features_source=last_translated_A.numpy() if last_translated_A is not None else None,
        avg_epoc_loss_d=avg_discriminator_loss             # Average discriminator loss
    )

    
    
    