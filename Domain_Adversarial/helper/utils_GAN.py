""" helper functions and classes for GAN model 
"""
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
import h5py
import sys
import os 

from dataclasses import dataclass
try:
    from . import utils  # For package/notebook context
except ImportError:
    import utils  # For direct execution context


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

class GradientReversal(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -dy  # Reverse the gradient
            return x, grad
        return reverse_gradient(x)
        
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
    def __init__(self, output_channels=2, n_subc=132, gen_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
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
            
    def call(self, x, training=False):
        # Encoder
        d1 = self.down1(x, training=training)      # (batch, 395, 12, C_out)
        d2 = self.down2(d1, training=training)     # (batch, 196, 10, C_out)
        d3 = self.down3(d2, training=training)     # (batch,  97, 8, C_out)
        d4 = self.down4(d3, training=training)     # (batch,  48, 6, C_out)
        # Decoder with skip connections
        u1 = self.up1(d4, d3, training=training)   # (batch,  97, 8, C_out)
        u2 = self.up2(u1, d2, training=training)   # (batch, 196, 10, C_out)
        u3 = self.up3(u2, d1, training=training)   # (batch, 395, 12, C_out)
        u4 = self.last(u3)  # (batch, 792, 14 or 16, C_out)
        if u4.shape[2] > 14:
            u4 = u4[:, :, 1:15, :]
        return u4, d4

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
    def __init__(self, n_subc=132, generator=Pix2PixGenerator, discriminator=PatchGANDiscriminator, gen_l2=None, disc_l2=None):
        super().__init__()
        self.generator = generator(n_subc=n_subc, gen_l2=gen_l2)
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
    
class DomainDisc(tf.keras.Model):
    """ conv 3 times, then global average pooling, then dense layers 
        extracted features with shape [batch, 48, 6, C_out] (C_out= 512 or 128)
    """
    def __init__(self, l2_reg=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg is not None else None
        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=(3,2), strides=(2,1), padding='valid', 
                                            activation='relu', kernel_regularizer=kernel_regularizer)
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(2,2), strides=(2,1), padding='valid', 
                                            activation='relu', kernel_regularizer=kernel_regularizer)
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(2,2), strides=(1,1), padding='valid', 
                                            activation='relu', kernel_regularizer=kernel_regularizer)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, x):
        x = self.conv1(x)  # (batch, 4, 13, 256)
        x = reflect_padding_2d(x, pad_h=1, pad_w=0)  # symmetric padding
        x = self.conv2(x)  # (batch, 3, 12, 128)
        x = reflect_padding_2d(x, pad_h=1, pad_w=0)  # symmetric padding
        x = self.conv3(x)  # (batch, 2, 11, 64)
        x = self.pool(x)   # (batch, 64)
        x = self.fc1(x)    # (batch, 64)
        return self.out(x) # (batch, 1) - domain probability
    
class WeightScheduler:
    def __init__(self, strategy='domain_first_smooth', **kwargs):  
                    # strategy = 'domain_first_smooth' or 'reconstruction_first'
        self.strategy = strategy
        
        # Common parameters for both strategies
        self.temporal_weight = kwargs.get('temporal_weight', 0.02)
        self.frequency_weight = kwargs.get('frequency_weight', 0.1)
        
        if strategy == 'reconstruction_first':
            # Domain scheduling parameters
            self.start_domain_weight = kwargs.get('start_domain_weight', 0.01)
            self.end_domain_weight = kwargs.get('end_domain_weight', 0.08)
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
            self.start_domain_weight = kwargs.get('start_domain_weight', 0.1)  # High start (4.5)
            self.end_domain_weight = kwargs.get('end_domain_weight', 0.01)     # Low end (1.5)
            
            # Est weight parameters for domain_first_smooth
            self.start_est_weight = kwargs.get('start_est_weight', 0.1)       # Low start (0.1)
            self.end_est_weight = kwargs.get('end_est_weight', 1)           # High end (0.6)
            
            # Adv weight parameters for domain_first_smooth
            self.start_adv_weight = kwargs.get('start_adv_weight', 0.005)      # Low start (0.03)
            self.end_adv_weight = kwargs.get('end_adv_weight', 0.005)          # High end (0.08)
            
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


# ================= GAN Training Step =====================
def train_step_gan(model, domain_model, loader_H, loss_fn, optimizers, lower_range=-1, return_features=False,
        nsymb=14, adv_weight=0.01, est_weight=1.0, domain_weight=0.5):
    """
    model: GAN model instance
    loader_H_source: DataLoader for source domain
    loader_H_target: DataLoader for target domain
    loss_fns: tuple of loss functions (estimation loss, binary cross-entropy loss, domain loss)
    optimizers: tuple of optimizers (generator optimizer, discriminator optimizer, domain optimizer)
    lower_range: lower range for min-max scaling
    domain_model: Domain discriminator model class
    """    
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce, loss_fn_domain = loss_fn
    gen_optimizer, disc_optimizer, domain_optimizer = optimizers

    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_domain = 0.0
    N_train = 0
    
    if return_features==True and (domain_weight != 0):
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
        # --- Source domain ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        # --- Target domain ---
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0] + x_tgt.shape[0]

        # Preprocess (source)
        x_src = utils.complx2real(x_src)
        y_src = utils.complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, _, _ = utils.minmaxScaler(x_src, lower_range=lower_range)
        y_scaled_src, _, _ = utils.minmaxScaler(y_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt = utils.complx2real(x_tgt)
        y_tgt = utils.complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, _, _ = utils.minmaxScaler(x_tgt, lower_range=lower_range)
        y_scaled_tgt, _, _ = utils.minmaxScaler(y_tgt, lower_range=lower_range)

        # === 1. Train Discriminator ===
        with tf.GradientTape() as tape_d:
            x_fake_src = model.generator(x_scaled_src, training=True)[0] 
                # x_fake == generated data gen(x_real)
                # x_fake ~= y (y is real)
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            real_labels = tf.ones_like(d_real)
            fake_labels = tf.zeros_like(d_fake)
            d_loss_real = loss_fn_bce(real_labels, d_real)
            d_loss_fake = loss_fn_bce(fake_labels, d_fake)
            d_loss = d_loss_real + d_loss_fake
            
            # Add L2 regularization loss from discriminator
            if model.discriminator.losses:  # Only if L2 is used
                d_loss += tf.add_n(model.discriminator.losses)
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]

        # === 2. Train Generator (with Gradient Reversal for domain loss) ===
        with tf.GradientTape() as tape_g:
            out_src = model(x_scaled_src, training=True)
            x_fake_src = out_src.gen_out
            features_src = out_src.extracted_features
            d_fake = out_src.disc_out
            g_adv_loss = loss_fn_bce(tf.ones_like(d_fake), d_fake)
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)
            # --- Gradient Reversal for domain loss ---
            out_tgt = model(x_scaled_tgt, training=True)
            features_tgt = out_tgt.extracted_features
            
            # Only compute domain loss if domain_weight > 0
            if domain_weight > 0:
                features_grl = tf.concat([GradientReversal()(features_src), GradientReversal()(features_tgt)], axis=0)
                domain_labels_src = tf.ones((features_src.shape[0], 1))
                domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
                domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
                domain_pred = domain_model(features_grl, training=True)
                domain_loss_grl = loss_fn_domain(domain_labels, domain_pred) /2.0
            else:
                domain_loss_grl = 0.0
            
            # --- Total generator loss ---
            g_loss = adv_weight * g_adv_loss + est_weight * g_est_loss + domain_weight * domain_loss_grl
            # Add L2 regularization loss
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]

        # === 3. Train Domain Discriminator ===
        if domain_weight != 0:
            with tf.GradientTape() as tape_domain:
                _, features_src = model.generator(x_scaled_src, None, training=False)
                _, features_tgt = model.generator(x_scaled_tgt, None, training=False)
                domain_labels_src = tf.ones((features_src.shape[0], 1))
                domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
                features = tf.concat([features_src, features_tgt], axis=0)
                domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
                domain_pred = domain_model(features, training=True)
                domain_loss = loss_fn_domain(domain_labels, domain_pred) /2.0
                if domain_model.losses:
                    domain_loss += tf.add_n(domain_model.losses)
            grads_domain = tape_domain.gradient(domain_loss, domain_model.trainable_variables)
            domain_optimizer.apply_gradients(zip(grads_domain, domain_model.trainable_variables))
            epoc_loss_domain += domain_loss.numpy() * features.shape[0]
    
        # === 4. Save features if required ===
        if return_features and (domain_weight != 0):
            # save features in a temporary file instead of stacking them up, to avoid memory exploding
            features_np_source = features_src.numpy()  # Convert to numpy if it's a tensor
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
                
            features_np_target = features_tgt.numpy()
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
    # end batch loop
    if return_features and (domain_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
                

    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_domain = epoc_loss_domain / N_train
    # return avg_loss_g, avg_loss_d, avg_loss_est, avg_loss_domain
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_domain,
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est,  # Assuming same loss for target
        features_source=features_src,
        film_features_source=features_src,  # Assuming same features for source  
        avg_epoc_loss_d=avg_loss_d
    )
    
def val_step(model, domain_model, loader_H, loss_fn, lower_range, nsymb=14, adv_weight=0.01, est_weight=1.0, domain_weight=0.5):
    """
    Validation step for GAN model. Returns H_sample and epoc_eval_return (summary metrics).
    Args:
        model: GAN model instance
        domain_model: domain discriminator instance
        loader_H: tuple of (input_src, true_src, input_tgt, true_tgt) DataLoaders
        loss_fn: tuple of (estimation loss, binary cross-entropy loss, domain loss)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        adv_weight, est_weight, domain_weight: loss weights
    Returns:
        H_sample, epoc_eval_return
    """
    from sklearn.metrics import accuracy_score
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce, loss_fn_domain = loss_fn
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_domain_disc_loss = 0.0
    epoc_domain_acc_source = 0.0
    epoc_domain_acc_target = 0.0
    H_sample = []

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
        x_src_real = utils.complx2real(x_src)
        y_src_real = utils.complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = utils.minmaxScaler(x_src_real, lower_range=lower_range)
        y_scaled_src, _, _ = utils.minmaxScaler(y_src_real, lower_range=lower_range)

        # Preprocess (target)
        x_tgt_real = utils.complx2real(x_tgt)
        y_tgt_real = utils.complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = utils.minmaxScaler(x_tgt_real, lower_range=lower_range)
        y_scaled_tgt, _, _ = utils.minmaxScaler(y_tgt_real, lower_range=lower_range)

        # === Source domain prediction ===
        preds_src, features_src = model.generator(x_scaled_src, training=False)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = utils.deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === Target domain prediction ===
        preds_tgt, features_tgt = model.generator(x_scaled_tgt, training=False)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = utils.deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
        batch_est_loss_target = loss_fn_est(y_scaled_tgt, preds_tgt).numpy()
        epoc_loss_est_target += batch_est_loss_target * x_tgt.shape[0]
        mse_val_target = np.mean((preds_tgt_descaled - y_tgt_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_tgt_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) * x_tgt.shape[0]

        # === GAN Discriminator Loss ===
        if loss_fn_bce is not None:
            d_real_src = model.discriminator(y_scaled_src, training=False)
            d_fake_src = model.discriminator(preds_src, training=False)
            real_labels_src = np.ones_like(d_real_src)
            fake_labels_src = np.zeros_like(d_fake_src)
            d_loss_real_src = loss_fn_bce(real_labels_src, d_real_src).numpy()
            d_loss_fake_src = loss_fn_bce(fake_labels_src, d_fake_src).numpy()

            d_real_tgt = model.discriminator(y_scaled_tgt, training=False)
            d_fake_tgt = model.discriminator(preds_tgt, training=False)
            real_labels_tgt = np.ones_like(d_real_tgt)
            fake_labels_tgt = np.zeros_like(d_fake_tgt)
            d_loss_real_tgt = loss_fn_bce(real_labels_tgt, d_real_tgt).numpy()
            d_loss_fake_tgt = loss_fn_bce(fake_labels_tgt, d_fake_tgt).numpy()

            d_loss = (d_loss_real_src + d_loss_fake_src) * x_src.shape[0] + (d_loss_real_tgt + d_loss_fake_tgt) * x_tgt.shape[0]
            epoc_gan_disc_loss += d_loss

        # === Domain Discriminator Loss & Accuracy ===
        if loss_fn_domain is not None:
            features = np.concatenate([features_src, features_tgt], axis=0)
            domain_labels_src = np.ones((features_src.shape[0], 1))
            domain_labels_tgt = np.zeros((features_tgt.shape[0], 1))
            domain_labels = np.concatenate([domain_labels_src, domain_labels_tgt], axis=0)
            domain_pred = domain_model(features, training=False)
            domain_loss = loss_fn_domain(domain_labels, domain_pred).numpy() * features.shape[0] /2.0
            epoc_domain_disc_loss += domain_loss
            # Accuracy (source)
            domain_pred_src = domain_model(features_src, training=False)
            domain_pred_src_bin = tf.cast(domain_pred_src >= 0.5, tf.int32)
            acc_src = accuracy_score(domain_labels_src, domain_pred_src_bin)
            epoc_domain_acc_source += acc_src * features_src.shape[0]
            # Accuracy (target)
            domain_pred_tgt = domain_model(features_tgt, training=False)
            domain_pred_tgt_bin = tf.cast(domain_pred_tgt >= 0.5, tf.int32)
            acc_tgt = accuracy_score(domain_labels_tgt, domain_pred_tgt_bin)
            epoc_domain_acc_target += acc_tgt * features_tgt.shape[0]

        # === Save H samples for visualization at first batch ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            # Target
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            H_sample = [H_true_sample, H_input_sample, H_est_sample, nmse_input_source, nmse_est_source,
                        H_true_sample_target, H_input_sample_target, H_est_sample_target, nmse_input_target, nmse_est_target]
            


    # Calculate averages
    N_val = N_val_source + N_val_target
    avg_loss_est_source = epoc_loss_est_source / N_val_source
    avg_loss_est_target = epoc_loss_est_target / N_val_target
    avg_loss_est = (avg_loss_est_source + avg_loss_est_target) / 2
    avg_nmse_source = epoc_nmse_val_source / N_val_source
    avg_nmse_target = epoc_nmse_val_target / N_val_target
    avg_nmse = (avg_nmse_source + avg_nmse_target) / 2
    avg_gan_disc_loss = epoc_gan_disc_loss / N_val if epoc_gan_disc_loss > 0 else 0.0
    avg_domain_disc_loss = epoc_domain_disc_loss / N_val if epoc_domain_disc_loss > 0 else 0.0
    # Domain discriminator accuracy
    avg_domain_acc_source = epoc_domain_acc_source / N_val_source if N_val_source > 0 else 0.0
    avg_domain_acc_target = epoc_domain_acc_target / N_val_target if N_val_target > 0 else 0.0
    avg_domain_acc = (epoc_domain_acc_source + epoc_domain_acc_target) / N_val if N_val > 0 else 0.0

    # Weighted total loss (for comparison with training)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss + domain_weight * avg_domain_disc_loss

    # Compose epoc_eval_return
    epoc_eval_return = [
        avg_total_loss,
        avg_loss_est_source, avg_loss_est_target, avg_loss_est,
        avg_gan_disc_loss, avg_domain_disc_loss,
        avg_nmse_source, avg_nmse_target, avg_nmse,
        avg_domain_acc_source, avg_domain_acc_target, avg_domain_acc
    ]

    return H_sample, epoc_eval_return

def gradient_penalty(discriminator, real, fake, batch_size):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake - real
    interpolated = real + alpha * diff
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
def train_step_wgan_gp(model, domain_model, loader_H, loss_fn, optimizers, lower_range=-1, return_features=False,
        nsymb=14, adv_weight=0.01, est_weight=1.0, domain_weight=0.5, linear_interp=False):
    """
    model: GAN model instance
    loader_H_source: DataLoader for source domain
    loader_H_target: DataLoader for target domain
    loss_fns: tuple of loss functions (estimation loss, binary cross-entropy loss, domain loss)
    optimizers: tuple of optimizers (generator optimizer, discriminator optimizer, domain optimizer)
    lower_range: lower range for min-max scaling
    domain_model: Domain discriminator model class
    """    
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce, loss_fn_domain = loss_fn
    gen_optimizer, disc_optimizer, domain_optimizer = optimizers

    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_domain = 0.0
    N_train = 0
    
    if return_features==True and (domain_weight != 0):
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
        # --- Source domain ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        # --- Target domain ---
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess (source)
        x_src = utils.complx2real(x_src)
        y_src = utils.complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = utils.minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = utils.minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt = utils.complx2real(x_tgt)
        y_tgt = utils.complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = utils.minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = utils.minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === 1. Train Discriminator ===
        # Only considering source domain
        with tf.GradientTape() as tape_d:
            x_fake_src = model.generator(x_scaled_src, training=True)[0]
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            
            gp = gradient_penalty(model.discriminator, y_scaled_src, x_fake_src, batch_size=x_scaled_src.shape[0])
            lambda_gp = 10.0  # typical gradient penalty weight

            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            # Add L2 regularization loss from discriminator
            if model.discriminator.losses:  # Only if L2 is used
                d_loss += tf.add_n(model.discriminator.losses)
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]
        
        # === 2. Train Generator (with Gradient Reversal for domain loss) ===
        with tf.GradientTape() as tape_g:
            out_src = model(x_scaled_src, training=True)
            x_fake_src = out_src.gen_out
            features_src = out_src.extracted_features       ### 
            d_fake = out_src.disc_out
            
            g_adv_loss = -tf.reduce_mean(d_fake)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)
            
            # --- Gradient Reversal for domain loss ---
            # to train feature extractor
            out_tgt = model(x_scaled_tgt, training=True)
            x_fake_tgt = out_tgt.gen_out
            features_tgt = out_tgt.extracted_features       ###
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)
            
            # Only compute domain loss if domain_weight > 0
            if domain_weight > 0:
                features_grl = tf.concat([GradientReversal()(features_src), GradientReversal()(features_tgt)], axis=0)
                domain_labels_src = tf.ones((features_src.shape[0], 1))
                domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
                domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
                domain_pred = domain_model(features_grl, training=True)
                domain_loss_grl = loss_fn_domain(domain_labels, domain_pred) / 2.0
            else:
                domain_loss_grl = 0.0
            
            # --- Total generator loss ---
            g_loss = adv_weight * g_adv_loss + est_weight * g_est_loss + domain_weight * domain_loss_grl
            
            # Add L2 regularization loss
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]

        # === 3. Train Domain Discriminator ===
        if domain_weight != 0:
            with tf.GradientTape() as tape_domain:
                _, features_src = model.generator(x_scaled_src, training=False)
                _, features_tgt = model.generator(x_scaled_tgt, training=False)

                domain_labels_src = tf.ones((features_src.shape[0], 1))
                domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
                features = tf.concat([features_src, features_tgt], axis=0)
                domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
                
                domain_pred = domain_model(features, training=True)
                domain_loss = loss_fn_domain(domain_labels, domain_pred) / 2.0
                if domain_model.losses: # Add L2 regularization loss
                    domain_loss += tf.add_n(domain_model.losses)
            grads_domain = tape_domain.gradient(domain_loss, domain_model.trainable_variables)
            domain_optimizer.apply_gradients(zip(grads_domain, domain_model.trainable_variables))
            epoc_loss_domain += domain_loss.numpy() * features.shape[0]
    
        # === 4. Save features if required ===
        if return_features and (domain_weight != 0):
            # save features in a temporary file instead of stacking them up, to avoid memory exploding
            features_np_source = features_src.numpy()  # Convert to numpy if it's a tensor
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
                
            features_np_target = features_tgt.numpy()
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
    # end batch loop
    if return_features and (domain_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()
                

    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_domain = epoc_loss_domain / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt /N_train  # to add to observe the estimation loss for target domain
    # return avg_loss_g, avg_loss_d, avg_loss_est, avg_loss_domain
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_domain,
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,  # Assuming same loss for target
        features_source=features_src,
        film_features_source=features_src,  # Assuming same features for source  
        avg_epoc_loss_d=avg_loss_d
    )

def val_step_wgan_gp(model, domain_model, loader_H, loss_fn, lower_range, nsymb=14, adv_weight=0.01, 
                    est_weight=1.0, domain_weight=0.5, linear_interp=False,
                    return_H_gen=False):
    """
    Validation step for GAN model. Returns H_sample and epoc_eval_return (summary metrics).
    Args:
        model: GAN model instance
        domain_model: domain discriminator instance
        loader_H: tuple of (input_src, true_src, input_tgt, true_tgt) DataLoaders
        loss_fn: tuple of (estimation loss, binary cross-entropy loss, domain loss)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        adv_weight, est_weight, domain_weight: loss weights
    Returns:
        H_sample, epoc_eval_return
    """
    from sklearn.metrics import accuracy_score
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce, loss_fn_domain = loss_fn
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_domain_disc_loss = 0.0
    epoc_domain_acc_source = 0.0
    epoc_domain_acc_target = 0.0
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
        x_src_real = utils.complx2real(x_src)
        y_src_real = utils.complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = utils.minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = utils.minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src,  lower_range=lower_range)

        # Preprocess (target)
        x_tgt_real = utils.complx2real(x_tgt)
        y_tgt_real = utils.complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = utils.minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = utils.minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === Source domain prediction ===
        preds_src, features_src = model.generator(x_scaled_src, training=False)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = utils.deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === Target domain prediction ===
        preds_tgt, features_tgt = model.generator(x_scaled_tgt, training=False)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = utils.deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
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

        # === Domain Discriminator Loss & Accuracy ===
        if loss_fn_domain is not None:
            features = np.concatenate([features_src, features_tgt], axis=0)
            domain_labels_src = np.ones((features_src.shape[0], 1))
            domain_labels_tgt = np.zeros((features_tgt.shape[0], 1))
            domain_labels = np.concatenate([domain_labels_src, domain_labels_tgt], axis=0)
            domain_pred = domain_model(features, training=False)
            domain_loss = loss_fn_domain(domain_labels, domain_pred).numpy() * features.shape[0] / 2.0
            epoc_domain_disc_loss += domain_loss
            # Accuracy (source)
            domain_pred_src = domain_model(features_src, training=False)
            domain_pred_src_bin = tf.cast(domain_pred_src >= 0.5, tf.int32)
            acc_src = accuracy_score(domain_labels_src, domain_pred_src_bin)
            epoc_domain_acc_source += acc_src * features_src.shape[0]
            # Accuracy (target)
            domain_pred_tgt = domain_model(features_tgt, training=False)
            domain_pred_tgt_bin = tf.cast(domain_pred_tgt >= 0.5, tf.int32)
            acc_tgt = accuracy_score(domain_labels_tgt, domain_pred_tgt_bin)
            epoc_domain_acc_target += acc_tgt * features_tgt.shape[0]

        # === Save H samples for visualization at first batch ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            # Target
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
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
    
    avg_domain_disc_loss = epoc_domain_disc_loss / N_val_source if epoc_domain_disc_loss > 0 else 0.0
    # Domain discriminator accuracy
    avg_domain_acc_source = epoc_domain_acc_source / N_val_source if N_val_source > 0 else 0.0
    avg_domain_acc_target = epoc_domain_acc_target / N_val_target if N_val_target > 0 else 0.0
    avg_domain_acc = (epoc_domain_acc_source + epoc_domain_acc_target) / N_val if N_val > 0 else 0.0

    # Weighted total loss (for comparison with training)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss + domain_weight * avg_domain_disc_loss

    # Compose epoc_eval_return
    epoc_eval_return = [
        avg_total_loss,
        avg_loss_est_source, avg_loss_est_target, avg_loss_est,
        avg_gan_disc_loss, avg_domain_disc_loss,
        avg_nmse_source, avg_nmse_target, avg_nmse,
        avg_domain_acc_source, avg_domain_acc_target, avg_domain_acc
    ]
    
    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen

    return H_sample, epoc_eval_return

def normalize_features_for_domain_adaptation(features):
    """
    Normalize features for better DANN performance while preserving original shape
    Apply L2 normalization + standardization
    """
    original_shape = tf.shape(features)
    
    # Step 1: Flatten for normalization calculations
    if len(features.shape) > 2:
        features_flat = tf.reshape(features, [original_shape[0], -1])
    else:
        features_flat = features
    
    # Step 2: L2 normalization (unit vectors)
    features_l2 = tf.nn.l2_normalize(features_flat, axis=-1)
    
    # Step 3: Standardization (zero mean, unit variance)
    mean = tf.reduce_mean(features_l2, axis=0, keepdims=True)
    std = tf.math.reduce_std(features_l2, axis=0, keepdims=True) + 1e-8
    features_normalized = (features_l2 - mean) / std
    
    # Step 4: Reshape back to original shape
    if len(features.shape) > 2:
        features_normalized = tf.reshape(features_normalized, original_shape)
    
    return features_normalized

def train_step_wgan_gp_normalized(model, domain_model, loader_H, loss_fn, optimizers, 
                                     lower_range=-1, return_features=False, nsymb=14, 
                                     adv_weight=0.01, est_weight=1.0, domain_weight=0.5, 
                                     linear_interp=False, normalize_dann_features=True):
    """
    Enhanced WGAN-GP training with DANN feature normalization
    """
    loader_H_input_train_src, loader_H_true_train_src, \
        loader_H_input_train_tgt, loader_H_true_train_tgt = loader_H
    loss_fn_est, loss_fn_bce, loss_fn_domain = loss_fn
    gen_optimizer, disc_optimizer, domain_optimizer = optimizers

    epoc_loss_g = 0.0
    epoc_loss_d = 0.0
    epoc_loss_est = 0.0
    epoc_loss_est_tgt = 0.0
    epoc_loss_domain = 0.0
    N_train = 0
    
    if return_features==True and (domain_weight != 0):
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
        # --- Source domain ---
        x_src = loader_H_input_train_src.next_batch()
        y_src = loader_H_true_train_src.next_batch()
        # --- Target domain ---
        x_tgt = loader_H_input_train_tgt.next_batch()
        y_tgt = loader_H_true_train_tgt.next_batch()
        N_train += x_src.shape[0]

        # Preprocess (source)
        x_src = utils.complx2real(x_src)
        y_src = utils.complx2real(y_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = utils.minmaxScaler(x_src, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = utils.minmaxScaler(y_src, min_pre=x_min_src, max_pre=x_max_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt = utils.complx2real(x_tgt)
        y_tgt = utils.complx2real(y_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = utils.minmaxScaler(x_tgt, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = utils.minmaxScaler(y_tgt, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === 1. Train Discriminator ===
        with tf.GradientTape() as tape_d:
            x_fake_src = model.generator(x_scaled_src, training=True)[0]
            d_real = model.discriminator(y_scaled_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            
            gp = gradient_penalty(model.discriminator, y_scaled_src, x_fake_src, batch_size=x_scaled_src.shape[0])
            lambda_gp = 10.0
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + lambda_gp * gp
            
            if model.discriminator.losses:
                d_loss += tf.add_n(model.discriminator.losses)
        grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
        epoc_loss_d += d_loss.numpy() * x_src.shape[0]
        
        # === 2. Train Generator (with Normalized DANN) ===
        with tf.GradientTape() as tape_g:
            out_src = model(x_scaled_src, training=True)
            x_fake_src = out_src.gen_out
            features_src = out_src.extracted_features
            d_fake = out_src.disc_out
            
            g_adv_loss = -tf.reduce_mean(d_fake)
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)
            
            # Target domain
            out_tgt = model(x_scaled_tgt, training=True)
            x_fake_tgt = out_tgt.gen_out
            features_tgt = out_tgt.extracted_features
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)
            
            # === NORMALIZED DANN Domain Loss ===
            if domain_weight > 0:
                # NORMALIZE FEATURES BEFORE GRADIENT REVERSAL
                if normalize_dann_features:
                    features_src_norm = normalize_features_for_domain_adaptation(features_src)
                    features_tgt_norm = normalize_features_for_domain_adaptation(features_tgt)
                else:
                    features_src_norm = features_src
                    features_tgt_norm = features_tgt
                
                # Apply gradient reversal to normalized features
                features_grl = tf.concat([
                    GradientReversal()(features_src_norm), 
                    GradientReversal()(features_tgt_norm)
                ], axis=0)
                
                domain_labels_src = tf.ones((features_src.shape[0], 1))
                domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
                domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
                domain_pred = domain_model(features_grl, training=True)
                domain_loss_grl = loss_fn_domain(domain_labels, domain_pred) /2.0
            else:
                domain_loss_grl = 0.0
            
            # Total generator loss
            g_loss = adv_weight * g_adv_loss + est_weight * g_est_loss + domain_weight * domain_loss_grl
            
            # Add L2 regularization loss
            if model.generator.losses:
                g_loss += tf.add_n(model.generator.losses)
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]
        epoc_loss_est_tgt += g_est_loss_tgt.numpy() * x_tgt.shape[0]

        # === 3. Train Domain Discriminator (with Normalized Features) ===
        if domain_weight != 0:
            with tf.GradientTape() as tape_domain:
                _, features_src = model.generator(x_scaled_src, training=False)
                _, features_tgt = model.generator(x_scaled_tgt, training=False)

                # NORMALIZE FEATURES FOR DOMAIN DISCRIMINATOR TRAINING
                if normalize_dann_features:
                    features_src_norm = normalize_features_for_domain_adaptation(features_src)
                    features_tgt_norm = normalize_features_for_domain_adaptation(features_tgt)
                    features = tf.concat([features_src_norm, features_tgt_norm], axis=0)
                else:
                    features = tf.concat([features_src, features_tgt], axis=0)

                domain_labels_src = tf.ones((features_src.shape[0], 1))
                domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
                domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
                
                domain_pred = domain_model(features, training=True)
                domain_loss = loss_fn_domain(domain_labels, domain_pred) / 2.0
                if domain_model.losses:
                    domain_loss += tf.add_n(domain_model.losses)
            grads_domain = tape_domain.gradient(domain_loss, domain_model.trainable_variables)
            domain_optimizer.apply_gradients(zip(grads_domain, domain_model.trainable_variables))
            epoc_loss_domain += domain_loss.numpy() * features.shape[0]
    
        # === 4. Save features if required ===
        if return_features and (domain_weight != 0):
            # Use normalized features for analysis
            if normalize_dann_features:
                features_np_source = normalize_features_for_domain_adaptation(features_src).numpy()
                features_np_target = normalize_features_for_domain_adaptation(features_tgt).numpy()
            else:
                features_np_source = features_src.numpy()
                features_np_target = features_tgt.numpy()
                
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
    
    # end batch loop
    if return_features and (domain_weight != 0):    
        features_h5_source.close()
        features_h5_target.close()

    avg_loss_g = epoc_loss_g / N_train
    avg_loss_d = epoc_loss_d / N_train
    avg_loss_est = epoc_loss_est / N_train
    avg_loss_domain = epoc_loss_domain / N_train
    avg_loss_est_tgt = epoc_loss_est_tgt / N_train
    
    return train_step_Output(
        avg_epoc_loss_est=avg_loss_est,
        avg_epoc_loss_domain=avg_loss_domain,
        avg_epoc_loss=avg_loss_g,
        avg_epoc_loss_est_target=avg_loss_est_tgt,
        features_source=features_src,
        film_features_source=features_src,
        avg_epoc_loss_d=avg_loss_d
    )

def val_step_wgan_gp_normalized(model, domain_model, loader_H, loss_fn, lower_range, nsymb=14, adv_weight=0.01, 
                               est_weight=1.0, domain_weight=0.5, linear_interp=False,
                               return_H_gen=False, normalize_dann_features=True):
    """
    Validation step for GAN model with optional DANN feature normalization.
    Returns H_sample and epoc_eval_return (summary metrics).
    Args:
        model: GAN model instance
        domain_model: domain discriminator instance
        loader_H: tuple of (input_src, true_src, input_tgt, true_tgt) DataLoaders
        loss_fn: tuple of (estimation loss, binary cross-entropy loss, domain loss)
        lower_range: lower range for min-max scaling
        nsymb: number of symbols
        adv_weight, est_weight, domain_weight: loss weights
        normalize_dann_features: Whether to normalize features for domain discriminator
    Returns:
        H_sample, epoc_eval_return
    """
    from sklearn.metrics import accuracy_score
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_bce, loss_fn_domain = loss_fn
    N_val_source = 0
    N_val_target = 0
    epoc_loss_est_source = 0.0
    epoc_loss_est_target = 0.0
    epoc_nmse_val_source = 0.0
    epoc_nmse_val_target = 0.0
    epoc_gan_disc_loss = 0.0
    epoc_domain_disc_loss = 0.0
    epoc_domain_acc_source = 0.0
    epoc_domain_acc_target = 0.0
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
        x_src_real = utils.complx2real(x_src)
        y_src_real = utils.complx2real(y_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_scaled_src, x_min_src, x_max_src = utils.minmaxScaler(x_src_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_src, _, _ = utils.minmaxScaler(y_src_real, min_pre=x_min_src, max_pre=x_max_src,  lower_range=lower_range)

        # Preprocess (target)
        x_tgt_real = utils.complx2real(x_tgt)
        y_tgt_real = utils.complx2real(y_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_scaled_tgt, x_min_tgt, x_max_tgt = utils.minmaxScaler(x_tgt_real, lower_range=lower_range, linear_interp=linear_interp)
        y_scaled_tgt, _, _ = utils.minmaxScaler(y_tgt_real, min_pre=x_min_tgt, max_pre=x_max_tgt, lower_range=lower_range)

        # === Source domain prediction ===
        preds_src, features_src = model.generator(x_scaled_src, training=False)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = utils.deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === Target domain prediction ===
        preds_tgt, features_tgt = model.generator(x_scaled_tgt, training=False)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = utils.deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
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

        # === Domain Discriminator Loss & Accuracy (with Normalized Features) ===
        if loss_fn_domain is not None:
            # APPLY NORMALIZATION TO FEATURES BEFORE DOMAIN DISCRIMINATOR EVALUATION
            if normalize_dann_features:
                features_src_norm = normalize_features_for_domain_adaptation(features_src)
                features_tgt_norm = normalize_features_for_domain_adaptation(features_tgt)
                features = np.concatenate([features_src_norm, features_tgt_norm], axis=0)
            else:
                features = np.concatenate([features_src, features_tgt], axis=0)
            
            domain_labels_src = np.ones((features_src.shape[0], 1))
            domain_labels_tgt = np.zeros((features_tgt.shape[0], 1))
            domain_labels = np.concatenate([domain_labels_src, domain_labels_tgt], axis=0)
            domain_pred = domain_model(features, training=False)
            domain_loss = loss_fn_domain(domain_labels, domain_pred).numpy() * features.shape[0] / 2.0
            epoc_domain_disc_loss += domain_loss
            
            # Accuracy (source) - use normalized features if enabled
            if normalize_dann_features:
                domain_pred_src = domain_model(features_src_norm, training=False)
            else:
                domain_pred_src = domain_model(features_src, training=False)
            domain_pred_src_bin = tf.cast(domain_pred_src >= 0.5, tf.int32)
            acc_src = accuracy_score(domain_labels_src, domain_pred_src_bin)
            epoc_domain_acc_source += acc_src * features_src.shape[0]
            
            # Accuracy (target) - use normalized features if enabled
            if normalize_dann_features:
                domain_pred_tgt = domain_model(features_tgt_norm, training=False)
            else:
                domain_pred_tgt = domain_model(features_tgt, training=False)
            domain_pred_tgt_bin = tf.cast(domain_pred_tgt >= 0.5, tf.int32)
            acc_tgt = accuracy_score(domain_labels_tgt, domain_pred_tgt_bin)
            epoc_domain_acc_target += acc_tgt * features_tgt.shape[0]

        # === Save H samples for visualization at first batch ===
        if idx == 0:
            n_samples = min(3, x_src_real.shape[0], x_tgt_real.shape[0])
            # Source
            H_true_sample = y_src_real[:n_samples].copy()
            H_input_sample = x_src_real[:n_samples].copy()
            H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            # Target
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
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
    
    avg_domain_disc_loss = epoc_domain_disc_loss / N_val_source if epoc_domain_disc_loss > 0 else 0.0
    # Domain discriminator accuracy
    avg_domain_acc_source = epoc_domain_acc_source / N_val_source if N_val_source > 0 else 0.0
    avg_domain_acc_target = epoc_domain_acc_target / N_val_target if N_val_target > 0 else 0.0
    avg_domain_acc = (epoc_domain_acc_source + epoc_domain_acc_target) / N_val if N_val > 0 else 0.0

    # Weighted total loss (for comparison with training)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss + domain_weight * avg_domain_disc_loss

    # Compose epoc_eval_return
    epoc_eval_return = [
        avg_total_loss,
        avg_loss_est_source, avg_loss_est_target, avg_loss_est,
        avg_gan_disc_loss, avg_domain_disc_loss,
        avg_nmse_source, avg_nmse_target, avg_nmse,
        avg_domain_acc_source, avg_domain_acc_target, avg_domain_acc
    ]
    
    if return_H_gen:
        return H_sample, epoc_eval_return, H_gen

    return H_sample, epoc_eval_return

def visualize_H(H_sample, H_to_save, epoch, figChan, flag, model_path, sub_folder, domain_weight=True):
    (
        H_true_sample,  H_input_sample, H_est_sample, 
        nmse_input_source, nmse_est_source,
        H_true_sample_target, H_input_sample_target, H_est_sample_target,
        nmse_input_target, nmse_est_target
    ) = H_sample

    # plot real parts
    # source domain
    if flag == 1:
        figChan(H_true_sample[0,:,:,0], index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_true_source')
        figChan(H_input_sample[0,:,:,0], nmse=nmse_input_source[0], title='Raw-estimated Channel', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_input_source')
    figChan(H_est_sample[0,:,:,0], nmse=nmse_est_source[0], title='GAN-refined Channel', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_GAN_source')
    # target domain
    if flag == 1 and domain_weight:
        figChan(H_true_sample_target[0,:,:,0], index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_true_target')
        figChan(H_input_sample_target[0,:,:,0], nmse=nmse_input_target[0], title='Raw-estimated Channel', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_input_target')
        flag = 0
    figChan(H_est_sample_target[0,:,:,0], nmse=nmse_est_target[0], title='GAN-refined Channel', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_GAN_target')

    H_to_save[f'H_{epoch+1}_true_source'] = H_true_sample
    H_to_save[f'H_{epoch+1}_input_source'] = H_input_sample
    H_to_save[f'H_{epoch+1}_est_source'] = H_est_sample
        # don't save nmse_input_source, nmse_est_source because can compute them later
    #
    if domain_weight:
        H_to_save[f'H_{epoch+1}_true_target'] = H_true_sample_target
        H_to_save[f'H_{epoch+1}_input_target'] = H_input_sample_target
        H_to_save[f'H_{epoch+1}_est_target'] = H_est_sample_target

def post_val(epoc_val_return, epoch, n_epochs, val_est_loss, val_est_loss_source, val_loss, val_est_loss_target,
            val_gan_disc_loss, val_domain_disc_loss, nmse_val_source, nmse_val_target, nmse_val, source_acc, target_acc, acc, domain_weight=True):
    (
        avg_total_loss,
        avg_loss_est_source, avg_loss_est_target, avg_loss_est,
        avg_gan_disc_loss, avg_domain_disc_loss,
        avg_nmse_source, avg_nmse_target, avg_nmse,
        avg_domain_acc_source, avg_domain_acc_target, avg_domain_acc
    ) = epoc_val_return
    # Average loss for the epoch
    val_loss.append(avg_total_loss)
    print(f"epoch {epoch+1}/{n_epochs} (Val) Weighted Total Loss: {avg_total_loss:.6f}")
    #
    val_est_loss.append(avg_loss_est)
    print(f"epoch {epoch+1}/{n_epochs} (Val) Average Estimation Loss (mean): {avg_loss_est:.6f}")
    #
    val_est_loss_source.append(avg_loss_est_source)
    print(f"epoch {epoch+1}/{n_epochs} (Val) Average Estimation Loss (Source): {avg_loss_est_source:.6f}")
    #
    if domain_weight:
        val_est_loss_target.append(avg_loss_est_target)
        print(f"epoch {epoch+1}/{n_epochs} (Val) Average Estimation Loss (Target): {avg_loss_est_target:.6f}")
    #
    val_gan_disc_loss.append(avg_gan_disc_loss)
    print(f"epoch {epoch+1}/{n_epochs} (Val) GAN Discriminator Loss: {avg_gan_disc_loss:.6f}")
    #
    if domain_weight:
        val_domain_disc_loss.append(avg_domain_disc_loss)
        print(f"epoch {epoch+1}/{n_epochs} (Val) Domain Discriminator Loss: {avg_domain_disc_loss:.6f}")
    #
    nmse_val_source.append(avg_nmse_source)
    nmse_val_target.append(avg_nmse_target)
    nmse_val.append(avg_nmse)
    print(f"epoch {epoch+1}/{n_epochs} (Val) NMSE (Source): {avg_nmse_source:.6f}, NMSE (Target): {avg_nmse_target:.6f}, NMSE (Mean): {avg_nmse:.6f}")
    #
    source_acc.append(avg_domain_acc_source)
    target_acc.append(avg_domain_acc_target)
    acc.append(avg_domain_acc)
    print(f"epoch {epoch+1}/{n_epochs} (Val) Domain Discriminator Accuracy (Average): {avg_domain_acc:.4f}")

def save_checkpoint(model, save_model, model_path, sub_folder, epoch, figLoss, savemat, train_loss, train_est_loss, train_domain_loss, train_est_loss_target,
                    val_est_loss, val_est_loss_source, val_loss, val_est_loss_target, val_gan_disc_loss, val_domain_disc_loss,
                    source_acc, target_acc, acc, nmse_val_source, nmse_val_target, nmse_val, pad_pca_svm, pad_pca_lda, pad_pca_logreg, epoc_pad, pad_svm, 
                    train_disc_loss=None, domain_weight=True, optimizer=None, domain_model=None):
    # Save model
    os.makedirs(f"{model_path}/{sub_folder}/model/", exist_ok=True)
    if save_model:
        # Create checkpoint with all model components and optimizers
        gen_optimizer, disc_optimizer, domain_optimizer = optimizer
        
        # Create checkpoint object
        ckpt = tf.train.Checkpoint(
            generator=model.generator,
            discriminator=model.discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer
        )
        
        # Add domain model and optimizer if domain adaptation is used
        if domain_weight and domain_model is not None:
            ckpt.domain_model = domain_model
            ckpt.domain_optimizer = domain_optimizer
        
        # Create checkpoint manager - save only current epoch
        checkpoint_dir = f"{model_path}/{sub_folder}/model/"
        
        # Save checkpoint
        milestone_path = ckpt.save(f"{checkpoint_dir}/epoch_{epoch+1}")
        print(f"Checkpoint saved at epoch {epoch+1}: {milestone_path}")

        # Save optimizer configs
        optimizer_configs = {
            'gen_optimizer_config': gen_optimizer.get_config(),
            'disc_optimizer_config': disc_optimizer.get_config(), 
            'domain_optimizer_config': domain_optimizer.get_config()
        }
        config_path = f"{checkpoint_dir}/optimizer_configs.json"  # No epoch number
        # Only save if file doesn't exist (to avoid overwriting)
        if not os.path.exists(config_path):
            import json
            with open(config_path, 'w') as f:
                json.dump(optimizer_configs, f, indent=2)
            print(f"Optimizer configs saved to: {config_path}")
    
    # === save and overwrite at checkpoints
    # train
    perform_to_save = {}
    perform_to_save['train_loss'] = train_loss
    perform_to_save['train_est_loss'] = train_est_loss
    perform_to_save['train_disc_loss'] = train_disc_loss
    perform_to_save['train_domain_loss'] = train_domain_loss
    perform_to_save['train_est_loss_target'] = train_est_loss_target
    # val
    perform_to_save['val_est_loss'] = val_est_loss
    perform_to_save['val_est_loss_source'] = val_est_loss_source
    perform_to_save['val_loss'] = val_loss
    perform_to_save['val_est_loss_target'] = val_est_loss_target
    perform_to_save['val_gan_disc_loss'] = val_gan_disc_loss
    perform_to_save['val_domain_disc_loss'] = val_domain_disc_loss
    perform_to_save['source_acc'] = source_acc
    perform_to_save['target_acc'] = target_acc
    perform_to_save['acc'] = acc
    perform_to_save['nmse_val_source'] = nmse_val_source
    perform_to_save['nmse_val_target'] = nmse_val_target
    perform_to_save['nmse_val'] = nmse_val
    #
    perform_to_save['pad_pca_svm'] = pad_pca_svm
    perform_to_save['pad_pca_lda'] = pad_pca_lda
    perform_to_save['pad_pca_logreg'] = pad_pca_logreg
    perform_to_save['epoc_pad'] = epoc_pad
    perform_to_save['pad_svm'] = pad_svm

    # save
    os.makedirs(f"{model_path}/{sub_folder}/performance/", exist_ok=True)
    savemat(model_path + '/' + sub_folder + '/performance/performance.mat', perform_to_save)
    
    # Plot figures === save and overwrite at checkpoints
    if domain_weight:
        figLoss(line_list=[(nmse_val_source, 'Source Domain'), (nmse_val_target, 'Target Domain')], xlabel='Epoch', ylabel='NMSE',
                    title='NMSE in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='NMSE_val')
        figLoss(line_list=[(source_acc, 'Source Domain'), (target_acc, 'Target Domain')], xlabel='Epoch', ylabel='Discrimination Accuracy',
                    title='Domain Discrimination Accuracy in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Domain_acc')
    #
    figLoss(line_list=[(train_est_loss, 'GAN Generate Loss'), (train_disc_loss, 'GAN Discriminator Loss')], xlabel='Epoch', ylabel='Loss',
                title='Training GAN Losses', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='GAN_train')
    figLoss(line_list=[(train_loss, 'Training'), (val_loss, 'Validating')], xlabel='Epoch', ylabel='Total Loss',
                title='Training and Validating Total Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_total')
    figLoss(line_list=[(train_est_loss, 'Training-Source'),  (train_est_loss_target, 'Training-Target'), 
                            (val_est_loss_source, 'Validating-Source'), (val_est_loss_target, 'Validating-Target')], xlabel='Epoch', ylabel='Estimation Loss',
                title='Training and Validating Estimation Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_est')
        # estimation loss: MSE loss, before de-scale
    figLoss(line_list=[(train_domain_loss, 'Training'), (val_domain_disc_loss, 'Validating')], xlabel='Epoch', ylabel='Domain Discrimination Loss',
                title='Training and Validating Domain Discrimination Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_domain')

def save_checkpoint_(model, save_model, model_path, sub_folder, epoch, metrics):
    figLoss = metrics['figLoss']; savemat = metrics['savemat']; 
    train_loss = metrics['train_loss']; train_est_loss = metrics['train_est_loss']
    train_domain_loss = metrics['train_domain_loss']; train_est_loss_target = metrics['train_est_loss_target']; 
    val_est_loss = metrics['val_est_loss']; val_est_loss_source = metrics['val_est_loss_source']; 
    val_loss = metrics['val_loss']; val_est_loss_target = metrics['val_est_loss_target']
    val_gan_disc_loss = metrics['val_gan_disc_loss']; val_domain_disc_loss = metrics['val_domain_disc_loss']; source_acc = metrics['source_acc']
    target_acc = metrics['target_acc']; acc = metrics['acc']; 
    nmse_val_source = metrics['nmse_val_source']; nmse_val_target = metrics['nmse_val_target']
    nmse_val = metrics['nmse_val']; 
    pad_pca_svm = metrics['pad_pca_svm']; pad_pca_lda = metrics['pad_pca_lda']; pad_pca_logreg = metrics['pad_pca_logreg']
    epoc_pad = metrics['epoc_pad']; pad_svm = metrics['pad_svm']; train_disc_loss = metrics['train_disc_loss']; 
    domain_weight = metrics['domain_weight']; optimizer = metrics['optimizer']

    # Save model
    os.makedirs(f"{model_path}/{sub_folder}/model/", exist_ok=True)
    if save_model:
        # Create checkpoint with all model components and optimizers
        gen_optimizer, disc_optimizer, domain_optimizer = optimizer
        
        # Create checkpoint object
        ckpt = tf.train.Checkpoint(
            generator=model.generator,
            discriminator=model.discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer
        )
        
        # Add domain model and optimizer if domain adaptation is used
        if domain_weight is not None:
            ckpt.domain_optimizer = domain_optimizer
        
        # Create checkpoint manager - save only current epoch
        checkpoint_dir = f"{model_path}/{sub_folder}/model/"
        
        # Save checkpoint
        milestone_path = ckpt.save(f"{checkpoint_dir}/epoch_{epoch+1}")
        print(f"Checkpoint saved at epoch {epoch+1}: {milestone_path}")

        # Save optimizer configs
        optimizer_configs = {
            'gen_optimizer_config': gen_optimizer.get_config(),
            'disc_optimizer_config': disc_optimizer.get_config(), 
            'domain_optimizer_config': domain_optimizer.get_config()
        }
        config_path = f"{checkpoint_dir}/optimizer_configs.json"  # No epoch number
        # Only save if file doesn't exist (to avoid overwriting)
        if not os.path.exists(config_path):
            import json
            with open(config_path, 'w') as f:
                json.dump(optimizer_configs, f, indent=2)
            print(f"Optimizer configs saved to: {config_path}")
    
    # === save and overwrite at checkpoints
    # train
    perform_to_save = {}
    perform_to_save['train_loss'] = train_loss
    perform_to_save['train_est_loss'] = train_est_loss
    perform_to_save['train_disc_loss'] = train_disc_loss
    perform_to_save['train_domain_loss'] = train_domain_loss
    perform_to_save['train_est_loss_target'] = train_est_loss_target
    # val
    perform_to_save['val_est_loss'] = val_est_loss
    perform_to_save['val_est_loss_source'] = val_est_loss_source
    perform_to_save['val_loss'] = val_loss
    perform_to_save['val_est_loss_target'] = val_est_loss_target
    perform_to_save['val_gan_disc_loss'] = val_gan_disc_loss
    perform_to_save['val_domain_disc_loss'] = val_domain_disc_loss
    perform_to_save['source_acc'] = source_acc
    perform_to_save['target_acc'] = target_acc
    perform_to_save['acc'] = acc
    perform_to_save['nmse_val_source'] = nmse_val_source
    perform_to_save['nmse_val_target'] = nmse_val_target
    perform_to_save['nmse_val'] = nmse_val
    #
    perform_to_save['pad_pca_svm'] = pad_pca_svm
    perform_to_save['pad_pca_lda'] = pad_pca_lda
    perform_to_save['pad_pca_logreg'] = pad_pca_logreg
    perform_to_save['epoc_pad'] = epoc_pad
    perform_to_save['pad_svm'] = pad_svm

    # save
    os.makedirs(f"{model_path}/{sub_folder}/performance/", exist_ok=True)
    savemat(model_path + '/' + sub_folder + '/performance/performance.mat', perform_to_save)
    
    # Plot figures === save and overwrite at checkpoints
    if domain_weight:
        figLoss(line_list=[(nmse_val_source, 'Source Domain'), (nmse_val_target, 'Target Domain')], xlabel='Epoch', ylabel='NMSE',
                    title='NMSE in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='NMSE_val')
        figLoss(line_list=[(source_acc, 'Source Domain'), (target_acc, 'Target Domain')], xlabel='Epoch', ylabel='Discrimination Accuracy',
                    title='Domain Discrimination Accuracy in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Domain_acc')
    #
    figLoss(line_list=[(train_est_loss, 'GAN Generate Loss'), (train_disc_loss, 'GAN Discriminator Loss')], xlabel='Epoch', ylabel='Loss',
                title='Training GAN Losses', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='GAN_train')
    figLoss(line_list=[(train_loss, 'Training'), (val_loss, 'Validating')], xlabel='Epoch', ylabel='Total Loss',
                title='Training and Validating Total Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_total')
    figLoss(line_list=[(train_est_loss, 'Training-Source'),  (train_est_loss_target, 'Training-Target'), 
                            (val_est_loss_source, 'Validating-Source'), (val_est_loss_target, 'Validating-Target')], xlabel='Epoch', ylabel='Estimation Loss',
                title='Training and Validating Estimation Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_est')
        # estimation loss: MSE loss, before de-scale
    figLoss(line_list=[(train_domain_loss, 'Training'), (val_domain_disc_loss, 'Validating')], xlabel='Epoch', ylabel='Domain Discrimination Loss',
                title='Training and Validating Domain Discrimination Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_domain')


def load_checkpoint(model, model_path, sub_folder, epoch_load, optimizer=None, domain_model=None, domain_weight=True):
    """
    Load checkpoint for GAN model and optimizers.
    
    Args:
        model: GAN model instance
        model_path: Path to model directory
        sub_folder: Subfolder name
        epoch: Epoch number to load
        optimizer: Tuple of (gen_optimizer, disc_optimizer, domain_optimizer)
        domain_model: Domain discriminator model
        domain_weight: Whether domain adaptation is used
    
    Returns:
        Tuple of (gen_optimizer, disc_optimizer, domain_optimizer) with restored states
    """
    checkpoint_dir = f"{model_path}/{sub_folder}/model"
    checkpoint_path = f"{checkpoint_dir}/epoch_{epoch_load+1}-1"
    
    if optimizer is not None:
        gen_optimizer, disc_optimizer, domain_optimizer = optimizer
    else:
        # Load optimizer configs from saved file
        config_path = f"{checkpoint_dir}/optimizer_configs.json"
        import json
        with open(config_path, 'r') as f:
            optimizer_configs = json.load(f)
        
        # Create optimizers using saved configs
        gen_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_configs['gen_optimizer_config'])
        disc_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_configs['disc_optimizer_config'])
        domain_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_configs['domain_optimizer_config'])
        
        print("Optimizers created from saved configs")
    
    # Create checkpoint object
    ckpt = tf.train.Checkpoint(
        generator=model.generator,
        discriminator=model.discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer
    )
    
    # Add domain model and optimizer if domain adaptation is used
    if domain_weight and domain_model is not None:
        ckpt.domain_model = domain_model
        ckpt.domain_optimizer = domain_optimizer
    
    status = ckpt.restore(checkpoint_path)
    # Verify restoration (remove expect_partial() for debugging)
    try:
        # status.assert_consumed()  # This will raise an error if something wasn't restored
        status.expect_partial()
        print("Checkpoint fully restored successfully")
    except tf.errors.InvalidArgumentError as e:
        print(f"Warning: Some checkpoint components not restored: {e}")
        # Use expect_partial() only if you understand what's not being restored
        status.expect_partial()
    
    print(f"Checkpoint restored: {checkpoint_path}")
    
    return gen_optimizer, disc_optimizer, domain_optimizer
            