""" helper functions and classes for GAN model 
"""
import tensorflow as tf
import numpy as np
import h5py
import os
import sys 
from dataclasses import dataclass

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
            
    def call(self, x, training=False, return_features=False):
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
        if return_features:
            # Return multiple feature layers for JMMD
            features = [
                tf.reshape(d2, [tf.shape(d2)[0], -1]),  # Feature layer 1
                tf.reshape(d3, [tf.shape(d3)[0], -1]),  # Feature layer 2
                tf.reshape(d4, [tf.shape(d4)[0], -1])   # Bottleneck layer
            ]
            return u4, features
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

# def gradient_penalty(discriminator, real, fake, batch_size):
#     alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
#     diff = fake - real
#     interpolated = real + alpha * diff
#     with tf.GradientTape() as gp_tape:
#         gp_tape.watch(interpolated)
#         pred = discriminator(interpolated, training=True)
#     grads = gp_tape.gradient(pred, [interpolated])[0]
#     norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
#     gp = tf.reduce_mean((norm - 1.0) ** 2)
#     return gp
def train_step_wgan_gp_jmmd(model, loader_H, loss_fn, optimizers, lower_range=-1, save_features = False,
                            nsymb=14, adv_weight=0.01, est_weight=1.0, jmmd_weight=0.5, linear_interp=False):
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
        N_train += x_src.shape[0] + x_tgt.shape[0]

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
            x_fake_src, _ = model.generator(x_scaled_src, training=True, return_features=False)
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
            x_fake_src, features_src = model.generator(x_scaled_src, training=True, return_features=True)
            d_fake_src = model.discriminator(x_fake_src, training=False)
            
            # Generate from target domain with features
            x_fake_tgt, features_tgt = model.generator(x_scaled_tgt, training=True, return_features=True)
            
            # Generator losses
            g_adv_loss = -tf.reduce_mean(d_fake_src)  # WGAN-GP adversarial loss
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)  # Estimation loss (source)
            g_est_loss_tgt = loss_fn_est(y_scaled_tgt, x_fake_tgt)  # Estimation loss (target, for monitoring)
            
            # JMMD loss between source and target features
            jmmd_loss = jmmd_loss_fn(features_src, features_tgt)
            
            # Total generator loss
            g_loss = (est_weight * g_est_loss + 
                     adv_weight * g_adv_loss + 
                     jmmd_weight * jmmd_loss)
            
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
    
def val_step_wgan_gp_jmmd(model, loader_H, loss_fn, lower_range, nsymb=14, adv_weight=0.01, est_weight=1.0, jmmd_weight=0.5, linear_interp=False):
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
    
    # For compatibility with existing code, we'll set domain accuracy to 0.5 (random)
    # since JMMD doesn't have classification accuracy
    avg_domain_acc_source = 0.5  # Neutral value for JMMD (no classification)
    avg_domain_acc_target = 0.5  # Neutral value for JMMD (no classification)
    avg_domain_acc = 0.5         # Neutral value for JMMD (no classification)

    # Weighted total loss (for comparison with training)
    avg_total_loss = est_weight * avg_loss_est + adv_weight * avg_gan_disc_loss + jmmd_weight * avg_jmmd_loss

    # Compose epoc_eval_return - Replace domain discriminator loss with JMMD loss
    epoc_eval_return = [
        avg_total_loss,
        avg_loss_est_source, avg_loss_est_target, avg_loss_est,
        avg_gan_disc_loss, avg_jmmd_loss,  # Replace domain_disc_loss with jmmd_loss
        avg_nmse_source, avg_nmse_target, avg_nmse,
        avg_domain_acc_source, avg_domain_acc_target, avg_domain_acc  # Keep for compatibility
    ]

    return H_sample, epoc_eval_return

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
