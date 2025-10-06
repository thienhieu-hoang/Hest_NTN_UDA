""" helper functions for GAN FiLM model 
    input shape: [batch_size, 2, 792, 14] = [batch_size, 2, nsubcs, nsymbs] - CSI-RS-estimated channel at slot 6
        conditional input shape: [batch_size, 2, 792, 1] = [batch_size, 2, nsubcs, 1] - CSI-RS-estimated channel at symbol 2 slot 1
    Output shape: [batch_size, 2, 792, 14]
"""
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import h5py
import sys
import os 

from dataclasses import dataclass

@dataclass
class train_step_Output:
    """Dataclass to hold the output of the train_step function."""
    avg_epoc_loss_est: float   
    avg_epoc_loss_domain: float
    avg_epoc_loss: float
    avg_epoc_loss_est_target: float  # Average loss for channel estimation on target domain
    features_source: tf.Tensor = None  # Features from the source domain, if return_features is True
    film_features_source: tf.Tensor = None  # Film features from the source domain, if return_features is True
    features_target: tf.Tensor = None  # Features from the target domain, if return_features is True
    film_features_target: tf.Tensor = None  # Film features from the target domain, if return_features is True
    pad: float = 0

try:
    # Case 1: Running as a .py script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..'))
except NameError:
    # Case 2: Running inside Jupyter Notebook
    notebook_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(notebook_dir, '..', '..'))
sys.path.append(project_root)

import Est_btween_CSIRS.helper.utils_CNN_FiLM as utils_CNN_FiLM
import Est_btween_CSIRS.helper.utils as utils_CNN

import tensorflow as tf

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


class FiLM4(tf.keras.layers.Layer):
    """
    Applies flatten, then a Dense layer to get gamma and beta of shape [batch, 1, 1, C_out].
    Input: x, cond of shape [batch, H, W, C_in]
    Output: out (modulated x), optionally gamma and beta
            gamma, beta of shape [batch, 1, 1, C_out]
    """
    def __init__(self, C_out, dropout_rate=0.3, return_params=False):
        super(FiLM4, self).__init__()
        self.C_out = C_out
        self.return_params = return_params
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape = (x, cond)
        # We'll only build layers now that we know cond shape
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.dense_gamma = tf.keras.layers.Dense(self.C_out)
        self.dense_beta = tf.keras.layers.Dense(self.C_out)
        super(FiLM4, self).build(input_shape)

    def call(self, x, cond=None, training=False):
        if cond is not None:
            cond_flat = self.flatten(cond)
            if self.dropout:
                cond_flat = self.dropout(cond_flat, training=training)
            gamma = self.dense_gamma(cond_flat)  # (batch, C_out)
            beta = self.dense_beta(cond_flat)    # (batch, C_out)
            gamma = tf.reshape(gamma, [-1, 1, 1, self.C_out])
            beta = tf.reshape(beta, [-1, 1, 1, self.C_out])
            out = x * gamma + beta
            if self.return_params:
                return out, gamma, beta
            return out
        else:
            return x  # No modulation if no condition provided
    
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

class DomainDiscriminator(tf.keras.Model):
    def __init__(self, in_channels=128, disc='dense', first_layer='pool', mode='main_feature'):
        # disc: 'dense' or 'conv'
        # first_layer: 'pool' or 'flatten'
        # mode: 'main_feature' or 'condition' (for condition domain discriminator)
        super().__init__()
        self.disc = disc
        self.first_layer = first_layer
        self.mode = mode
        #
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc0 = tf.keras.layers.Dense(256, activation='relu') 
        #
        self.fc1 = tf.keras.layers.Dense(in_channels, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        if disc == 'dense':
            self.out = tf.keras.layers.Dense(1, activation='sigmoid')  # Binary: source or target

        self.conv1 = tf.keras.layers.Conv2D(in_channels, (5, 3), strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(8, (3, 3), strides=2, padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        if disc == 'conv':
            self.out = tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification


    def call(self, x, training=False):
        if self.disc == 'dense':
            if self.first_layer == 'pool':
                if self.mode == 'main_feature':
                    x = self.pool(x) # skip this if 'condition domain discriminator'
            elif self.first_layer == 'flatten':
                x = self.flatten(x)
                x = self.fc0(x)
            x = self.fc1(x)
            x = self.fc2(x)
        elif self.disc == 'conv':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
        return self.out(x)    

class DomainDiscriminator2(tf.keras.Model):
    def __init__(self, cnn_filters=[64, 32], fc_units=[128, 64], mode='main_feature'):
        """
        Hybrid CNN + FC domain discriminator for large feature maps.
        cnn_filters: list of filters for Conv2D layers
        fc_units: list of units for Dense layers
        """
        super().__init__()
        self.mode = mode
        self.cnn_layers = []
        for f in cnn_filters:
            self.cnn_layers.append(tf.keras.layers.Conv2D(f, (3, 3), strides=2, padding='same', activation='relu'))
        self.cnn_layers = tf.keras.Sequential(self.cnn_layers)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layers = []
        for u in fc_units:
            self.fc_layers.append(tf.keras.layers.Dense(u, activation='relu'))
        self.fc_layers = tf.keras.Sequential(self.fc_layers)
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        x = self.cnn_layers(x)
        x = self.pool(x)
        x = self.fc_layers(x)
        return self.out(x)

class DomainDiscriminator3(tf.keras.Model):
    """ conv 3 times, then global average pooling, then dense layers 
        extracted features with shape [batch, 48, 6, C_out] (C_out= 512 or 128)
    """
    def __init__(self, l2_reg=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg is not None else None
        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=(4,3), strides=(2,1), padding='valid', 
                                            activation='relu', kernel_regularizer=kernel_regularizer)
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3,2), strides=(2,1), padding='valid', 
                                            activation='relu', kernel_regularizer=kernel_regularizer)
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3,2), strides=(2,1), padding='valid', 
                                            activation='relu', kernel_regularizer=kernel_regularizer)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, x):
        x = self.conv1(x)  # (batch, 23, 4, 256)
        x = self.conv2(x)  # (batch, 11, 3, 128)
        x = self.conv3(x)  # (batch, 5, 2, 64)
        x = self.pool(x)   # (batch, 64)
        x = self.fc1(x)    # (batch, 64)
        return self.out(x) # (batch, 1) - domain probability


def prepare_data(loader_H_1_6, nsymb=14):
    return utils_CNN_FiLM.prepare_data(loader_H_1_6, nsymb)
    
# Domain labels
def make_domain_labels(batch_size, domain):
    return tf.ones((batch_size, 1)) if domain == 'source' else tf.zeros((batch_size, 1))
    

class GAN_FiLM_Output:
    """Dataclass to hold the output of the GAN_FiLM model."""
    def __init__(self, gen_out, disc_out, extracted_features):
        self.gen_out = gen_out  # Generator output
        self.disc_out = disc_out  # Discriminator output
        self.extracted_features = extracted_features  # Extracted features
        
def reflect_padding_2d(x, pad_h, pad_w):
    return tf.pad(x, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='SYMMETRIC')


# ================= GAN_FiLM Model =====================
class UNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, apply_dropout=False, FiLM = FiLM4, kernel_size=(4,3), strides=(2,1), pad_h=0, pad_w=1, gen_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides=strides, padding='valid',
                                            kernel_regularizer=kernel_regularizer)
        self.norm = InstanceNormalization()
        self.film = FiLM(filters)
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None

    def call(self, x, cond, training):
        if self.pad_h > 0 or self.pad_w > 0:
            x = reflect_padding_2d(x, pad_h=0, pad_w=1)  # symmetric padding
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.film(x, cond)
        x = tf.nn.leaky_relu(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        return x

class UNetUpBlock(tf.keras.layers.Layer):
    def __init__(self, filters, apply_dropout=False, FiLM=FiLM4, kernel_size=(4,3), strides=(2,1), gen_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        self.deconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,
                                                        padding='valid', kernel_regularizer=kernel_regularizer)
        self.norm = InstanceNormalization()
        self.film = FiLM(filters)
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None

    def call(self, x, skip, cond, training):
        x = self.deconv(x)
        if x.shape[2] >14: 
            x = x[:, :, 1:15, :]
        x = self.norm(x, training=training)
        x = self.film(x, cond)
        x = tf.nn.relu(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        x = tf.concat([x, skip], axis=-1)
        return x

class Pix2PixFiLMGenerator(tf.keras.Model):
    def __init__(self, output_channels=2, n_subc=792, gen_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(gen_l2) if gen_l2 is not None else None
        if n_subc == 792 or n_subc==312:
            # Encoder
            self.down1 = UNetBlock(64, apply_dropout=False, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
            self.down2 = UNetBlock(128, kernel_size=(5,3), strides=(2,1), gen_l2=gen_l2)
            self.down3 = UNetBlock(256, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
            self.down4 = UNetBlock(512, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
            # Decoder
            self.up1 = UNetUpBlock(256, apply_dropout=True, kernel_size=(3,3), strides=(2,1), gen_l2=gen_l2)
            self.up2 = UNetUpBlock(128, apply_dropout=True, kernel_size=(4,3), strides=(2,1), gen_l2=gen_l2)
            self.up3 = UNetUpBlock(64, kernel_size=(5,3), strides=(2,1), gen_l2=gen_l2)
            self.last = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=(4,3), strides=(2,1), padding='valid',
                                                        activation='tanh', kernel_regularizer=kernel_regularizer)
            
    def call(self, x, cond, training=False):
        # Encoder
        d1 = self.down1(x, cond, training=training)      # (batch, 395, 12, C_out)
        d2 = self.down2(d1, cond, training=training)     # (batch, 196, 10, C_out)
        d3 = self.down3(d2, cond, training=training)     # (batch,  97, 8, C_out)
        d4 = self.down4(d3, cond, training=training)     # (batch,  48, 6, C_out)
        # Decoder with skip connections
        u1 = self.up1(d4, d3, cond, training=training)   # (batch,  97, 8, C_out)
        u2 = self.up2(u1, d2, cond, training=training)   # (batch, 196, 10, C_out)
        u3 = self.up3(u2, d1, cond, training=training)   # (batch, 395, 12, C_out)
        u4 = self.last(u3)  # (batch, 792, 14 or 16, C_out)
        if u4.shape[2] > 14:
            u4 = u4[:, :, 1:15, :]
        return u4, d4
                # estimator return (batch, 792, 14, 2)
                # feature extractor return (batch, 48, 6, C_out) (d4)
###
class PatchGANDiscriminator(tf.keras.Model):
    """
    PatchGAN Discriminator for Pix2Pix GAN.
    Input: (batch, H, W, C)
    Output: (batch, H_out, W_out, 1) patch-level real/fake probabilities
    """
    def __init__(self, filters=[64, 128, 256, 512], n_subc=792, disc_l2=None):
        super().__init__()
        kernel_regularizer = tf.keras.regularizers.l2(disc_l2) if disc_l2 is not None else None
        if n_subc==792 or n_subc==312:
            self.conv1 = tf.keras.layers.Conv2D(filters[0], kernel_size=(4,3), strides=(2,1), padding='valid',
                                                kernel_regularizer=kernel_regularizer)
            self.conv2 = tf.keras.layers.Conv2D(filters[1], kernel_size=(5,3), strides=(2,1), padding='valid',
                                                kernel_regularizer=kernel_regularizer)
            self.norm2 = InstanceNormalization()
            self.conv3 = tf.keras.layers.Conv2D(filters[2], kernel_size=(4,3), strides=(2,1), padding='valid',
                                                kernel_regularizer=kernel_regularizer)
            self.norm3 = InstanceNormalization()
            self.conv4 = tf.keras.layers.Conv2D(filters[3], kernel_size=(3,3), strides=(2,1), padding='valid',
                                                kernel_regularizer=kernel_regularizer)
            self.norm4 = InstanceNormalization()
            self.last = tf.keras.layers.Conv2D(1, kernel_size=(4,3), strides=(2,1), padding='valid',
                                                kernel_regularizer=kernel_regularizer)  # Output: patch map

    def call(self, x, training=False):
        x = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)  # (batch, 395, 12, C_out)
        x = tf.nn.leaky_relu(self.norm2(self.conv2(x), training=training), alpha=0.2)  # (batch, 196, 10, C_out)
        x = tf.nn.leaky_relu(self.norm3(self.conv3(x), training=training), alpha=0.2)  # (batch, 97, 8, C_out)
        x = tf.nn.leaky_relu(self.norm4(self.conv4(x), training=training), alpha=0.2)  # (batch, 48, 6, C_out)
        return self.last(x)  # (batch, 23, 3, 1) - patch-level real/fake probabilities

        
class GAN_FiLM(tf.keras.Model):
    def __init__(self, n_subc=792, generator=Pix2PixFiLMGenerator, discriminator=PatchGANDiscriminator, gen_l2=None, disc_l2=None):
        super().__init__()
        self.generator = generator(n_subc=n_subc, gen_l2=gen_l2)
        self.discriminator = discriminator(n_subc=n_subc, disc_l2=disc_l2)

    def call(self, inputs, training=False):
        # Optionally implement a forward pass if needed
        if isinstance(inputs, tuple):
            x, cond = inputs
        else:
            x, cond = inputs, None
        gen_out, features = self.generator(x, cond, training=training)
        disc_out = self.discriminator(gen_out, training=training)
        return GAN_FiLM_Output(
            gen_out=gen_out,
            disc_out=disc_out,
            extracted_features=features
        )


# ================= GAN Training Step =====================
def train_step_gan(model, domain_model, loader_H, loss_fn, optimizers, lower_range=-1, return_features=False,
        nsymb=14, adv_weight=0.01, est_weight=1.0, domain_weight=0.5):
    """
    model: GAN_FiLM model instance
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
    
    if return_features==True:
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
        x_cond_src, x_src = prepare_data(x_src, nsymb=nsymb)
        y_src = loader_H_true_train_src.next_batch()
        y_src = y_src[:, :, 70:84]
        # --- Target domain ---
        x_tgt = loader_H_input_train_tgt.next_batch()
        x_cond_tgt, x_tgt = prepare_data(x_tgt, nsymb=nsymb)
        y_tgt = loader_H_true_train_tgt.next_batch()
        y_tgt = y_tgt[:, :, 70:84]
        N_train += x_src.shape[0] + x_tgt.shape[0]

        # Preprocess (source)
        x_src = utils_CNN.complx2real(x_src)
        y_src = utils_CNN.complx2real(y_src)
        x_cond_src = utils_CNN.complx2real(x_cond_src)
        x_src = np.transpose(x_src, (0, 2, 3, 1))
        x_cond_src = np.transpose(x_cond_src, (0, 2, 3, 1))
        y_src = np.transpose(y_src, (0, 2, 3, 1))
        x_all_src = np.concatenate([x_cond_src, x_src], axis=2)
        x_all_scaled_src, _, _ = utils_CNN.minmaxScaler(x_all_src, lower_range=lower_range)
        x_cond_src = x_all_scaled_src[:, :, :1, :]
        x_scaled_src = x_all_scaled_src[:, :, 1:, :]
        y_scaled_src, _, _ = utils_CNN.minmaxScaler(y_src, lower_range=lower_range)

        # Preprocess (target)
        x_tgt = utils_CNN.complx2real(x_tgt)
        y_tgt = utils_CNN.complx2real(y_tgt)
        x_cond_tgt = utils_CNN.complx2real(x_cond_tgt)
        x_tgt = np.transpose(x_tgt, (0, 2, 3, 1))
        x_cond_tgt = np.transpose(x_cond_tgt, (0, 2, 3, 1))
        y_tgt = np.transpose(y_tgt, (0, 2, 3, 1))
        x_all_tgt = np.concatenate([x_cond_tgt, x_tgt], axis=2)
        x_all_scaled_tgt, _, _ = utils_CNN.minmaxScaler(x_all_tgt, lower_range=lower_range)
        x_cond_tgt = x_all_scaled_tgt[:, :, :1, :]
        x_scaled_tgt = x_all_scaled_tgt[:, :, 1:, :]
        y_scaled_tgt, _, _ = utils_CNN.minmaxScaler(y_tgt, lower_range=lower_range)

        # === 1. Train Discriminator ===
        if domain_weight!= 0:
            with tf.GradientTape() as tape_d:
                x_fake_src = model.generator(x_scaled_src, x_cond_src, training=True)[0] 
                    # x_fake == generated data gen(x_real)
                    # x_fake ~= y (y is real)
                d_real = model.discriminator(y_scaled_src, training=True)
                d_fake = model.discriminator(x_fake_src, training=True)
                real_labels = tf.ones_like(d_real)
                fake_labels = tf.zeros_like(d_fake)
                d_loss_real = loss_fn_bce(real_labels, d_real)
                d_loss_fake = loss_fn_bce(fake_labels, d_fake)
                d_loss = d_loss_real + d_loss_fake
            grads_d = tape_d.gradient(d_loss, model.discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(grads_d, model.discriminator.trainable_variables))
            epoc_loss_d += d_loss.numpy() * x_src.shape[0]

        # === 2. Train Generator (with Gradient Reversal for domain loss) ===
        with tf.GradientTape() as tape_g:
            x_fake_src, features_src = model.generator(x_scaled_src, x_cond_src, training=True)
            d_fake = model.discriminator(x_fake_src, training=True)
            g_adv_loss = loss_fn_bce(tf.ones_like(d_fake), d_fake)
            g_est_loss = loss_fn_est(y_scaled_src, x_fake_src)
            # --- Gradient Reversal for domain loss ---
            _, features_tgt = model.generator(x_scaled_tgt, x_cond_tgt, training=True)
            features_grl = tf.concat([GradientReversal()(features_src), GradientReversal()(features_tgt)], axis=0)
            domain_labels_src = tf.ones((features_src.shape[0], 1))
            domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
            domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
            domain_pred = domain_model(features_grl, training=True)
            domain_loss_grl = loss_fn_domain(domain_labels, domain_pred)
            # --- Total generator loss ---
            g_loss = adv_weight * g_adv_loss + est_weight * g_est_loss + domain_weight * domain_loss_grl
        grads_g = tape_g.gradient(g_loss, model.generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads_g, model.generator.trainable_variables))
        epoc_loss_g += g_loss.numpy() * x_src.shape[0]
        epoc_loss_est += g_est_loss.numpy() * x_src.shape[0]

        # === 3. Train Domain Discriminator ===
        with tf.GradientTape() as tape_domain:
            _, features_src = model.generator(x_scaled_src, x_cond_src, training=False)
            _, features_tgt = model.generator(x_scaled_tgt, x_cond_tgt, training=False)
            domain_labels_src = tf.ones((features_src.shape[0], 1))
            domain_labels_tgt = tf.zeros((features_tgt.shape[0], 1))
            features = tf.concat([features_src, features_tgt], axis=0)
            domain_labels = tf.concat([domain_labels_src, domain_labels_tgt], axis=0)
            domain_pred = domain_model(features, training=True)
            domain_loss = loss_fn_domain(domain_labels, domain_pred)
        grads_domain = tape_domain.gradient(domain_loss, domain_model.trainable_variables)
        domain_optimizer.apply_gradients(zip(grads_domain, domain_model.trainable_variables))
        epoc_loss_domain += domain_loss.numpy() * features.shape[0]
    
        # === 4. Save features if required ===
        if return_features:
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
    if return_features:    
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
    )
    
def val_step(model, domain_model, loader_H, loss_fn, lower_range, nsymb=14, adv_weight=0.01, est_weight=1.0, domain_weight=0.5):
    """
    Validation step for GAN_FiLM model. Returns H_sample and epoc_eval_return (summary metrics).
    Args:
        model: GAN_FiLM model instance
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
        x_cond_src, x_src = prepare_data(x_src, nsymb=nsymb)
        y_src = loader_H_true_val_source.next_batch()
        y_src = y_src[:, :, 70:84]
        # --- Target domain ---
        x_tgt = loader_H_input_val_target.next_batch()
        x_cond_tgt, x_tgt = prepare_data(x_tgt, nsymb=nsymb)
        y_tgt = loader_H_true_val_target.next_batch()
        y_tgt = y_tgt[:, :, 70:84]
        N_val_source += x_src.shape[0]
        N_val_target += x_tgt.shape[0]

        # Preprocess (source)
        x_src_real = utils_CNN.complx2real(x_src)
        y_src_real = utils_CNN.complx2real(y_src)
        x_cond_src_real = utils_CNN.complx2real(x_cond_src)
        x_src_real = np.transpose(x_src_real, (0, 2, 3, 1))
        x_cond_src_real = np.transpose(x_cond_src_real, (0, 2, 3, 1))
        y_src_real = np.transpose(y_src_real, (0, 2, 3, 1))
        x_all_src = np.concatenate([x_cond_src_real, x_src_real], axis=2)
        x_all_scaled_src, x_min_src, x_max_src = utils_CNN.minmaxScaler(x_all_src, lower_range=lower_range)
        x_cond_src_scaled = x_all_scaled_src[:, :, :1, :]
        x_scaled_src = x_all_scaled_src[:, :, 1:, :]
        y_scaled_src, _, _ = utils_CNN.minmaxScaler(y_src_real, lower_range=lower_range)

        # Preprocess (target)
        x_tgt_real = utils_CNN.complx2real(x_tgt)
        y_tgt_real = utils_CNN.complx2real(y_tgt)
        x_cond_tgt_real = utils_CNN.complx2real(x_cond_tgt)
        x_tgt_real = np.transpose(x_tgt_real, (0, 2, 3, 1))
        x_cond_tgt_real = np.transpose(x_cond_tgt_real, (0, 2, 3, 1))
        y_tgt_real = np.transpose(y_tgt_real, (0, 2, 3, 1))
        x_all_tgt = np.concatenate([x_cond_tgt_real, x_tgt_real], axis=2)
        x_all_scaled_tgt, x_min_tgt, x_max_tgt = utils_CNN.minmaxScaler(x_all_tgt, lower_range=lower_range)
        x_cond_tgt_scaled = x_all_scaled_tgt[:, :, :1, :]
        x_scaled_tgt = x_all_scaled_tgt[:, :, 1:, :]
        y_scaled_tgt, _, _ = utils_CNN.minmaxScaler(y_tgt_real, lower_range=lower_range)

        # === Source domain prediction ===
        preds_src, features_src = model.generator(x_scaled_src, x_cond_src_scaled, training=False)
        preds_src = preds_src.numpy() if hasattr(preds_src, 'numpy') else preds_src
        preds_src_descaled = utils_CNN.deMinMax(preds_src, x_min_src, x_max_src, lower_range=lower_range)
        batch_est_loss_source = loss_fn_est(y_scaled_src, preds_src).numpy()
        epoc_loss_est_source += batch_est_loss_source * x_src.shape[0]
        mse_val_source = np.mean((preds_src_descaled - y_src_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_src_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x_src.shape[0]

        # === Target domain prediction ===
        preds_tgt, features_tgt = model.generator(x_scaled_tgt, x_cond_tgt_scaled, training=False)
        preds_tgt = preds_tgt.numpy() if hasattr(preds_tgt, 'numpy') else preds_tgt
        preds_tgt_descaled = utils_CNN.deMinMax(preds_tgt, x_min_tgt, x_max_tgt, lower_range=lower_range)
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
            domain_loss = loss_fn_domain(domain_labels, domain_pred).numpy() * features.shape[0]
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
            H_input_condition = x_cond_src_real[:n_samples].copy()
            H_est_sample = preds_src_descaled[:n_samples].numpy().copy()
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)
            # Target
            H_true_sample_target = y_tgt_real[:n_samples].copy()
            H_input_sample_target = x_tgt_real[:n_samples].copy()
            H_input_condition_target = x_cond_tgt_real[:n_samples].copy()
            H_est_sample_target = preds_tgt_descaled[:n_samples].numpy().copy()
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)
            H_sample = [H_true_sample, H_input_sample, H_est_sample, H_input_condition, nmse_input_source, nmse_est_source,
                        H_true_sample_target, H_input_sample_target, H_est_sample_target, H_input_condition_target, nmse_input_target, nmse_est_target]
            


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


def visualize_H(H_sample, H_to_save, epoch, figChan, flag, model_path, sub_folder):
    (
        H_true_sample,  H_input_sample, H_est_sample, H_input_condition, 
        nmse_input_source, nmse_est_source,
        H_true_sample_target, H_input_sample_target, H_est_sample_target, H_input_condition_target,
        nmse_input_target, nmse_est_target
    ) = H_sample

    # plot real parts
    # source domain
    if flag == 1:
        figChan(H_true_sample[0,:,:,0], title='True channel Slot 6', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_true_source')
        figChan(H_input_sample[0,:,:,0], nmse=nmse_input_source[0], title='Raw-estimated Channel Slot 6', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_input_source')
        # if sub_folder == 'in_15_symbs' or sub_folder == 'FiLM':
        figChan(H_input_condition[0,:,0,0], title='Condition Channel', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_input_condition_source')
    figChan(H_est_sample[0,:,:,0], nmse=nmse_est_source[0], title='GAN-refined Channel Slot 6', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_GAN_source')
    # target domain
    if flag == 1:
        figChan(H_true_sample_target[0,:,:,0], title='True channel Slot 6', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_true_target')
        figChan(H_input_sample_target[0,:,:,0], nmse=nmse_input_target[0], title='Raw-estimated Channel Slot 6', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_input_target')
        # if sub_folder == 'in_15_symbs' or sub_folder == 'FiLM':
        figChan(H_input_condition_target[0,:,0,0], title='Condition Channel', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_input_condition_target')
        flag = 0
    figChan(H_est_sample[0,:,:,0], nmse=nmse_est_target[0], title='GAN-refined Channel Slot 6', index_save=epoch+1, 
                                figure_save_path=model_path + '/' + sub_folder + '/H_visualize', name='H_GAN_target')

    H_to_save[f'{epoch+1}_H_true_source'] = H_true_sample
    H_to_save[f'{epoch+1}_H_input_source'] = H_input_sample
    H_to_save[f'{epoch+1}_H_est_source'] = H_est_sample
    H_to_save[f'{epoch+1}_H_input_condition_source'] = H_input_condition
        # don't save nmse_input_source, nmse_est_source because can compute them later
    #
    H_to_save[f'{epoch+1}_H_true_target'] = H_true_sample_target
    H_to_save[f'{epoch+1}_H_input_target'] = H_input_sample_target
    H_to_save[f'{epoch+1}_H_est_target'] = H_est_sample_target
    H_to_save[f'{epoch+1}_H_input_condition_target'] = H_input_condition_target

def post_val(epoc_val_return, epoch, n_epochs, val_est_loss, val_est_loss_source, val_loss, val_est_loss_target,
            val_gan_disc_loss, val_domain_disc_loss, nmse_val_source, nmse_val_target, nmse_val, source_acc, target_acc, acc):
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
    val_est_loss_target.append(avg_loss_est_target)
    print(f"epoch {epoch+1}/{n_epochs} (Val) Average Estimation Loss (Target): {avg_loss_est_target:.6f}")
    #
    val_gan_disc_loss.append(avg_gan_disc_loss)
    print(f"epoch {epoch+1}/{n_epochs} (Val) GAN Discriminator Loss: {avg_gan_disc_loss:.6f}")
    #
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
                    source_acc, target_acc, acc, nmse_val_source, nmse_val_target, nmse_val, pad_observe, epoc_pad):
    # Save model
    os.makedirs(f"{model_path}/{sub_folder}/model/", exist_ok=True)
    if save_model:
        model.save(f"{model_path}/{sub_folder}/model/epoch_{epoch+1}.keras")
        print(f"Model saved at epoch {epoch+1} to {model_path}/{sub_folder}/epoch_{epoch+1}")    
    
    # === save and overwrite at checkpoints
    # train
    perform_to_save = {}
    perform_to_save['train_loss'] = train_loss
    perform_to_save['train_est_loss'] = train_est_loss
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
    perform_to_save['pad_observe'] = pad_observe
    perform_to_save['epoc_pad'] = epoc_pad
    
    # save
    os.makedirs(f"{model_path}/{sub_folder}/performance/", exist_ok=True)
    savemat(model_path + '/' + sub_folder + '/performance/performance.mat', perform_to_save)
    
    # Plot figures === save and overwrite at checkpoints
    figLoss(line_list=[(nmse_val_source, 'Source Domain'), (nmse_val_target, 'Target Domain')], xlabel='Epoch', ylabel='NMSE',
                title='NMSE in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='NMSE_val')
    figLoss(line_list=[(source_acc, 'Source Domain'), (target_acc, 'Target Domain')], xlabel='Epoch', ylabel='Discrimination Accuracy',
                title='Domain Discrimination Accuracy in Validation', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Domain_acc')
    #
    figLoss(line_list=[(train_loss, 'Training'), (val_loss, 'Validating')], xlabel='Epoch', ylabel='Total Loss',
                title='Training and Validating Total Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_total')
    figLoss(line_list=[(train_est_loss, 'Training-Source'), (train_est_loss_target, 'Training-Target'), 
                            (val_est_loss_source, 'Validating-Source'), (val_est_loss_target, 'Validating-Target')], xlabel='Epoch', ylabel='Estimation Loss',
                title='Training and Validating Estimation Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_est')
    figLoss(line_list=[(train_domain_loss, 'Training'), (val_domain_disc_loss, 'Validating')], xlabel='Epoch', ylabel='Domain Discrimination Loss',
                title='Training and Validating Domain Discrimination Loss', index_save=1, figure_save_path= model_path + '/' + sub_folder + '/performance', fig_name='Loss_domain')
            