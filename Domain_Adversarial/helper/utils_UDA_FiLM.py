""" helper functions for CNN FiLM model 
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
class UDA_FiLM_Output:
    """Dataclass to hold the output of the UDA_FiLM model."""
    x: tf.Tensor  # Output tensor from the model
    domain_pred: tf.Tensor  # Domain prediction tensor
    condition_domain_pred: tf.Tensor  # Condition domain prediction tensor
    features: tf.Tensor = None  # Optional features tensor, if return_features is True
    film_features: tf.Tensor = None  # Optional film features tensor, if return_features is True
    
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

notebook_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', '..')))
import Est_btween_CSIRS.helper.utils_CNN_FiLM as utils_CNN_FiLM
import Est_btween_CSIRS.helper.utils as utils_CNN

class FiLM(layers.Layer):
    def __init__(self, n_channels, return_params=False):
        super().__init__()
        self.n_channels = n_channels
        self.return_params = return_params  # If True, return gamma and beta as well

    def build(self, input_shapes):
        cond_shape = input_shapes[1]  # input_shapes =(None, 792, 1)
        input_dim = cond_shape        # 792
        # Explicitly define input shape for Dense
        self.gamma_dense = layers.Dense(self.n_channels)
        self.beta_dense = layers.Dense(self.n_channels)

    def call(self, x, cond):
        gamma = self.gamma_dense(cond)  # (batch, n_channels)
        beta = self.beta_dense(cond)

        
        # gamma = tf.expand_dims(gamma, axis=2) 
        # beta = tf.expand_dims(beta, axis=2) 
        
        if self.return_params:
            return x * gamma + beta, gamma, beta
        
        return x * gamma + beta

class FiLM2(layers.Layer):
    """
    FiLM2: Applies a Dense layer over the H dimension to reduce H to H_out, then a Dense over channels to get C_out.
    Input: x of shape [batch, H, W, C_in]
    Output: gamma, beta of shape [batch, H_out, 1, C_out]
    """
    def __init__(self, C_out, H_out=792,return_params=False):
        super().__init__()
        self.H_out = H_out
        self.C_out = C_out
        self.return_params = return_params
        self.dense_h = None  # Will be built in build()
        self.dense_c = None

    def build(self, input_shapes):
        # input_shapes: (x_shape, cond_shape)
        # cond_shape: [batch, H, 1, C_in]
        C_in = input_shapes[-1]
        H_in = input_shapes[1]
        self.dense_h = tf.keras.layers.Dense(self.H_out)
        self.dense_c = tf.keras.layers.Dense(self.C_out)

    def call(self, x, cond):
        # cond: [batch, H, 1, C_in]
        batch = tf.shape(cond)[0]
        H_in = tf.shape(cond)[1]
        C_in = tf.shape(cond)[-1]
        # Remove singleton width
        cond_reshaped = tf.reshape(cond, [batch, H_in, C_in])  # [batch, H, C_in]
        # Transpose to [batch, C_in, H]
        cond_trans = tf.transpose(cond_reshaped, [0, 2, 1])  # [batch, C_in, H]
        # Dense over H to get H_out
        gamma_h = self.dense_h(cond_trans)  # [batch, C_in, H_out]
        beta_h = self.dense_h(cond_trans)   # [batch, C_in, H_out]
        # Transpose back to [batch, H_out, C_in]
        gamma_h = tf.transpose(gamma_h, [0, 2, 1])  # [batch, H_out, C_in]
        beta_h = tf.transpose(beta_h, [0, 2, 1])    # [batch, H_out, C_in]
        # Dense over channels to get C_out
        gamma = self.dense_c(gamma_h)  # [batch, H_out, C_out]
        beta = self.dense_c(beta_h)    # [batch, H_out, C_out]
        # Add singleton width
        gamma = tf.expand_dims(gamma, axis=2)  # [batch, H_out, 1, C_out]
        beta = tf.expand_dims(beta, axis=2)    # [batch, H_out, 1, C_out]
        # Broadcast if needed
        out = x * gamma + beta
        if self.return_params:
            return out, gamma, beta
        return out

class FiLM3(layers.Layer):
    """
    Applies average pooling over the H dimension, then a Dense layer to get gamma and beta of shape [batch, 1, 1, C_out].
    Input: x, cond of shape [batch, H, W, C_in]
    Output: gamma, beta of shape [batch, 1, 1, C_out]
    """
    def __init__(self, C_out, return_params=False):
        super().__init__()
        self.C_out = C_out
        self.return_params = return_params
        self.dense = tf.keras.layers.Dense(C_out)

    def call(self, x, cond):
        # cond: [batch, H, 1, C_in]
        # Average pool over H
        cond_pooled = tf.reduce_mean(cond, axis=1, keepdims=True)  # [batch, 1, 1, C_in]
        gamma = self.dense(cond_pooled)  # [batch, 1, 1, C_out]
        beta = self.dense(cond_pooled)   # [batch, 1, 1, C_out]
        out = x * gamma + beta  # Broadcasting over H
        if self.return_params:
            return out, gamma, beta
        return out

class FiLM4(tf.keras.layers.Layer):
    """
    Applies flatten, then a Dense layer to get gamma and beta of shape [batch, 1, 1, C_out].
    Input: x, cond of shape [batch, H, W, C_in]
    Output: out (modulated x), optionally gamma and beta
            gamma, beta of shape [batch, 1, 1, C_out]
    """
    def __init__(self, C_out, dropout_rate=0.3, return_params=False):
        super().__init__()
        self.C_out = C_out
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(C_out)
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.return_params = return_params

    def call(self, x, cond):
        cond_flat = self.flatten(cond)  # (batch, 792*2)
        if self.dropout:
            cond_flat = self.dropout(cond_flat)
        gamma = self.dense(cond_flat)  # (batch, C_out)
        beta = self.dense(cond_flat)   # (batch, C_out)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.C_out])
        beta = tf.reshape(beta, [-1, 1, 1, self.C_out])
        out = x * gamma + beta
        if self.return_params:
            return out, gamma, beta
        return out

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

class UDA_FiLM(tf.keras.Model):
    def __init__(self, dropOut=0, act='ReLU', dropOutPos=[2, 4], domain_disc = 'dense', extract_layer=3, first_layer_disc='pool', input_shape=(792, 14, 1), condition_shape=(792, 1)):
        super().__init__()

        self.extract_layer = extract_layer
        
        self.dropOut = dropOut
        self.dropOutPos = dropOutPos
        
        self.normalization = tf.keras.layers.BatchNormalization()
        
        self.conv1 = tf.keras.layers.Conv2D(32, (7, 3), strides=(1, 1), padding='valid')
        self.film1 = FiLM(32, return_params=True)
        
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 3), strides=(1, 1), padding='valid')
        self.film2 = FiLM(64, return_params=True)
        
        self.conv3 = tf.keras.layers.Conv2D(128, (5, 3), strides=(1, 1), padding='valid')
        self.film3 = FiLM(128, return_params=True)
        
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='valid')
        self.film4 = FiLM(128, return_params=True)
        
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid')
        self.film5 = FiLM(64, return_params=True)
        
        self.conv6 = tf.keras.layers.Conv2D(2, (3, 3), strides=(1, 1), padding='valid')
        
        
        # Select activation function
        if act == 'ReLU':
            self.activate = tf.keras.layers.ReLU()
        elif act == 'Tanh':
            self.activate = tf.keras.layers.Activation('tanh')
        elif act == 'Sigmoid':
            self.activate = tf.keras.layers.Activation('sigmoid')
        elif act == 'LeakyReLU':
            self.activate = tf.keras.layers.LeakyReLU(alpha=0.01)

        if dropOut != 0:
            self.dropout = tf.keras.layers.Dropout(dropOut)
            
        # Add GRL and domain discriminator
        self.grl = GradientReversal()
        if self.extract_layer == 3:
            extract_in_channels = 128
        elif self.extract_layer == 5:
            extract_in_channels = 64

        if domain_disc == 'hybrid':
            self.domain_discriminator = DomainDiscriminator2()
        else:
            self.domain_discriminator = DomainDiscriminator(in_channels=extract_in_channels, disc=domain_disc, first_layer=first_layer_disc)
        self.condition_domain_discriminator = DomainDiscriminator(disc='dense', first_layer=first_layer_disc, mode='condition')  # For condition domain classification
        
            
    def padLayer(self, x, top, bottom, left, right, mode='REFLECT'):  # mode can be 'REFLECT' or 'SYMMETRIC' or 'CONSTANT'
        return tf.pad(x, [[0, 0], [top, bottom], [left, right], [0, 0]], mode=mode)

    def call(self, inputs, training=False, return_domain=False, return_features=False):
        raw_input, condition = inputs
        # raw_input == (batch_size, nsubcs, nsymbs, 2)
        # condition == (batch_size, nsubcs, 2)
        
        x = self.normalization(raw_input, training=training)
        
        if 0 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        # Pass through Conv2D and FiLM layers
        x = self.conv1(self.padLayer(x, 3, 3, 1, 1))
        x = self.activate(x)
        x, gamma1, beta1 = self.film1(x, condition)
        if 1 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv2(self.padLayer(x, 2, 2, 1, 1))
        x = self.activate(x)
        x, gamma2, beta2 = self.film2(x, condition)
        if 2 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv3(self.padLayer(x, 2, 2, 1, 1))
        x = self.activate(x)
        x, gamma3, beta3 = self.film3(x, condition)
        if 3 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        # extract features for domain classification
        if self.extract_layer == 3:
            features = x  
                
        x = self.conv4(self.padLayer(x, 1, 1, 1, 1))
        x = self.activate(x)
        x, gamma4, beta4 = self.film4(x, condition)
        if 4 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        x = self.conv5(self.padLayer(x, 1, 1, 1, 1))
        x, gamma5, beta5 = self.film5(x, condition)
        if 5 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
            
        # extract features for domain classification
        if self.extract_layer == 5:
            features = x 
            

        x = self.conv6(self.padLayer(x, 1, 1, 1, 1))
        
        batch_size = x.shape[0]
        
        gammas = [gamma1, gamma2, gamma3, gamma4, gamma5]
        betas = [beta1, beta2, beta3, beta4, beta5]
        
        gamma_cat = tf.concat([tf.reshape(g, [batch_size, -1]) for g in gammas], axis=-1)  # (batch, sum(n_i))
        beta_cat  = tf.concat([tf.reshape(b, [batch_size, -1]) for b in betas], axis=-1)   # (batch, sum(n_i))
        film_features = tf.concat([gamma_cat, beta_cat], axis=-1)  # (batch, sum(n_i), 2)
    
        domain_pred = self.domain_discriminator(self.grl(features), training=training)
        condition_domain_pred = self.condition_domain_discriminator(self.grl(film_features), training=training)

        return UDA_FiLM_Output(x, domain_pred, condition_domain_pred, features, film_features)
    

def prepare_data(loader_H_1_6, nsymb=14):
    return utils_CNN_FiLM.prepare_data(loader_H_1_6, nsymb)
    
# Domain labels
def make_domain_labels(batch_size, domain):
    return tf.ones((batch_size, 1)) if domain == 'source' else tf.zeros((batch_size, 1))
    
def train_step(model, loader_H, loss_fn, lower_range, nsymb=14, lambda_domain=0.1, lamda_domain_cond=0.1, return_features=False):
    # save extracted features to .h5 temporary files to calculate PAD
    loader_H_input_train, loader_H_true_train, loader_H_train_target, loader_H_true_target = loader_H
    loss_fn_ce, loss_fn_domain, optimizer = loss_fn
        # loader_H_input_train, loader_H_true_train - in the source domain
        # loss_fn_ce - loss function for channel estimation
        # loss_fn_domain - loss function for domain classification
    
    epoc_loss            = 0.0
    epoc_loss_est        = 0.0
    epoc_loss_domain     = 0.0
    epoc_loss_est_target = 0.0
    #
    N_train_source = 0
    N_train_target = 0
        # N_train_source == N_train_target
    
    features_source_list = []
    film_features_source_list = []
    features_target_list = []
    film_features_target_list = []
    
    # Train for 1 epoch
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
        
    # Iterate over batches
    for batch_idx in range(loader_H_true_train.total_batches):
        if batch_idx % 20 == 0 or batch_idx == loader_H_true_train.total_batches - 1:
            print(f"batch_idx: {batch_idx+1}/{loader_H_true_train.total_batches}")
        x = loader_H_input_train.next_batch()  
        x_cond, x = prepare_data(x, nsymb=nsymb) # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y = loader_H_true_train.next_batch()  # Label: H_true
        y = y[:,:,70:84]  # only consider 14 symbols of  slot 6
        N_train_source += x.shape[0]
        #
        x_target = loader_H_train_target.next_batch()  # Target domain data
        x_cond_target, x_target = prepare_data(x_target, nsymb=nsymb)  # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y_target = loader_H_true_target.next_batch()  # Target domain label - just for monitoring
        y_target = y_target[:,:,70:84]
        N_train_target += x_target.shape[0]
    
        # Ensure tensors and float32 dtype
        x = utils_CNN.complx2real(x)  # (batch_size, 2, nsubcs, nsymbs)
        y = utils_CNN.complx2real(y)  
        x_cond = utils_CNN.complx2real(x_cond)  # (batch_size, 2, nsubcs, 1)
        #
        x_target = utils_CNN.complx2real(x_target)  # (batch_size, 2, nsubcs, nsymbs)
        y_target = utils_CNN.complx2real(y_target)
        x_cond_target = utils_CNN.complx2real(x_cond_target)  # (batch_size, 2, nsubcs, 1)
        
        x      = np.transpose(x, (0, 2, 3, 1)) # (batch_size, nsubcs, nsymbs, 2)
        x_cond = np.transpose(x_cond, (0, 2, 3, 1)) # (batch_size, nsubcs, 1, 2)
        y      = np.transpose(y, (0, 2, 3, 1)) 
        x_target      = np.transpose(x_target, (0, 2, 3, 1))
        x_cond_target = np.transpose(x_cond_target, (0, 2, 3, 1)) # (batch_size, nsubcs, 1, 2)
        y_target      = np.transpose(y_target, (0, 2, 3, 1))
        
        # min_max scale 
        x_all    = np.concatenate([x_cond, x], axis=2)  # shape: (batch_size, nsubcs, 1+nsymbs, 2)
        x_all_scaled, _, _ = utils_CNN.minmaxScaler(x_all, lower_range=lower_range)
        x_cond   = x_all_scaled[:, :, :1, :]      # (batch_size, nsubcs, 1, 2)
        x_scaled = x_all_scaled[:, :, 1:, :]       # (batch_size, nsubcs, nsymbs, 2)
        y_scaled, _, _ = utils_CNN.minmaxScaler(y, lower_range=lower_range)
        #
        x_all_target    = np.concatenate([x_cond_target, x_target], axis=2)  # shape: (batch_size, nsubcs, 1+nsymbs, 2)
        x_all_scaled_target, _, _ = utils_CNN.minmaxScaler(x_all_target, lower_range=lower_range)
        x_cond_target   = x_all_scaled_target[:, :, :1, :]      # (batch_size, nsubcs, 1, 2)
        x_target_scaled = x_all_scaled_target[:, :, 1:, :]       # (batch_size, nsubcs, nsymbs, 2)
        y_target_scaled, _, _ = utils_CNN.minmaxScaler(y_target, lower_range=lower_range)

        # GradientTape for automatic differentiation
        with tf.GradientTape() as tape:
            UDA_FiLM_output_source  = model([x_scaled, x_cond], training=True)         
            predictions_source = UDA_FiLM_output_source.x
            domain_pred_source = UDA_FiLM_output_source.domain_pred
            domain_pred_source_cond = UDA_FiLM_output_source.condition_domain_pred
            
            if return_features:
                # save features in a temporary file instead of stacking them up, to avoid memory exploding
                features_np_source = UDA_FiLM_output_source.features.numpy()  # Convert to numpy if it's a tensor
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
            
            loss_est = loss_fn_ce(y_scaled, predictions_source)

            # Forward pass on target domain (only for domain)
            UDA_FiLM_output_target = model([x_target_scaled, x_cond_target], training=True, return_domain=True, return_features=return_features)
            predictions_target = UDA_FiLM_output_target.x
            domain_pred_target = UDA_FiLM_output_target.domain_pred
            domain_pred_target_cond = UDA_FiLM_output_target.condition_domain_pred
            
            if return_features:
                # save features in a temporary file instead of stacking them up, to avoid memory exploding
                features_np_target = UDA_FiLM_output_target.features.numpy()
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
                
            
            # Domain labels
            source_domain_labels = make_domain_labels(tf.shape(x_scaled)[0], 'source')
            target_domain_labels = make_domain_labels(tf.shape(x_target_scaled)[0], 'target')
            source_domain_labels_cond = make_domain_labels(tf.shape(x_cond)[0], 'source')
            target_domain_labels_cond = make_domain_labels(tf.shape(x_cond_target)[0], 'target')
            
            # Domain classification loss (on both)
            domain_loss_source = loss_fn_domain(source_domain_labels, domain_pred_source)
            domain_loss_target = loss_fn_domain(target_domain_labels, domain_pred_target)
            domain_loss_source_cond = loss_fn_domain(source_domain_labels_cond, domain_pred_source_cond)
            domain_loss_target_cond = loss_fn_domain(target_domain_labels_cond, domain_pred_target_cond)
            
            domain_loss      = domain_loss_source + domain_loss_target
            domain_loss_cond = domain_loss_source_cond + domain_loss_target_cond
            
            # Total loss: Estimation + λ * Domain Loss
            # lambda_domain = 0.1  # weight for domain loss
            total_loss = loss_est + lambda_domain * domain_loss + lamda_domain_cond * domain_loss_cond
            
        batch_size = x.shape[0]
        # just for monitoring, not used in training
        target_loss_est = loss_fn_ce(y_target_scaled, predictions_target)
        epoc_loss_est_target += target_loss_est.numpy() * batch_size
        
        # Backpropagation
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoc_loss += total_loss.numpy() * batch_size
        epoc_loss_est += loss_est.numpy() * batch_size
        epoc_loss_domain += domain_loss.numpy() * batch_size
    # end batch loop
    
    if return_features:    
        features_h5_source.close()
        features_h5_target.close()
    
    avg_epoc_loss_est        = epoc_loss_est / N_train_source
    avg_epoc_loss_domain     = epoc_loss_domain / (N_train_source + N_train_target)
    avg_epoc_loss            = epoc_loss / (N_train_source + N_train_target)
    avg_epoc_loss_est_target = epoc_loss_est_target / N_train_target
    
    if features_source_list:
        features_source = tf.concat(features_source_list, axis=0)
        film_features_source = tf.concat(film_features_source_list, axis=0)
        features_target = tf.concat(features_target_list, axis=0)
        film_features_target = tf.concat(film_features_target_list, axis=0)
    else:
        features_source = None
        film_features_source = None
        features_target = None
        film_features_target = None
    
    # can return domain_loss_cond also but no need for now     
    return train_step_Output(avg_epoc_loss_est, avg_epoc_loss_domain, avg_epoc_loss, avg_epoc_loss_est_target, 
                            features_source, film_features_source, features_target, film_features_target)
                            # x == (batch_size, nsubcs, nsymbs, 2) real matrix

def train_step2(model, sgd_classifiers, loader_H, loss_fn, lower_range, nsymb=14, lambda_domain=0.1, lamda_domain_cond=0.1, return_features=False):
    # train SGDClassifier (to calculate PAD) along with the original (estimation) network
    # return_features == return features to calculate PAD
    loader_H_input_train, loader_H_true_train, loader_H_train_target, loader_H_true_target = loader_H
    loss_fn_ce, loss_fn_domain, optimizer = loss_fn
        # loader_H_input_train, loader_H_true_train - in the source domain
        # loss_fn_ce - loss function for channel estimation
        # loss_fn_domain - loss function for domain classification
    
    epoc_loss            = 0.0
    epoc_loss_est        = 0.0
    epoc_loss_domain     = 0.0
    epoc_loss_est_target = 0.0
    #
    N_train_source = 0
    N_train_target = 0
        # N_train_source == N_train_target
    
    features_source_list = []
    film_features_source_list = []
    features_target_list = []
    film_features_target_list = []

    if return_features:
        # return_features == return features to calculate PAD
        n_classifiers = len(sgd_classifiers)
        test_errors = [[] for _ in range(n_classifiers)]  # List of lists for each classifier
    
    # Train for 1 epoch
    # Iterate over batches
    for batch_idx in range(loader_H_true_train.total_batches):
        if batch_idx % 20 == 0 or batch_idx == loader_H_true_train.total_batches - 1:
            print(f"batch_idx: {batch_idx+1}/{loader_H_true_train.total_batches}")
        x = loader_H_input_train.next_batch()  
        x_cond, x = prepare_data(x, nsymb=nsymb) # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y = loader_H_true_train.next_batch()  # Label: H_true
        y = y[:,:,70:84]  # only consider 14 symbols of  slot 6
        N_train_source += x.shape[0]
        #
        x_target = loader_H_train_target.next_batch()  # Target domain data
        x_cond_target, x_target = prepare_data(x_target, nsymb=nsymb)  # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y_target = loader_H_true_target.next_batch()  # Target domain label - just for monitoring
        y_target = y_target[:,:,70:84]
        N_train_target += x_target.shape[0]
    
        # Ensure tensors and float32 dtype
        x = utils_CNN.complx2real(x)  # (batch_size, 2, nsubcs, nsymbs)
        y = utils_CNN.complx2real(y)  
        x_cond = utils_CNN.complx2real(x_cond)  # (batch_size, 2, nsubcs, 1)
        #
        x_target = utils_CNN.complx2real(x_target)  # (batch_size, 2, nsubcs, nsymbs)
        y_target = utils_CNN.complx2real(y_target)
        x_cond_target = utils_CNN.complx2real(x_cond_target)  # (batch_size, 2, nsubcs, 1)
        
        x      = np.transpose(x, (0, 2, 3, 1)) # (batch_size, nsubcs, nsymbs, 2)
        x_cond = np.transpose(x_cond, (0, 2, 3, 1)) # (batch_size, nsubcs, 1, 2)
        y      = np.transpose(y, (0, 2, 3, 1)) 
        x_target      = np.transpose(x_target, (0, 2, 3, 1))
        x_cond_target = np.transpose(x_cond_target, (0, 2, 3, 1)) # (batch_size, nsubcs, 1, 2)
        y_target      = np.transpose(y_target, (0, 2, 3, 1))
        
        # min_max scale 
        x_all    = np.concatenate([x_cond, x], axis=2)  # shape: (batch_size, nsubcs, 1+nsymbs, 2)
        x_all_scaled, _, _ = utils_CNN.minmaxScaler(x_all, lower_range=lower_range)
        x_cond   = x_all_scaled[:, :, :1, :]      # (batch_size, nsubcs, 1, 2)
        x_scaled = x_all_scaled[:, :, 1:, :]       # (batch_size, nsubcs, nsymbs, 2)
        y_scaled, _, _ = utils_CNN.minmaxScaler(y, lower_range=lower_range)
        #
        x_all_target    = np.concatenate([x_cond_target, x_target], axis=2)  # shape: (batch_size, nsubcs, 1+nsymbs, 2)
        x_all_scaled_target, _, _ = utils_CNN.minmaxScaler(x_all_target, lower_range=lower_range)
        x_cond_target   = x_all_scaled_target[:, :, :1, :]      # (batch_size, nsubcs, 1, 2)
        x_target_scaled = x_all_scaled_target[:, :, 1:, :]       # (batch_size, nsubcs, nsymbs, 2)
        y_target_scaled, _, _ = utils_CNN.minmaxScaler(y_target, lower_range=lower_range)

        # GradientTape for automatic differentiation
        with tf.GradientTape() as tape:
            UDA_FiLM_output_source  = model([x_scaled, x_cond], training=True)         
            predictions_source = UDA_FiLM_output_source.x
            domain_pred_source = UDA_FiLM_output_source.domain_pred
            domain_pred_source_cond = UDA_FiLM_output_source.condition_domain_pred
            #
            loss_est = loss_fn_ce(y_scaled, predictions_source)
            #
            # Forward pass on target domain
            UDA_FiLM_output_target = model([x_target_scaled, x_cond_target], training=True, return_domain=True, return_features=return_features)
            predictions_target = UDA_FiLM_output_target.x
            domain_pred_target = UDA_FiLM_output_target.domain_pred
            domain_pred_target_cond = UDA_FiLM_output_target.condition_domain_pred
            
            ### For calculating PAD   # return_features == return features to calculate PAD
            if return_features:
                if batch_idx == 0:
                    # Initialize with all classes for partial_fit
                    feature_dim = np.prod(UDA_FiLM_output_source.features.shape[1:]) 
                    for clf in sgd_classifiers:
                        clf.partial_fit(np.zeros((1, feature_dim)), np.array([0]), classes=np.array([0, 1]))

                # Extract and reshape features for source and target
                features_np_source = UDA_FiLM_output_source.features.numpy().reshape(UDA_FiLM_output_source.features.shape[0], -1).astype(np.float64)
                features_np_target = UDA_FiLM_output_target.features.numpy().reshape(UDA_FiLM_output_target.features.shape[0], -1).astype(np.float64)

                # Create labels: 0 for source, 1 for target
                source_labels = np.zeros(features_np_source.shape[0], dtype=int)
                target_labels = np.ones(features_np_target.shape[0], dtype=int)

                if batch_idx < loader_H_true_train.total_batches // 2:
                    # TRAINING  # Online update for each classifier
                    for clf in sgd_classifiers:
                        clf.partial_fit(features_np_source, source_labels)
                        clf.partial_fit(features_np_target, target_labels)
                else:
                    # TESTING: accumulate error for each classifier
                    for i, clf in enumerate(sgd_classifiers):
                        # Predict on source and target
                        pred_source = clf.predict(features_np_source)
                        pred_target = clf.predict(features_np_target)
                        # Calculate error (misclassification rate)
                        error_source = np.mean(pred_source != source_labels)
                        error_target = np.mean(pred_target != target_labels)
                        # Store average error for this batch
                        test_errors[i].append((error_source + error_target) / 2)
            ###

            ## For Domain Discriminator
            # Domain labels
            source_domain_labels = make_domain_labels(tf.shape(x_scaled)[0], 'source')
            target_domain_labels = make_domain_labels(tf.shape(x_target_scaled)[0], 'target')
            source_domain_labels_cond = make_domain_labels(tf.shape(x_cond)[0], 'source')
            target_domain_labels_cond = make_domain_labels(tf.shape(x_cond_target)[0], 'target')
            
            # Domain classification loss (on both)
            domain_loss_source = loss_fn_domain(source_domain_labels, domain_pred_source)
            domain_loss_target = loss_fn_domain(target_domain_labels, domain_pred_target)
            domain_loss_source_cond = loss_fn_domain(source_domain_labels_cond, domain_pred_source_cond)
            domain_loss_target_cond = loss_fn_domain(target_domain_labels_cond, domain_pred_target_cond)
            
            domain_loss      = domain_loss_source + domain_loss_target
            domain_loss_cond = domain_loss_source_cond + domain_loss_target_cond
            
            # Total loss: Estimation + λ * Domain Loss
            # lambda_domain = 0.1  # weight for domain loss
            total_loss = loss_est + lambda_domain * domain_loss + lamda_domain_cond * domain_loss_cond
            
        batch_size = x.shape[0]
        # just for monitoring, not used in training
        target_loss_est = loss_fn_ce(y_target_scaled, predictions_target)
        epoc_loss_est_target += target_loss_est.numpy() * batch_size
        
        # Backpropagation
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoc_loss += total_loss.numpy() * batch_size
        epoc_loss_est += loss_est.numpy() * batch_size
        epoc_loss_domain += domain_loss.numpy() * batch_size
    # end batch loop
    if return_features:
        # Calculate average error for each SGDClassifier
        avg_errors = [np.mean(errs) for errs in test_errors]
        min_error = min(avg_errors)
        pad = 2 * (1 - 2 * min_error)
        print(f'Proxy A-Distance: ', pad, '\n')
    
    avg_epoc_loss_est        = epoc_loss_est / N_train_source
    avg_epoc_loss_domain     = epoc_loss_domain / (N_train_source + N_train_target)
    avg_epoc_loss            = epoc_loss / (N_train_source + N_train_target)
    avg_epoc_loss_est_target = epoc_loss_est_target / N_train_target
    
    if features_source_list:
        features_source = tf.concat(features_source_list, axis=0)
        film_features_source = tf.concat(film_features_source_list, axis=0)
        features_target = tf.concat(features_target_list, axis=0)
        film_features_target = tf.concat(film_features_target_list, axis=0)
    else:
        features_source = None
        film_features_source = None
        features_target = None
        film_features_target = None
    
    # can return domain_loss_cond also but no need for now     
    return train_step_Output(avg_epoc_loss_est, avg_epoc_loss_domain, avg_epoc_loss, avg_epoc_loss_est_target, 
                            features_source, film_features_source, features_target, film_features_target, pad)
                            # x == (batch_size, nsubcs, nsymbs, 2) real matrix

def val_step(model, loader_H, loss_fn, lower_range, epoch=None, nsymb=14, lambda_domain=0.1, lambda_domain_cond=0.1):
    loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target = loader_H
    loss_fn_est, loss_fn_domain = loss_fn
    H_sample = []
    N_val_source = 0 # number of evaluating samples in source domain
    N_val_target = 0 
    
    epoc_total_loss = 0.0
    epoc_val_domain_loss_source = 0.0
    epoc_val_est_loss_source    = 0.0
    epoc_source_acc             = 0.0
    epoc_nmse_val_source        = 0.0
    # 
    epoc_val_domain_loss_target = 0.0
    epoc_val_est_loss_target    = 0.0
    epoc_target_acc             = 0.0
    epoc_nmse_val_target        = 0.0
    
    # Evaluate after 1 epoch
    # Iterate over batches
    for idx in range(loader_H_true_val_source.total_batches):
        x = loader_H_input_val_source.next_batch()
        x_cond, x = prepare_data(x, nsymb= nsymb) # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y = loader_H_true_val_source.next_batch()
        y = y[:, :, 70:84]
        #
        x_target = loader_H_input_val_target.next_batch()
        x_cond_target, x_target = prepare_data(x_target, nsymb=nsymb)  # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y_target = loader_H_true_val_target.next_batch()
        y_target = y_target[:, :, 70:84]
        #
        N_val_source += x.shape[0]
        N_val_target += x_target.shape[0]
        
        x_real      = utils_CNN.complx2real(x)   # Tensor [nbatch, 2, nsubcs, nsymbs]
        x_real_cond = utils_CNN.complx2real(x_cond) # Tensor [nbatch, 2, nsubcs, 1]
        y_real      = utils_CNN.complx2real(y)
        x_real      = np.transpose(x_real, (0, 2, 3, 1)) # Tensor [nbatch, nsubcs, nsymbs, 2] --- # to fit CNN size
        x_real_cond = np.transpose(x_real_cond, (0, 2, 3, 1)) # Tensor [nbatch, nsubcs, 1, 2]
        y_real      = np.transpose(y_real, (0, 2, 3, 1))
        #
        x_target_real      = utils_CNN.complx2real(x_target)   # Tensor [nbatch, 2, nsubcs, nsymbs+1]
        x_target_real_cond = utils_CNN.complx2real(x_cond_target) # Tensor [nbatch, 2, nsubcs, 1]
        y_target_real      = utils_CNN.complx2real(y_target)
        x_target_real      = np.transpose(x_target_real, (0, 2, 3, 1))
        x_target_real_cond = np.transpose(x_target_real_cond, (0, 2, 3, 1)) # Tensor [nbatch, nsubcs, 1, 2]
        y_target_real      = np.transpose(y_target_real, (0, 2, 3, 1))

        if (epoch is not None) and (idx == 0):
            H_true_sample     = y_real[:3,:,:,:].copy() # [3, nsubcs, nsymbs, 2]
            H_input_sample    = x_real[:3,:,:,:].copy() # [3, nsubcs, nsymbs, 2] -- to calculate NMSE and visualization
            H_input_condition = x_real_cond[:3,:,:,:].copy() # [3, nsubcs, 1, 2] -- the CSI-RS of slot 1 -- for visualization
                                # of sample 0,1,2
            H_true_sample_target     = y_target_real[:3,:,:,:].copy() # [3, nsubcs, nsymbs, 2]
            H_input_sample_target    = x_target_real[:3,:,:,:].copy() # [3, nsubcs, nsymbs, 2] -- to calculate NMSE and visualization
            H_input_condition_target = x_target_real_cond[:3,:,:,:].copy() # [3, nsubcs, 1, 2] -- the CSI-RS of slot 1 -- for visualization
            nmse_input_source = 0
            nmse_est_source   = 0
            nmse_input_target = 0
            nmse_est_target   = 0

        # ======= source domain ========
        x_all           = np.concatenate([x_real_cond, x_real], axis=2)  # shape: (batch_size, nsubcs, 1+nsymbs, 2)
        x_all_scaled, x_min, x_max = utils_CNN.minmaxScaler(x_all, lower_range=lower_range)
        x_cond_scaled   = x_all_scaled[:, :, :1, :]      # (batch_size, nsubcs, 1, 2)
        x_scaled        = x_all_scaled[:, :, 1:, :]       # (batch_size, nsubcs, nsymbs, 2)
        y_scaled, _, _  = utils_CNN.minmaxScaler(y_real, lower_range=lower_range)
        ###############
        UDA_FiLM_output_source = model([x_scaled, x_cond_scaled], training=False, return_domain=True)
        preds        = UDA_FiLM_output_source.x  # (batch_size, nsubcs, nsymbs, 2)
        domain_preds = UDA_FiLM_output_source.domain_pred  # (batch_size, 1)
        # domain_preds_cond = UDA_FiLM_output_source.condition_domain_pred  # (batch_size, 1)
        preds_descaled = utils_CNN.deMinMax(preds, x_min, x_max, lower_range=lower_range)
        ###############
        if (epoch is not None) and (idx == 0):
            H_est_sample = tf.identity(preds_descaled[:3,:,:,:]) # [3,nsubcs, nsymbs, 2]
            mse_sample_source   = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3)) # shape (3,)
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source     = mse_sample_source / (power_sample_source + 1e-30)  # avoid divide-by-zero
            #
            mse_input_source  = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)    # shape (3,)
            domain_sample_return_source = ((domain_preds[:3] >= 0.5).numpy()).astype(np.int32)
            
        # ======= target domain ========
        x_all_target          = np.concatenate([x_target_real_cond, x_target_real], axis=2)  # shape: (batch_size, nsubcs, 1+nsymbs, 2)
        x_all_scaled_target, x_min_target, x_max_target = utils_CNN.minmaxScaler(x_all_target, lower_range=lower_range)
        x_cond_scaled_target  = x_all_scaled_target[:, :, :1, :]      # (batch_size, nsubcs, 1, 2)
        x_scaled_target       = x_all_scaled_target[:, :, 1:, :]       # (batch_size, nsubcs, nsymbs, 2)
        y_scaled_target, _, _ = utils_CNN.minmaxScaler(y_target_real, lower_range=lower_range)
        ##############
        UDA_FiLM_output_target = model([x_scaled_target, x_cond_scaled_target], training=False, return_domain=True)
        preds_target        = UDA_FiLM_output_target.x  # (batch_size, nsubcs, nsymbs, 2)
        domain_preds_target = UDA_FiLM_output_target.domain_pred  # (batch_size, 1)
        # domain_preds_cond_target = UDA_FiLM_output_target.condition_domain_pred  # (batch_size, 1)
        preds_descaled_target = utils_CNN.deMinMax(preds_target, x_min_target, x_max_target, lower_range=lower_range)
        ##############
        if  (epoch is not None) and (idx == 0):
            H_est_sample_target = tf.identity(preds_descaled_target[:3,:,:,:]) # [3,nsubcs, nsymbs, 2]
            mse_sample_target = np.mean((H_est_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            power_sample_target = np.mean(H_true_sample_target ** 2, axis=(1, 2, 3))
            nmse_est_target = mse_sample_target / (power_sample_target + 1e-30)  # avoid divide-by-zero
            #
            mse_input_target = np.mean((H_input_sample_target - H_true_sample_target) ** 2, axis=(1, 2, 3))
            nmse_input_target = mse_input_target / (power_sample_target + 1e-30)   # shape (3,)
            domain_sample_return_target = ((domain_preds_target[:3] >= 0.5).numpy()).astype(np.int32)
        
        # ++++++ to return ++++++
        if  (epoch is not None) and (idx == 0):
            H_sample = [H_true_sample,  H_input_sample, H_est_sample, H_input_condition, 
                    nmse_input_source, nmse_est_source, domain_sample_return_source,
                    H_true_sample_target, H_input_sample_target, H_est_sample_target, H_input_condition_target,
                    nmse_input_target, nmse_est_target, domain_sample_return_target]
        
        
        # ============ Compute batch losses ============
        # (the losses the same as the losses in training)
        # === Domain discrimination loss ===
        # Source domain
        source_domain_labels = make_domain_labels(tf.shape(x_scaled)[0], 'source')
        domain_loss_source = loss_fn_domain(source_domain_labels, domain_preds)
        epoc_val_domain_loss_source += domain_loss_source * x.shape[0]
        # Target domain
        target_domain_labels = make_domain_labels(tf.shape(x_scaled_target)[0], 'target')
        domain_loss_target = loss_fn_domain(target_domain_labels, domain_preds_target)
        epoc_val_domain_loss_target += domain_loss_target * x_target.shape[0]
        #
        # === Estimation loss === 
        # Source domain
        batch_est_loss_source = loss_fn_est(y_scaled, preds).numpy()
        epoc_val_est_loss_source += batch_est_loss_source * x.shape[0]  # weight by batch size
        # Target domain
        batch_est_loss_target = loss_fn_est(y_scaled_target, preds_target).numpy()
        epoc_val_est_loss_target += batch_est_loss_target * x_target.shape[0]
        #   
        # ------ total loss ------
        # total loss = estimation (source) + lambda * DomainLoss
        total_loss = batch_est_loss_source + lambda_domain * (domain_loss_source + domain_loss_target)  # average over 1 batch
        epoc_total_loss += total_loss* x.shape[0]   # sum over 1 batch

        # ============ Evaluate Performance of Estimation and Domain Discrimination ============
        # === Evaluate domain discrimination performance ===
        # Source domain
        source_domain_labels = np.ones_like(domain_preds)
        source_preds = ((domain_preds >= 0.5).numpy()).astype(np.int32)
        source_acc = accuracy_score(source_domain_labels, source_preds) # batch accuracy
        epoc_source_acc += source_acc * x.shape[0]
        # Target domain
        target_domain_labels = np.zeros_like(domain_preds_target)
        target_preds = ((domain_preds_target >= 0.5).numpy()).astype(np.int32)
        target_acc = accuracy_score(target_domain_labels, target_preds) # batch accuracy
        epoc_target_acc += target_acc * x_target.shape[0]
        #
        # === Evaluate estimation performance ===
        # Source domain
        mse_val_source = np.mean((preds_descaled - y_real) ** 2, axis=(1, 2, 3))
        power_source = np.mean(y_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_source += np.mean(mse_val_source / (power_source + 1e-30)) * x.shape[0]  # avoid divide-by-zero
        # Target domain
        mse_val_target = np.mean((preds_descaled_target - y_target_real) ** 2, axis=(1, 2, 3))
        power_target = np.mean(y_target_real ** 2, axis=(1, 2, 3))
        epoc_nmse_val_target += np.mean(mse_val_target / (power_target + 1e-30)) *x_target.shape[0]  # avoid divide-by-zero
        
    # end of batch loop
    
    # ======== Calculate average over all samples (epoch) ========
    # === Losses ===
    N_val = N_val_source + N_val_target
    avg_epoc_total_loss = epoc_total_loss / N_val
    #
    avg_epoc_val_domain_loss_source = epoc_val_domain_loss_source / N_val_source
    avg_epoc_val_domain_loss_target = epoc_val_domain_loss_target / N_val_target
    avg_epoc_val_domain_loss        = (epoc_val_domain_loss_source + epoc_val_domain_loss_target) / N_val
    #
    avg_epoc_val_est_loss_source = epoc_val_est_loss_source / N_val_source
    avg_epoc_val_est_loss_target = epoc_val_est_loss_target / N_val_target
    avg_epoc_val_est_loss        = (epoc_val_est_loss_source + epoc_val_est_loss_target) / N_val
    #
    # === Performance ===
    avg_epoc_source_acc = epoc_source_acc / N_val_source
    avg_epoc_target_acc = epoc_target_acc / N_val_target
    avg_epoc_acc        = (epoc_source_acc + epoc_target_acc) / N_val
    #
    avg_epoc_nmse_val_source = epoc_nmse_val_source / N_val_source
    avg_epoc_nmse_val_target = epoc_nmse_val_target / N_val_target
    avg_epoc_nmse_val        = (epoc_nmse_val_source + epoc_nmse_val_target) / N_val
    
        
    # ++++++ to return ++++++
    epoc_eval_return = [avg_epoc_total_loss,
                        avg_epoc_val_domain_loss_source, avg_epoc_val_domain_loss_target, avg_epoc_val_domain_loss,
                        avg_epoc_val_est_loss_source, avg_epoc_val_est_loss_target, avg_epoc_val_est_loss,
                        avg_epoc_source_acc, avg_epoc_target_acc, avg_epoc_acc,
                        avg_epoc_nmse_val_source, avg_epoc_nmse_val_target, avg_epoc_nmse_val
                        ]
        
        
    return H_sample, epoc_eval_return

class UDA_FiLM_8conv(tf.keras.Model):
    def __init__(self, dropOut=0, act='ReLU', dropOutPos=[2, 4], domain_disc = 'dense', extract_layer=5, first_layer_disc='pool', input_shape=(792, 14, 1), condition_shape=(792, 1)):
        super().__init__()

        self.extract_layer = extract_layer
        
        self.dropOut = dropOut
        self.dropOutPos = dropOutPos
        
        self.normalization = tf.keras.layers.BatchNormalization()
        
        self.conv1 = tf.keras.layers.Conv2D(32, (7, 3), strides=(1, 1), padding='valid')
        self.film1 = FiLM(32, return_params=True)
        
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 3), strides=(1, 1), padding='valid')
        self.film2 = FiLM(64, return_params=True)
        
        self.conv3 = tf.keras.layers.Conv2D(128, (5, 3), strides=(1, 1), padding='valid')
        self.film3 = FiLM(128, return_params=True)
        
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='valid')
        self.film4 = FiLM(128, return_params=True)

        self.conv5 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='valid')
        self.film5 = FiLM(128, return_params=True)

        self.conv6 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='valid')
        self.film6 = FiLM(128, return_params=True)
        
        self.conv7 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid')
        self.film7 = FiLM(64, return_params=True)
        
        self.conv8 = tf.keras.layers.Conv2D(2, (3, 3), strides=(1, 1), padding='valid')
        
        
        # Select activation function
        if act == 'ReLU':
            self.activate = tf.keras.layers.ReLU()
        elif act == 'Tanh':
            self.activate = tf.keras.layers.Activation('tanh')
        elif act == 'Sigmoid':
            self.activate = tf.keras.layers.Activation('sigmoid')
        elif act == 'LeakyReLU':
            self.activate = tf.keras.layers.LeakyReLU(alpha=0.01)

        if dropOut != 0:
            self.dropout = tf.keras.layers.Dropout(dropOut)
            
        # Add GRL and domain discriminator
        self.grl = GradientReversal()
        if self.extract_layer == 3:
            extract_in_channels = 128
        elif self.extract_layer == 5:
            extract_in_channels = 64

        if domain_disc == 'hybrid':
            self.domain_discriminator = DomainDiscriminator2()
        else:
            self.domain_discriminator = DomainDiscriminator(in_channels=extract_in_channels, disc=domain_disc, first_layer=first_layer_disc)
        self.condition_domain_discriminator = DomainDiscriminator(disc='dense', first_layer=first_layer_disc, mode='condition')  # For condition domain classification
        
            
    def padLayer(self, x, top, bottom, left, right, mode='REFLECT'):  # mode can be 'REFLECT' or 'SYMMETRIC' or 'CONSTANT'
        return tf.pad(x, [[0, 0], [top, bottom], [left, right], [0, 0]], mode=mode)

    def call(self, inputs, training=False, return_domain=False, return_features=False):
        raw_input, condition = inputs
        # raw_input == (batch_size, nsubcs, nsymbs, 2)
        # condition == (batch_size, nsubcs, 2)
        
        x = self.normalization(raw_input, training=training)
        
        if 0 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        # Pass through Conv2D and FiLM layers
        x = self.conv1(self.padLayer(x, 3, 3, 1, 1))
        x = self.activate(x)
        x, gamma1, beta1 = self.film1(x, condition)
        if 1 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv2(self.padLayer(x, 2, 2, 1, 1))
        x = self.activate(x)
        x, gamma2, beta2 = self.film2(x, condition)
        if 2 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv3(self.padLayer(x, 2, 2, 1, 1))
        x = self.activate(x)
        x, gamma3, beta3 = self.film3(x, condition)
        if 3 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        # extract features for domain classification
        if self.extract_layer == 3:
            features = x  
                
        x = self.conv4(self.padLayer(x, 1, 1, 1, 1))
        x = self.activate(x)
        x, gamma4, beta4 = self.film4(x, condition)
        if 4 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        x = self.conv5(self.padLayer(x, 1, 1, 1, 1))
        x = self.activate(x)
        x, gamma5, beta5 = self.film5(x, condition)
        if 5 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        # extract features for domain classification
        if self.extract_layer == 5:
            features = x 
        
        x = self.conv6(self.padLayer(x, 1, 1, 1, 1))
        x = self.activate(x)
        x, gamma6, beta6 = self.film6(x, condition)
        if 6 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        x = self.conv7(self.padLayer(x, 1, 1, 1, 1))
        x, gamma7, beta7 = self.film7(x, condition)
        if 7 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
            
        # extract features for domain classification
        if self.extract_layer == 7:
            features = x 
            

        x = self.conv8(self.padLayer(x, 1, 1, 1, 1))
        
        batch_size = x.shape[0]
        
        gammas = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7]
        betas = [beta1, beta2, beta3, beta4, beta5, beta6, beta7]
        
        gamma_cat = tf.concat([tf.reshape(g, [batch_size, -1]) for g in gammas], axis=-1)  # (batch, sum(n_i))
        beta_cat  = tf.concat([tf.reshape(b, [batch_size, -1]) for b in betas], axis=-1)   # (batch, sum(n_i))
        film_features = tf.concat([gamma_cat, beta_cat], axis=-1)  # (batch, sum(n_i), 2)
    
        domain_pred = self.domain_discriminator(self.grl(features), training=training)
        condition_domain_pred = self.condition_domain_discriminator(self.grl(film_features), training=training)

        return UDA_FiLM_Output(x, domain_pred, condition_domain_pred, features, film_features)
    

class UDA_FiLM_UpDown(tf.keras.Model):
    def __init__(self, dropOut=0, act='ReLU', dropOutPos=[3, 4, 5], domain_disc = 'dense', extract_layer=4, first_layer_disc='pool', input_shape=(792, 14, 1), condition_shape=(792, 1)):
        super().__init__()

        self.extract_layer = extract_layer
        
        self.dropOut = dropOut
        self.dropOutPos = dropOutPos
        
        self.normalization = tf.keras.layers.BatchNormalization()
        
        self.conv1 = tf.keras.layers.Conv2D(32, (4, 3), strides=(2, 1))
        # self.film1 = FiLM2(32, H_out=395, return_params=True)
        self.film1 = FiLM3(32, return_params=True)

        self.conv2 = tf.keras.layers.Conv2D(64, (5, 3), strides=(2, 1))
        # self.film2 = FiLM2(64, H_out=196, return_params=True)
        self.film2 = FiLM3(64, return_params=True)
        
        self.conv3 = tf.keras.layers.Conv2D(128, (4, 3), strides=(2, 1))
        # self.film3 = FiLM2(128, H_out=97, return_params=True)
        self.film3 = FiLM3(128, return_params=True)
        
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 1))
        # self.film4 = FiLM2(128, H_out=48, return_params=True)
        self.film4 = FiLM3(128, return_params=True)

        self.conv5 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 1))
        # self.film5 = FiLM2(128, H_out=97, return_params=True)   
        self.film5 = FiLM3(128, return_params=True)

        self.conv6 = tf.keras.layers.Conv2DTranspose(64, (4, 3), strides=(2, 1))
        # self.film6 = FiLM2(64, H_out=196, return_params=True)
        self.film6 = FiLM3(64, return_params=True)
        
        self.conv7 = tf.keras.layers.Conv2DTranspose(32, (5, 3), strides=(2, 1))
        # self.film7 = FiLM2(32, H_out=395, return_params=True)
        self.film7 = FiLM3(32, return_params=True)
        
        self.conv8 = tf.keras.layers.Conv2DTranspose(2, (4, 3), strides=(2, 1))
        
        
        # Select activation function
        if act == 'ReLU':
            self.activate = tf.keras.layers.ReLU()
        elif act == 'Tanh':
            self.activate = tf.keras.layers.Activation('tanh')
        elif act == 'Sigmoid':
            self.activate = tf.keras.layers.Activation('sigmoid')
        elif act == 'LeakyReLU':
            self.activate = tf.keras.layers.LeakyReLU(alpha=0.01)

        if dropOut != 0:
            self.dropout = tf.keras.layers.Dropout(dropOut)
            
        # Add GRL and domain discriminator
        self.grl = GradientReversal()
        if self.extract_layer in (3, 4, 5):
            extract_in_channels = 128

        if domain_disc == 'hybrid':
            self.domain_discriminator = DomainDiscriminator2()
        else:
            self.domain_discriminator = DomainDiscriminator(in_channels=extract_in_channels, disc=domain_disc, first_layer=first_layer_disc)
        self.condition_domain_discriminator = DomainDiscriminator(disc='dense', first_layer=first_layer_disc, mode='condition')  # For condition domain classification
        
        

    def call(self, inputs, training=False, return_domain=False, return_features=False):
        raw_input, condition = inputs
        # raw_input == (batch_size, nsubcs, nsymbs, 2)
        # condition == (batch_size, nsubcs, 2)
        
        x = self.normalization(raw_input, training=training)
        
        if 0 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        # Pass through Conv2D and FiLM layers
        x = self.conv1(x)
        x = self.activate(x)
        x, gamma1, beta1 = self.film1(x, condition)
        if 1 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.activate(x)
        x, gamma2, beta2 = self.film2(x, condition)
        if 2 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv3(x)
        x = self.activate(x)
        x, gamma3, beta3 = self.film3(x, condition)
        if 3 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        # extract features for domain classification
        if self.extract_layer == 3:
            features = x  
                
        x = self.conv4(x)
        x = self.activate(x)
        x, gamma4, beta4 = self.film4(x, condition)
        if 4 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        if self.extract_layer == 4:
            features = x
        
        x = self.conv5(x)    
        x = self.activate(x)
        x, gamma5, beta5 = self.film5(x, condition)
        if 5 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        # extract features for domain classification
        if self.extract_layer ==5:
            features = x 
        
        x = self.conv6(x)
        x = self.activate(x)
        x, gamma6, beta6 = self.film6(x, condition)
        if 6 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
        
        x = self.conv7(x)
        x, gamma7, beta7 = self.film7(x, condition)
        if 7 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
            
        # extract features for domain classification
        if self.extract_layer == 7:
            features = x 
            

        x = self.conv8(x)
        
        batch_size = x.shape[0]
        
        gammas = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7]
        betas = [beta1, beta2, beta3, beta4, beta5, beta6, beta7]
        
        gamma_cat = tf.concat([tf.reshape(g, [batch_size, -1]) for g in gammas], axis=-1)  # (batch, sum(n_i))
        beta_cat  = tf.concat([tf.reshape(b, [batch_size, -1]) for b in betas], axis=-1)   # (batch, sum(n_i))
        film_features = tf.concat([gamma_cat, beta_cat], axis=-1)  # (batch, sum(n_i), 2)
    
        domain_pred = self.domain_discriminator(self.grl(features), training=training)
        condition_domain_pred = self.condition_domain_discriminator(self.grl(film_features), training=training)

        return UDA_FiLM_Output(x, domain_pred, condition_domain_pred, features, film_features)
