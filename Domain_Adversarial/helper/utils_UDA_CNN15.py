"""helpers for CNN model: 
    input shape: [batch_size, 2, 792, 1+14] = [batch_size, 2, nsubcs, 1+nsymbs]
                concatenate raw-estimated channels at symbol 2 of slot 1 and slot 6
    Output shape: [batch_size, 2, 792, 14] = [batch_size, 2, nsubcs, nsymbs] 
                CNN estimated channels at slot 6
"""
import sys
import os
notebook_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', '..')))
import Est_btween_CSIRS.helper.utils_CNN15 as utils_CNN15
import Est_btween_CSIRS.helper.utils as utils_CNN

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

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
    def __init__(self, in_channels=128, disc='dense', first_layer='pool'):
        # disc: 'dense' or 'conv'
        # first_layer: 'pool' or 'flatten'
        super(DomainDiscriminator, self).__init__()
        self.disc = disc
        self.first_layer = first_layer
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
                x = self.pool(x)
            elif self.first_layer == 'flatten':
                x = self.flatten(x)
                self.fc0(x)
            x = self.fc1(x)
            x = self.fc2(x)
        elif self.disc == 'conv':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
        return self.out(x)    

class UDA_CNN(tf.keras.Model):
    
    def __init__(self, dropOut=0, act='ReLU', dropOutPos=[2, 4], extractLayer=3, disc='dense', first_layer_disc='pool'):
        super(UDA_CNN, self).__init__()

        self.extractLayer = extractLayer  # layer to extract features for domain classification
        
        self.dropOut = dropOut
        self.dropOutPos = dropOutPos

        self.normalization = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(32, (7, 3), strides=(1, 1), padding='valid')

        self.conv2 = tf.keras.layers.Conv2D(64, (5, 4), strides=(1, 1), padding='valid')

        self.conv3 = tf.keras.layers.Conv2D(128, (5, 3), strides=(1, 1), padding='valid')

        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='valid')

        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid')

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
        if self.extractLayer == 3:
            self.domain_discriminator = DomainDiscriminator(in_channels=128, disc=disc, first_layer=first_layer_disc)
        if self.extractLayer == 5:
            self.domain_discriminator = DomainDiscriminator(in_channels=64, disc=disc, first_layer=first_layer_disc)

    def padLayer(self, x, top, bottom, left, right, mode='REFLECT'):  # mode can be 'REFLECT' or 'SYMMETRIC' or 'CONSTANT'
        return tf.pad(x, [[0, 0], [top, bottom], [left, right], [0, 0]], mode=mode)

    
    def call(self, x, training=False, return_domain=False):
        x = self.normalization(x, training=training)

        if 0 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv1(self.padLayer(x, 3, 3, 1, 1))
        x = self.activate(x)

        if 1 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv2(self.padLayer(x, 2, 2, 1, 1))
        x = self.activate(x)

        if 2 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv3(self.padLayer(x, 2, 2, 1, 1))
        x = self.activate(x)

        if 3 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv4(self.padLayer(x, 1, 1, 1, 1))
        x = self.activate(x)
        
        # extract features for domain classification
        if self.extractLayer == 3:
            features = x

        if 4 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)

        x = self.conv5(self.padLayer(x, 1, 1, 1, 1))

        if 5 in self.dropOutPos and self.dropOut:
            x = self.dropout(x, training=training)
            
        # extract features for domain classification
        if self.extractLayer == 5:
            features = x

        x = self.conv6(self.padLayer(x, 1, 1, 1, 1))
        
        if return_domain:
            domain_pred = self.domain_discriminator(self.grl(features), training=training)
            return x, domain_pred  # output, domain prediction

        return x
    
def prepare_data(loader_H_1_6_11, nsymb=14):
    return utils_CNN15.prepare_data(loader_H_1_6_11, nsymb)
    
# Domain labels
def make_domain_labels(batch_size, domain):
    return tf.ones((batch_size, 1)) if domain == 'source' else tf.zeros((batch_size, 1))
    
def train_step(model, loader_H, loss_fn, lower_range, nsymb=14, lambda_domain=0.1):
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
    
    # Train for 1 epoch
    # Iterate over batches
    for batch_idx in range(loader_H_true_train.total_batches):
        if batch_idx % 20 == 0 or batch_idx == loader_H_true_train.total_batches - 1:
            print(f"batch_idx: {batch_idx+1}/{loader_H_true_train.total_batches}")
        x = loader_H_input_train.next_batch()  
        x = prepare_data(x, nsymb=nsymb) # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y = loader_H_true_train.next_batch()  # Label: H_true
        y = y[:,:,70:84]  # only consider 14 symbols of  slot 6
        N_train_source += x.shape[0]
        #
        x_target = loader_H_train_target.next_batch()  # Target domain data
        x_target = prepare_data(x_target, nsymb=nsymb)  # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y_target = loader_H_true_target.next_batch()  # Target domain label - just for monitoring
        y_target = y_target[:,:,70:84]
        N_train_target += x_target.shape[0]
    
        # Ensure tensors and float32 dtype
        x = utils_CNN.complx2real(x)  # (batch_size, 2, nsubcs, nsymbs)
        y = utils_CNN.complx2real(y)  
        #
        x_target = utils_CNN.complx2real(x_target)  # (batch_size, 2, nsubcs, nsymbs)
        y_target = utils_CNN.complx2real(y_target)
        
        x = np.transpose(x, (0, 2, 3, 1)) # (batch_size, nsubcs, nsymbs, 2)
        y = np.transpose(y, (0, 2, 3, 1)) 
        x_target = np.transpose(x_target, (0, 2, 3, 1))
        y_target = np.transpose(y_target, (0, 2, 3, 1))
        
        # min_max scale 
        x_scaled, _, _ = utils_CNN.minmaxScaler(x, lower_range=lower_range) # scale to [lower_range, 1]   # x_scaled, x_min, x_max
        y_scaled, _, _ = utils_CNN.minmaxScaler(y, lower_range=lower_range)
        x_target_scaled, _, _ = utils_CNN.minmaxScaler(x_target, lower_range=lower_range) # scale to [lower_range, 1]
        y_target_scaled, _, _ = utils_CNN.minmaxScaler(y_target, lower_range=lower_range)

        # GradientTape for automatic differentiation
        with tf.GradientTape() as tape:
            predictions_source, source_domain_pred = model(x_scaled, training=True, return_domain=True)         
            loss_est = loss_fn_ce(y_scaled, predictions_source)

            # Forward pass on target domain (only for domain)
            predictions_target, target_domain_pred = model(x_target_scaled, training=True, return_domain=True)

            # Domain labels
            source_domain_labels = make_domain_labels(tf.shape(x_scaled)[0], 'source')
            target_domain_labels = make_domain_labels(tf.shape(x_target_scaled)[0], 'target')
            
            # Domain classification loss (on both)
            domain_loss_source = loss_fn_domain(source_domain_labels, source_domain_pred)
            domain_loss_target = loss_fn_domain(target_domain_labels, target_domain_pred)
            domain_loss = domain_loss_source + domain_loss_target
            
            # Total loss: Estimation + λ * Domain Loss
            # lambda_domain = 0.1  # weight for domain loss
            total_loss = loss_est + lambda_domain * domain_loss
            
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
        
    avg_epoc_loss_est        = epoc_loss_est / N_train_source
    avg_epoc_loss_domain     = epoc_loss_domain / (N_train_source + N_train_target)
    avg_epoc_loss            = epoc_loss / (N_train_source + N_train_target)
    avg_epoc_loss_est_target = epoc_loss_est_target / N_train_target
        
    return avg_epoc_loss_est, avg_epoc_loss_domain, avg_epoc_loss, avg_epoc_loss_est_target    # x == (batch_size, nsubcs, nsymbs, 2) real matrix


def val_step(model, loader_H, loss_fn, lower_range, epoch=None, nsymb=14, lambda_domain=0.1):
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
        x = prepare_data(x, nsymb= nsymb) # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y = loader_H_true_val_source.next_batch()
        y = y[:, :, 70:84]
        #
        x_target = loader_H_input_val_target.next_batch()
        x_target = prepare_data(x_target, nsymb=nsymb)  # Input: concatenated of (nsample, nsubcs,1) (row 2 of slot 1) to (nsample, nsubcs, nsymbs) (duplicate of plot 6)
        y_target = loader_H_true_val_target.next_batch()
        y_target = y_target[:, :, 70:84]
        #
        N_val_source += x.shape[0]
        N_val_target += x_target.shape[0]
        
        x_real = utils_CNN.complx2real(x)   # Tensor [nbatch, 2, nsubcs, nsymbs+1]
        y_real = utils_CNN.complx2real(y)
        x_real = np.transpose(x_real, (0, 2, 3, 1)) # Tensor [nbatch, nsubcs, nsymbs+1, 2] --- # to fit CNN size
        y_real = np.transpose(y_real, (0, 2, 3, 1))
        #
        x_target_real = utils_CNN.complx2real(x_target)   # Tensor [nbatch, 2, nsubcs, nsymbs+1]
        y_target_real = utils_CNN.complx2real(y_target)
        x_target_real = np.transpose(x_target_real, (0, 2, 3, 1))
        y_target_real = np.transpose(y_target_real, (0, 2, 3, 1))

        if (epoch is not None) and (idx == 0):
            H_true_sample  = y_real[:3,:,:,:].copy() # [3, nsubcs, nsymbs, 2]
            H_input_sample = x_real[:3,:,1:,:].copy() # [3, nsubcs, nsymbs, 2] -- to calculate NMSE and visualization
            H_input_condition = x_real[:3,:,:,:].copy() # [3, nsubcs, 1, 2] -- the CSI-RS of slot 1 -- for visualization
                                # of sample 0,1,2
            H_true_sample_target  = y_target_real[:3,:,:,:].copy() # [3, nsubcs, nsymbs, 2]
            H_input_sample_target  = x_target_real[:3,:,1:,:].copy() # [3, nsubcs, nsymbs, 2] -- to calculate NMSE and visualization
            H_input_condition_target  = x_target_real[:3,:,:,:].copy() # [3, nsubcs, 1, 2] -- the CSI-RS of slot 1 -- for visualization
            nmse_input_source = 0
            nmse_est_source   = 0
            nmse_input_target = 0
            nmse_est_target   = 0

        # ======= source domain ========
        x_scaled, x_min, x_max = utils_CNN.minmaxScaler(x_real, lower_range=lower_range)
        y_scaled, _, _ = utils_CNN.minmaxScaler(y_real, lower_range=lower_range)
        #
        preds, domain_preds = model(x_scaled, training=False, return_domain=True)
        preds_descaled = utils_CNN.deMinMax(preds, x_min, x_max, lower_range=lower_range)
        if (epoch is not None) and (idx == 0):
            H_est_sample = tf.identity(preds_descaled[:3,:,:,:]) # [3,nsubcs, nsymbs, 2]
            mse_sample_source = np.mean((H_est_sample - H_true_sample) ** 2, axis=(1, 2, 3)) # shape (3,)
            power_sample_source = np.mean(H_true_sample ** 2, axis=(1, 2, 3))
            nmse_est_source = mse_sample_source / (power_sample_source + 1e-30)  # avoid divide-by-zero
            #
            mse_input_source = np.mean((H_input_sample - H_true_sample) ** 2, axis=(1, 2, 3))
            nmse_input_source = mse_input_source / (power_sample_source + 1e-30)    # shape (3,)
            domain_sample_return_source = ((domain_preds[:3] >= 0.5).numpy()).astype(np.int32)
            
        # ======= target domain ========
        x_scaled_target, x_min_target, x_max_target = utils_CNN.minmaxScaler(x_target_real, lower_range=lower_range)
        y_scaled_target, _, _ = utils_CNN.minmaxScaler(y_target_real, lower_range=lower_range)
        #
        preds_target, domain_preds_target = model(x_scaled_target, training=False, return_domain=True)
        preds_descaled_target = utils_CNN.deMinMax(preds_target, x_min_target, x_max_target, lower_range=lower_range)
        #
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