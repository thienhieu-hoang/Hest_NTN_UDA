"""helpers for general purposes: 
"""
import h5py
import tensorflow as tf
from tensorflow.keras.utils import Sequence

import os
import numpy as np
import scipy.io

import datetime

class H5BatchLoader:
    def __init__(self, file, dataset_name, batch_size, shuffled_indices=None):
        self.file = file
        self.data = self.file[dataset_name]
        self.batch_size = batch_size
        self.total_samples = shuffled_indices.shape[0]
        
        # Use pre-shuffled indices if provided, otherwise not shuffle them
        self.indexes = shuffled_indices if shuffled_indices is not None else np.arange(self.total_samples)
        
        self.total_batches = self.total_samples // self.batch_size
        self.current_batch = 0

    def next_batch(self):
        if self.current_batch >= self.total_batches:
            raise StopIteration("All batches have been loaded.")
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indexes[start:end]
        
        # Sort indices and get the order to unsort later
        sorted_indices = np.sort(batch_indices)
        original_order = np.argsort(np.argsort(batch_indices))
        # Fetch sorted data from HDF5
        batch_sorted = self.data[sorted_indices]
        # Reorder to match original (shuffled) order
        batch = batch_sorted[original_order]
        
        self.current_batch += 1
        return batch

    def reset(self):
        self.current_batch = 0

from functools import partial

def clip_sample(sample, a, b, c, d):
    # Extract the real and imaginary parts
    real = sample[:, :, 0]
    imag = sample[:, :, 1]

    # Compute min/max over the region of interest for real
    roi_real = real[a:b, c:d]
    min_real = tf.reduce_min(roi_real)
    max_real = tf.reduce_max(roi_real)

    # Compute min/max over the region of interest for imag
    roi_imag = imag[a:b, c:d]
    min_imag = tf.reduce_min(roi_imag)
    max_imag = tf.reduce_max(roi_imag)

    # Clip full real and imag parts
    real_clipped = tf.clip_by_value(real, min_real, max_real)
    imag_clipped = tf.clip_by_value(imag, min_imag, max_imag)

    # Stack back the two channels (real, imag)
    return tf.stack([real_clipped, imag_clipped], axis=-1)

def minmaxScaler(x, min_pre=None, max_pre=None, lower_range = -1, linear_interp=False):
    # lower_range = -1 -- scale to [-1 1] range
    # lower_range =  1 -- scale to [0 1] range
    # x == tensor [Nsamples, n_subcs, n_symb, 2]
    # min_pre, max_pre == None or [Nsamples, 2] -- predefined min, max range to scale 
# return 
    # x_scaled = tensor, size(x)  [Nsamples, n_subcs, n_symb, 2]
    # x_min, x_max = [Nsamples, 2] -- min, max of real and imag, of each sample 
# if linear interpolation: clip values that go beyond the estimated pilot
    
    x_shape = tf.shape(x)
    N = x_shape[0] # Nsamples
    
    if linear_interp:
        if x_shape[1] == 312:
            clip_fn = partial(clip_sample, a=1, b=309, c=3, d=12)
            # Apply over all samples
            x = tf.map_fn(clip_fn, x)
    
    # Flatten last two dimensions for min/max computation
    x_reshaped = tf.reshape(x, [N, -1, 2])  # [N, n_subcs * n_symb, 2]
    
    if min_pre is not None and max_pre is not None:
        x_min = min_pre  # [N, 2]
        x_max = max_pre
    else:
        x_min = tf.reduce_min(x_reshaped, axis=1)  # [N, 2]
        x_max = tf.reduce_max(x_reshaped, axis=1)  # [N, 2]
    
    scale = tf.clip_by_value(x_max - x_min, 1e-8, tf.float32.max)  # avoid divide-by-zero # [N, 2]

    # Reshape for broadcasting
    x_min_broadcast = tf.reshape(x_min, [N, 1, 1, 2])
    scale_broadcast = tf.reshape(scale, [N, 1, 1, 2])
    
    # Normalize
    x_scaled = (x - x_min_broadcast) / scale_broadcast  # [N, n_subcs, n_symb, 2]

    if lower_range == -1:
        x_scaled = x_scaled * 2.0 - 1.0
    elif lower_range == 0:
        x_scaled = x_scaled
        
    return x_scaled, x_min, x_max

def deMinMax(x_normd, x_min, x_max, lower_range=-1):
    # x_normd size is [Nsamples, sub, symb, 2] (real, imag) 
    #              or [Nsamples, sub, symb, 1] (complex)
    # x_max, x_min == torch [Nsamples, 2] -- min, max of real and imag, of each sample
    x_normd = tf.convert_to_tensor(x_normd, dtype=tf.float32)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)

    if lower_range ==-1:
        # scale [-1, 1] -> original
        scale = (x_max - x_min) / 2.0
        shift = (x_max + x_min) / 2.0
        x_denormed = x_normd * scale[:, None, None, :] + shift[:, None, None, :]
            
    elif lower_range ==0:
        if len(x_max.shape) == 1:
            # x_max, x_min shape: [N] # and x_normd size is [Nsamples, sub, symb, 1] (complex)
            scale = x_max - x_min
            x_denormed = x_normd * scale[:, None, None, :] + x_min[:, None, None, :]
        elif len(x_max.shape) == 2:
            # x_max, x_min shape: [N, 2] and x_normd size is [Nsamples, sub, symb, 2] (real, imag)
            scale = x_max - x_min
            x_denormed = x_normd * scale[:, None, None, :] + x_min[:, None, None, :]

                
    return  x_denormed

def complx2real(H_struct):
    """
    Convert structured complex ndarray of shape [N, S, T] with dtype [('real', float), ('imag', float)]
    into a float tensor of shape [N, 2, S, T], separating real and imaginary parts.
    """
    real = H_struct['real'].astype(np.float32)  # shape: (Nsamp, n_subcs, n_symbs)
    imag = H_struct['imag'].astype(np.float32)  # shape: (Nsamp, n_subcs, n_symbs)
    
    # Stack along new dimension (axis=1) => shape: (Nsamp, 2, n_subcs, n_symbs)
    combined = np.stack([real, imag], axis=1)
    
    # Convert to TensorFlow tensor
    return tf.convert_to_tensor(combined, dtype=tf.float32)

def val_step(model, val_loader, criterion, epoch, num_epochs, H_NN_val):
    running_val_loss = 0.0
    i = 0

    for val_inputs, val_targets, val_targetsMin, val_targetsMax in val_loader:
        val_inputs_real = tf.expand_dims(val_inputs[:, 0, :, :], axis=1)   # shape: [batch, 1, 612, 14]
        val_inputs_imag = tf.expand_dims(val_inputs[:, 1, :, :], axis=1)
        val_targets_real = tf.expand_dims(val_targets[:, 0, :, :], axis=1)
        val_targets_imag = tf.expand_dims(val_targets[:, 1, :, :], axis=1)

        # No need to call model.eval() in TF — just avoid gradient tape
        val_outputs_real = model(val_inputs_real, training=False)
        val_loss_real = criterion(val_targets_real, val_outputs_real)
        running_val_loss += float(val_loss_real)

        val_outputs_imag = model(val_inputs_imag, training=False)
        val_loss_imag = criterion(val_targets_imag, val_outputs_imag)
        running_val_loss += float(val_loss_imag)

        if epoch == num_epochs - 1:
            batch_size = tf.shape(val_outputs_real)[0]
            H_NN_val[i:i+batch_size, 0, :, :].assign(tf.squeeze(val_outputs_real, axis=1))
            H_NN_val[i:i+batch_size, 1, :, :].assign(tf.squeeze(val_outputs_imag, axis=1))
            i += batch_size

    avg_val_loss = running_val_loss / (2 * len(val_loader))
    return avg_val_loss, H_NN_val
    
def calNMSE(x, target, return_mse=False):
    # x, target: shape [batch, 612, 14], complex dtype
    x = tf.convert_to_tensor(x)
    target = tf.convert_to_tensor(target)
    
    NMSE_array = tf.TensorArray(dtype=tf.float32, size=x.shape[0])
    MSE_array = tf.TensorArray(dtype=tf.float32, size=x.shape[0])

    for i in tf.range(x.shape[0]):
        target_i = target[i, :, :]
        x_i = x[i, :, :]
        target_power = tf.reduce_mean(tf.abs(target_i) ** 2)
        mse_i = tf.reduce_mean(tf.abs(x_i - target_i) ** 2)
        NMSE_array = NMSE_array.write(i, mse_i / target_power)
        MSE_array = MSE_array.write(i, mse_i)

    NMSE_result = NMSE_array.stack()
    MSE_result = MSE_array.stack()

    return (NMSE_result, MSE_result) if return_mse else NMSE_result
    
    
# def standardize(x):
#     # x == torch [Nsamples, 2, 612, 14]
# # return 
#     # x_normd = torch, size(x)  [Nsamples, 2, 612, 14]
#     # x_mean, x_var = [Nsamples, 2] -- mean, var of real and imag, of each sample 
#     x_mean = torch.empty((0,2)).to(x.device)
#     x_var  = torch.empty((0,2)).to(x.device)
#     x_normd = torch.empty(x.shape)
#     for i in range(x.shape[0]):
#         sample_real = x[i,0,:,:] # [2,612,14]
#         sample_imag = x[i,1,:,:]
        
#         # Compute mean and variance for the current sample
#         mean   = torch.stack((sample_real.mean(), sample_imag.mean()))    # tensor[-1e-5 , -1e-5] device=cuda
#         variance = torch.stack((sample_real.var(), sample_imag.var()))    # tensor[-1e-5 , -1e-5] device=cuda
        
#         x_normd[i,0,:,:] = (sample_real - mean[0]) / np.sqrt(variance[0])
#         x_normd[i,1,:,:] = (sample_imag - mean[1]) / np.sqrt(variance[1])
        
#         x_mean = torch.cat((x_mean,    mean.unsqueeze(0)), dim=0)
#         x_var  = torch.cat((x_var, variance.unsqueeze(0)), dim=0)
#     return x_normd, x_mean, x_var

# def deSTD(x_normd, mean, var):
#     # x_normd = torch, size(x)  [Nsamples, 2, 612, 14]
#     # mean, var = [Nsamples, 2] -- mean, var of real and imag, of each sample 
#     x_denormd = torch.empty(x_normd.shape)
#     for i in range(x_normd.shape[0]):
#         x_denormd[i,0,:,:] = x_normd[i,0,:,:]* np.sqrt(var[i,0]) + mean[i,0]
#         x_denormd[i,1,:,:] = x_normd[i,1,:,:]* np.sqrt(var[i,1]) + mean[i,1]
        
#     return x_denormd  

# def deNorm(x_normd, x_1, x_2, norm_approach, lower_range=-1):
#     # lower_range used in case of deMinMax
#     # norm_approach = 'minmax' or 'std' 
#     if norm_approach == 'minmax':
#         x_denormd = deMinMax(x_normd, x_1, x_2, lower_range)
#                             # x_1 -- x_min
#                             # x_2 -- x_max
#     elif norm_approach == 'std':
#         x_denormd = deSTD(x_normd, x_1, x_2)
#                         # x_1 == x_mean
#                         # x_2 == x_var
#     elif norm_approach == 'no':
#         x_denormd = x_normd
        
#     return x_denormd