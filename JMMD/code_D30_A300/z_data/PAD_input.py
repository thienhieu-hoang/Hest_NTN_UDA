"""
Script to analyze PAD (Proxy A-Distance) before and after FDA translation
Refactored with reusable functions for unscaled, scaled, and FDA versions
"""

import tensorflow as tf
import os
import sys
import numpy as np
from scipy.io import savemat
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA, PCA

# Add the root project directory
try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(code_dir, '..', '..', '..'))
except NameError:
    code_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(code_dir, '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Code dir:", code_dir)
print("Project root:", project_root)

from sklearn.decomposition import IncrementalPCA, PCA
from Domain_Adversarial.helper import loader, plotfig, PAD
from Domain_Adversarial.helper.utils import H5BatchLoader, minmaxScaler, complx2real, deMinMax
from Domain_Adversarial.helper.utils_GAN import visualize_H
from JMMD.helper.utils_GAN import save_checkpoint_jmmd as save_checkpoint

# ============================================================================
# CONFIGURATION
# ============================================================================

SNR_SOURCE = -5
SNR_TARGET = -5

source_data_file_path = os.path.abspath(os.path.join(
    code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_D_30_sim', 
    f'SNR_{SNR_SOURCE}dB', 'matlabNTN.mat'
))

target_data_file_path = os.path.abspath(os.path.join(
    code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_A_300_sim', 
    f'SNR_{SNR_TARGET}dB', 'matlabNTN.mat'
))

norm_approach = 'minmax'
lower_range = -1

batch_size = 8
pca_components_first = 96 # 2000
pca_components_second = 100
incremental_batch_size = 64
max_fitting_batches = 3

fda_win_h = 13
fda_win_w = 3

# Paths to save
path_temp = code_dir + f'/results/'
os.makedirs(path_temp, exist_ok=True)
idx_save_path = loader.find_incremental_filename(path_temp, 'ver', '_', '')
save_path = path_temp + f'ver{idx_save_path}_'
os.makedirs(save_path, exist_ok=True)

print(f"\nSaving results to: {save_path}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_fda_translation(x_src_real, x_tgt_real, win_h=13, win_w=3):
    """Apply FDA translation: mix source input with target style"""
    from JMMD.helper.utils_GAN import F_extract_DD, fda_mix_pixels, F_inverse_DD, apply_phase_to_amplitude
    
    # Convert to complex
    x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
    x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
    
    # Extract amplitude and phase
    src_amplitude, src_phase = F_extract_DD(x_src_complex)
    tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)
    
    # Mix amplitudes (low frequency from target, rest from source)
    mixed_amplitude = fda_mix_pixels(src_amplitude, tgt_amplitude, win_h, win_w)
    
    # Reconstruct with source phase
    mixed_complex_dd = apply_phase_to_amplitude(mixed_amplitude, src_phase)
    x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd)
    
    # Convert back to real
    x_fda_mixed = tf.stack(
        [tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], 
        axis=-1
    )
    
    return x_fda_mixed.numpy()


def apply_incremental_pca_and_save(data_source, data_target, h5_path_source, h5_path_target,
                                    batch_size=8, pca_components_first=2000, 
                                    incremental_batch_size=64, max_fitting_batches=3,
                                    data_name="data"):
    """
    Apply IncrementalPCA to source and target data and save to .h5 files
    Learn incrementally from ALL batches by stacking max_fitting_batches together
    
    Args:
        data_source: Source data array (N, H, W, C)
        data_target: Target data array (N, H, W, C)
        h5_path_source: Path to save source .h5 file
        h5_path_target: Path to save target .h5 file
        batch_size: Batch size for processing (8)
        pca_components_first: Number of PCA components (2000)
        incremental_batch_size: Batch size for IncrementalPCA (64)
        max_fitting_batches: Number of batches to stack together for each partial_fit (3)
        data_name: Name for logging
    
    Returns:
        pca_src, pca_tgt, explained_var_src, explained_var_tgt
    """
    
    print("\n" + "="*80)
    print(f"APPLYING INCREMENTAL PCA TO {data_name.upper()}")
    print("="*80)
    
    # Convert TensorFlow tensor to NumPy array
    data_source_np = data_source.numpy() if isinstance(data_source, tf.Tensor) else data_source
    data_target_np = data_target.numpy() if isinstance(data_target, tf.Tensor) else data_target

    # Flatten data
    N_samples_src = data_source_np.shape[0]
    N_samples_tgt = data_target_np.shape[0]
    original_dim = data_source_np.shape[1] * data_source_np.shape[2] * data_source_np.shape[3]
    
    data_source_flat = data_source_np.reshape(N_samples_src, -1)
    data_target_flat = data_target_np.reshape(N_samples_tgt, -1)
    
    print(f"Flattened source shape: {data_source_flat.shape}")
    print(f"Flattened target shape: {data_target_flat.shape}")
    print(f"Original dimension: {original_dim} → Target dimension: {pca_components_first}")
    
    # Initialize PCA
    pca_src = IncrementalPCA(n_components=pca_components_first, batch_size=incremental_batch_size)
    pca_tgt = IncrementalPCA(n_components=pca_components_first, batch_size=incremental_batch_size)
    
    # Create HDF5 files
    if os.path.exists(h5_path_source):
        os.remove(h5_path_source)
    features_h5_source = h5py.File(h5_path_source, 'w')
    features_dataset_source = None
    
    if os.path.exists(h5_path_target):
        os.remove(h5_path_target)
    features_h5_target = h5py.File(h5_path_target, 'w')
    features_dataset_target = None
    
    # Calculate total batches
    total_batches = int(np.ceil(N_samples_src / batch_size))
    
    # Need ≥ pca_components_first samples for INITIAL fit
    batches_needed_for_initial_fit = int(np.ceil(pca_components_first / batch_size))  # 250 batches = 2000 samples
    
    print(f"\nTotal batches: {total_batches}")
    print(f"Phase 1: Initial fit on {min(batches_needed_for_initial_fit, total_batches)} batches ({min(batches_needed_for_initial_fit, total_batches) * batch_size} samples)")
    print(f"Phase 2: Incremental fit (stacking {max_fitting_batches} batches) on ALL {total_batches} batches")
    print(f"  Each stack = {max_fitting_batches} batches × {batch_size} samples = {max_fitting_batches * batch_size} samples per partial_fit()")
    
    # ============ PHASE 1: COLLECT & FIT (Initialize PCA) ============
    print("\nPhase 1: Collecting batches for initial PCA fitting...")
    
    fitting_batches_src = []
    fitting_batches_tgt = []
    
    for batch_idx in range(min(batches_needed_for_initial_fit, total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N_samples_src)
        
        batch_src = data_source_flat[start_idx:end_idx]
        batch_tgt = data_target_flat[start_idx:end_idx]
        
        fitting_batches_src.append(batch_src)
        fitting_batches_tgt.append(batch_tgt)
        
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == min(batches_needed_for_initial_fit, total_batches):
            print(f"  Collected batch {batch_idx+1}/{min(batches_needed_for_initial_fit, total_batches)}")
    
    # FIT on all initial samples at once (initialization)
    print(f"\nInitializing IncrementalPCA on {len(fitting_batches_src)} batches ({len(fitting_batches_src) * batch_size} samples)...")
    fitting_data_src = np.vstack(fitting_batches_src)
    fitting_data_tgt = np.vstack(fitting_batches_tgt)
    
    pca_src.partial_fit(fitting_data_src)
    pca_tgt.partial_fit(fitting_data_tgt)
    
    explained_var_src = np.sum(pca_src.explained_variance_ratio_)
    explained_var_tgt = np.sum(pca_tgt.explained_variance_ratio_)
    print(f"✓ PCA initialized!")
    
    del fitting_batches_src, fitting_batches_tgt, fitting_data_src, fitting_data_tgt
    
    # ============ PHASE 2: INCREMENTAL FIT + TRANSFORM + SAVE (ALL BATCHES, stacked) ============
    print(f"\nPhase 2: Incremental fitting (stacking {max_fitting_batches} batches) + transforming on ALL {total_batches} batches...")
    
    batch_group_idx = 0
    group_count = 0
    
    while batch_group_idx < total_batches:
        # Collect max_fitting_batches batches
        batch_group_src = []
        batch_group_tgt = []
        batch_indices = []
        
        for offset in range(min(max_fitting_batches, total_batches - batch_group_idx)):
            batch_idx = batch_group_idx + offset
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N_samples_src)
            
            batch_src = data_source_flat[start_idx:end_idx]
            batch_tgt = data_target_flat[start_idx:end_idx]
            
            batch_group_src.append(batch_src)
            batch_group_tgt.append(batch_tgt)
            batch_indices.append(batch_idx)
        
        # Stack batches: (max_fitting_batches * batch_size, original_dim)
        # e.g., (3 * 8, 3696) = (24, 3696)
        batch_src_stacked = np.vstack(batch_group_src)
        batch_tgt_stacked = np.vstack(batch_group_tgt)
        
        # TRANSFORM with current PCA state
        batch_src_pca = pca_src.transform(batch_src_stacked)
        batch_tgt_pca = pca_tgt.transform(batch_tgt_stacked)
        
        # LEARN from this stacked batch (incremental refinement)
        pca_src.partial_fit(batch_src_stacked)
        pca_tgt.partial_fit(batch_tgt_stacked)
        
        # Update explained variance
        explained_var_src = np.sum(pca_src.explained_variance_ratio_)
        explained_var_tgt = np.sum(pca_tgt.explained_variance_ratio_)
        
        # Save to HDF5 (source)
        if features_dataset_source is None:
            features_dataset_source = features_h5_source.create_dataset(
                'features',
                data=batch_src_pca,
                maxshape=(None, pca_components_first),
                chunks=True,
                dtype='float32'
            )
        else:
            features_dataset_source.resize(features_dataset_source.shape[0] + batch_src_pca.shape[0], axis=0)
            features_dataset_source[-batch_src_pca.shape[0]:] = batch_src_pca
        
        # Save to HDF5 (target)
        if features_dataset_target is None:
            features_dataset_target = features_h5_target.create_dataset(
                'features',
                data=batch_tgt_pca,
                maxshape=(None, pca_components_first),
                chunks=True,
                dtype='float32'
            )
        else:
            features_dataset_target.resize(features_dataset_target.shape[0] + batch_tgt_pca.shape[0], axis=0)
            features_dataset_target[-batch_tgt_pca.shape[0]:] = batch_tgt_pca
        
        # Print progress
        group_count += 1
        batch_group_idx += max_fitting_batches
    

    # Close files
    features_h5_source.close()
    features_h5_target.close()
    
    return pca_src, pca_tgt, explained_var_src, explained_var_tgt

def load_and_calculate_pad(h5_path_source, h5_path_target, pca_components_second=100,
                            stage_name="UNSCALED"):
    """
    Load compressed features from .h5 files, combine, and calculate PAD
    
    Args:
        h5_path_source: Path to source .h5 file
        h5_path_target: Path to target .h5 file
        pca_components_second: Number of components for second PCA
        stage_name: Name for logging
    
    Returns:
        pad_svm, X_combined_2000d, y_combined
    """
    
    print("\n" + "="*80)
    print(f"LOADING & CALCULATING PAD - {stage_name}")
    print("="*80)
    
    # Load features
    print("\nLoading compressed features from .h5 files...")
    with h5py.File(h5_path_source, 'r') as f_src:
        X_source_2000d = f_src['features'][:]
        
    with h5py.File(h5_path_target, 'r') as f_tgt:
        X_target_2000d = f_tgt['features'][:]
        
    # Combine
    X_combined_2000d = np.vstack([X_source_2000d, X_target_2000d])
    y_combined = np.concatenate([
        np.zeros(len(X_source_2000d)),
        np.ones(len(X_target_2000d))
    ])

    
    # Calculate PAD
    print(f"\nCalculating PAD with second PCA ({pca_components_second}D) and SVM...")
    pad_svm = PAD.calc_pad_pca_svm(
        X_combined_2000d,
        y_combined,
        final_pca_components=pca_components_second,
        scale=True
    )
    
    print(f"\n✓ PAD (SVM) [{stage_name}]: {pad_svm:.6f}")
    
    return pad_svm, X_combined_2000d, y_combined


def plot_distributions(data_real, data_scaled=None, save_prefix="", stage_name=""):
    """
    Plot label/input distributions
    
    Args:
        data_real: Real data (unscaled)
        data_scaled: Scaled data (optional)
        save_prefix: Path prefix for saving figures
        stage_name: Name for plot title
    """
    
    if data_scaled is None:
        data_scaled = data_real
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Distribution Analysis: {stage_name}', fontsize=16, fontweight='bold')
    
    # (Implement plotting logic here - similar to original code)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved distributions to: {save_prefix}_distributions.png")
    plt.close()


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

source_file = h5py.File(source_data_file_path, 'r')
H_true_source = source_file['H_perfect']
H_input_source = source_file['H_li']  # H_li H_prac
N_samp_source = H_true_source.shape[0]

target_file = h5py.File(target_data_file_path, 'r')
H_true_target = target_file['H_perfect']
H_input_target = target_file['H_li']  # H_li H_prac
N_samp_target = H_true_target.shape[0]

print(f'N_samp_source = {N_samp_source}')
print(f'N_samp_target = {N_samp_target}')

# For testing
N_use_source = 96
N_use_target = 96

# For full analysis (uncomment)
# N_use_source = int(N_samp_source * 0.9)
# N_use_target = int(N_samp_target * 0.9)

print(f"Using {N_use_source} source and {N_use_target} target samples")

# Extract and convert data
H_true_source_use = H_true_source[:N_use_source]
H_input_source_use = H_input_source[:N_use_source]
H_true_target_use = H_true_target[:N_use_target]
H_input_target_use = H_input_target[:N_use_target]

H_input_source_real = complx2real(H_input_source_use)
H_input_source_real = np.transpose(H_input_source_real, (0, 2, 3, 1))

H_input_target_real = complx2real(H_input_target_use)
H_input_target_real = np.transpose(H_input_target_real, (0, 2, 3, 1))

H_true_source_real = complx2real(H_true_source_use)
H_true_source_real = np.transpose(H_true_source_real, (0, 2, 3, 1))

H_true_target_real = complx2real(H_true_target_use)
H_true_target_real = np.transpose(H_true_target_real, (0, 2, 3, 1))

print(f"H_input_source: {H_input_source_real.shape}")
print(f"H_input_target: {H_input_target_real.shape}")

# ============================================================================
# SCALE INPUTS
# ============================================================================

print("\n" + "="*80)
print("SCALING INPUTS TO [-1, 1]")
print("="*80)

H_input_source_scaled, x_min_src, x_max_src = minmaxScaler(
    H_input_source_real, lower_range=lower_range, linear_interp=False
)

H_input_target_scaled, x_min_tgt, x_max_tgt = minmaxScaler(
    H_input_target_real, lower_range=lower_range, linear_interp=False
)

print("✓ Scaling complete")

# ============================================================================
# STEP 1A: PAD UNSCALED RAW INPUTS
# ============================================================================

h5_path_source_unscaled = f'{save_path}/features_source_unscaled.h5'
h5_path_target_unscaled = f'{save_path}/features_target_unscaled.h5'

pca_src_unscaled, pca_tgt_unscaled, var_src_unscaled, var_tgt_unscaled = \
    apply_incremental_pca_and_save(
        H_input_source_real, H_input_target_real,
        h5_path_source_unscaled, h5_path_target_unscaled,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="UNSCALED"
    )

pad_svm_unscaled, X_unscaled, y_unscaled = load_and_calculate_pad(
    h5_path_source_unscaled, h5_path_target_unscaled,
    pca_components_second=pca_components_second,
    stage_name="UNSCALED"
)
# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_unscaled):
    os.remove(h5_path_source_unscaled)
    print(f"✓ Deleted: {h5_path_source_unscaled}")
if os.path.exists(h5_path_target_unscaled):
    os.remove(h5_path_target_unscaled)
    print(f"✓ Deleted: {h5_path_target_unscaled}")
    
# ============================================================================
# STEP 1B: PAD SCALED INPUTS
# ============================================================================

h5_path_source_scaled = f'{save_path}/features_source_scaled.h5'
h5_path_target_scaled = f'{save_path}/features_target_scaled.h5'

pca_src_scaled, pca_tgt_scaled, var_src_scaled, var_tgt_scaled = \
    apply_incremental_pca_and_save(
        H_input_source_scaled, H_input_target_scaled,
        h5_path_source_scaled, h5_path_target_scaled,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="SCALED"
    )

pad_svm_scaled, X_scaled, y_scaled = load_and_calculate_pad(
    h5_path_source_scaled, h5_path_target_scaled,
    pca_components_second=pca_components_second,
    stage_name="SCALED"
)
    
# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_scaled):
    os.remove(h5_path_source_scaled)
    print(f"✓ Deleted: {h5_path_source_scaled}")
if os.path.exists(h5_path_target_scaled):
    os.remove(h5_path_target_scaled)
    print(f"✓ Deleted: {h5_path_target_scaled}")

# ============================================================================
# STEP 2: FDA TRANSLATION (on original unscaled inputs)
# ============================================================================

print("\n" + "="*80)
print("APPLYING FDA TRANSLATION")
print("="*80)

print(f"\nApplying FDA with window ({fda_win_h}, {fda_win_w})...")
H_input_source_fda = []

for i in range(N_use_source):
    if (i + 1) % 20 == 0:
        print(f"  Processing sample {i+1}/{N_use_source}")
    
    tgt_idx = i % N_use_target
    x_fda = apply_fda_translation(
        H_input_source_real[i:i+1],
        H_input_target_real[tgt_idx:tgt_idx+1],
        win_h=fda_win_h,
        win_w=fda_win_w
    )
    H_input_source_fda.append(x_fda[0])

H_input_source_fda = np.array(H_input_source_fda)
print(f"✓ FDA shape: {H_input_source_fda.shape}")

# ============================================================================
# STEP 2A: FDA FOR FDA unscaled inputs
# ============================================================================

print("\nCalculating PAD for FDA-translated unscaled inputs...")

h5_path_source_fda = f'{save_path}/features_source_fda.h5'
h5_path_target_fda = f'{save_path}/features_target_fda.h5'

pca_src_fda, pca_tgt_fda, var_src_fda, var_tgt_fda = \
    apply_incremental_pca_and_save(
        H_input_source_fda, H_input_target_real,
        h5_path_source_fda, h5_path_target_fda,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="FDA_UNSCALED"
    )

pad_svm_fda, X_fda, y_fda = load_and_calculate_pad(
    h5_path_source_fda, h5_path_target_fda,
    pca_components_second=pca_components_second,
    stage_name="FDA_UNSCALED"
)

# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_fda):
    os.remove(h5_path_source_fda)
    print(f"✓ Deleted: {h5_path_source_fda}")
if os.path.exists(h5_path_target_fda):
    os.remove(h5_path_target_fda)
    print(f"✓ Deleted: {h5_path_target_fda}")

# ============================================================================
# STEP 2B: FDA FOR FDA scaled inputs
# ============================================================================
print("\nCalculating PAD for FDA-translated scaled inputs...")

h5_path_source_fda = f'{save_path}/features_source_fda.h5'
h5_path_target_fda = f'{save_path}/features_target_fda.h5'

H_input_source_scaled, x_min_src, x_max_src = minmaxScaler(
    H_input_source_real, lower_range=lower_range, linear_interp=False
)

H_input_target_scaled, x_min_tgt, x_max_tgt = minmaxScaler(
    H_input_target_real, lower_range=lower_range, linear_interp=False
)

pca_src_fda, pca_tgt_fda, var_src_fda, var_tgt_fda = \
    apply_incremental_pca_and_save(
        H_input_source_scaled, H_input_target_scaled,
        h5_path_source_fda, h5_path_target_fda,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="FDA_SCALED"
    )

pad_svm_fda, X_fda, y_fda = load_and_calculate_pad(
    h5_path_source_fda, h5_path_target_fda,
    pca_components_second=pca_components_second,
    stage_name="FDA_SCALED"
)

# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_fda):
    os.remove(h5_path_source_fda)
    print(f"✓ Deleted: {h5_path_source_fda}")
if os.path.exists(h5_path_target_fda):
    os.remove(h5_path_target_fda)
    print(f"✓ Deleted: {h5_path_target_fda}")